require(tidyverse)
require(arrow)  # to read parquet
library(dplyr)
library(ggplot2)

# --- Original Helper Functions ---
is_correct <- function(df, is_table_lock=FALSE) {
  if (is_table_lock) {
    correct <- (df$gt_table == df$pred_table)
  } else {
    correct <- (df$gt_table == df$pred_table) &
      (df$gt_pageid == df$pred_pageid) # Row locks require both table and pageid to be correct
    # we can remove the page_id columns now to save memory:
    df$gt_pageid <- NULL
    df$pred_pageid <- NULL
  }
  correct[is.na(correct)] <- FALSE  # Explicitly set NA results to FALSE
  # NA's occur when the prediction output token length is less than the ground truth token length
  return(correct)
}

load_parquet <- function(path, is_table_lock=FALSE, filter_tail_ns=3e11) {
  predictions <- read_parquet(path, col_select = c("in_lock_sequences_id", "in_lock_start_time", "data", "horizon", "unique_id", "horizon_position", "gt_table", "gt_pageid", "pred_table", "pred_pageid", "iteration")) %>% 
      filter(iteration <= 10 & gt_table != "warehouse")
  predictions$is_correct <- is_correct(predictions, is_table_lock)
  predictions$horizon <- as.factor(predictions$horizon)
  
  if (filter_tail_ns > 0) {
    predictions <- predictions %>%
      filter(in_lock_start_time < max(in_lock_start_time) - filter_tail_ns)
  }
  
  return(predictions)
}

check_iterations <- function(df) {
  horizon_iteration_counts <- df %>%
    group_by(horizon, iteration) %>%
    summarise(n = n(), .groups='drop') %>%
    group_by(horizon) %>%
    summarise(n = n(), .groups='drop')

  if (any(horizon_iteration_counts$n < 10)) {
    print(horizon_iteration_counts)
    stop("Some horizons have less than 10 iterations")
  }
  else {
    print("All horizons have at least 10 iterations")
  }
}

horizon_iteration_performance <- function(predictions) {
  correct <- predictions %>%
    group_by(horizon, iteration, unique_id) %>%
    summarise(is_correct = all(is_correct), .groups='drop') %>%
    group_by(horizon, iteration) %>%
    summarise(mean_percent_correct = mean(is_correct), .groups='drop')

  print(correct %>%
    group_by(horizon) %>%
    summarise(percent_correct = mean(mean_percent_correct), .groups='drop'))
  
  return(correct)
}

# --- Modified horizon_iteration_performance_by_table ---
horizon_iteration_performance_by_table <- function(predictions) {
  # Calculate overall correctness for each prediction sequence (unique_id) within each horizon, iteration, and ground truth table.
  correct_by_table <- predictions %>%
    group_by(horizon, iteration, unique_id, gt_table) %>% 
    summarise(is_correct = all(is_correct), .groups='drop') %>% 
    group_by(horizon, iteration, gt_table) %>% 
    summarise(mean_percent_correct = mean(is_correct), .groups='drop') 

  print(correct_by_table %>%
    group_by(horizon, gt_table) %>%
    summarise(percent_correct = mean(mean_percent_correct), .groups='drop'))
  
  # Count occurrences of each combination of horizon, iteration, ground truth table, and predicted table.
  conf <- predictions %>%
    group_by(horizon, iteration, gt_table, pred_table) %>%
    summarise(n = n(), .groups='drop') 
  
  all_tables <- union(conf$gt_table, conf$pred_table)
  all_tables <- all_tables[!is.na(all_tables)] # Ensure NA is not in all_tables

  # Create a data frame with all possible combinations of horizon, iteration, and table names.
  all_combos <- conf %>%
    distinct(horizon, iteration) %>% 
    tidyr::crossing(table = all_tables) 
  
  metrics <- all_combos %>%
    # 1) True Positives (TP): Ground truth table and predicted table are the same as the current 'table'.
    left_join(
      conf %>%
        filter(gt_table == pred_table) %>% 
        rename(table = gt_table) %>% 
        select(horizon, iteration, table, n),
      by = c("horizon", "iteration", "table")
    ) %>%
    rename(tp = n) %>% 
    mutate(tp = tidyr::replace_na(tp, 0)) %>%
    
    # 2) False Positives (FP): Predicted table is the current 'table', but ground truth table is different.
    left_join(
      conf %>%
        group_by(horizon, iteration, pred_table) %>% 
        summarise(fp = sum(ifelse(gt_table != pred_table, n, 0)), 
                  .groups='drop') %>%
        rename(table = pred_table),
      by = c("horizon", "iteration", "table")
    ) %>%
    mutate(fp = tidyr::replace_na(fp, 0)) %>%
    
    # 3) False Negatives (FN): Ground truth table is the current 'table', but predicted table is different.
    left_join(
      conf %>%
        group_by(horizon, iteration, gt_table) %>% 
        summarise(fn = sum(ifelse(gt_table != pred_table, n, 0)), 
                  .groups='drop') %>%
        rename(table = gt_table),
      by = c("horizon", "iteration", "table")
    ) %>%
    mutate(fn = tidyr::replace_na(fn, 0)) %>%
    
    # 4) True Negatives (TN): Ground truth table is NOT 'table' AND predicted table is also NOT 'table'.
    left_join(
      conf %>%
        tidyr::crossing(table_to_eval = all_tables) %>%
        filter(gt_table != table_to_eval & pred_table != table_to_eval) %>%
        group_by(horizon, iteration, table_to_eval) %>%
        summarise(tn = sum(n, na.rm = TRUE), .groups = 'drop') %>%
        rename(table = table_to_eval),
      by = c("horizon", "iteration", "table")
    ) %>%
    mutate(tn = tidyr::replace_na(tn, 0)) %>%
    
    # Compute Precision, Recall, and F1 Score from TP, FP, FN.
    mutate(
      precision = ifelse(tp + fp == 0, NA, tp / (tp + fp)), 
      recall    = ifelse(tp + fn == 0, NA, tp / (tp + fn)), 
      f1        = ifelse(
        !is.na(precision) & !is.na(recall) & (precision + recall > 0), 
        2 * precision * recall / (precision + recall), 
        NA 
      )
    )  %>%
    rename(gt_table = table) # Rename 'table' to 'gt_table' to match `correct_by_table`
  
  return(correct_by_table %>% left_join(metrics, by = c("horizon", "iteration", "gt_table")))
}


export_csv <- function(df, path) {
  df %>%
    group_by(horizon) %>%
    summarise(
      mean_percent_correct_csv = mean(mean_percent_correct),
      median_percent_correct_csv = median(mean_percent_correct),
      .groups='drop'
    ) %>%
    write_csv(path)
}

# --- Modified export_csv_by_table ---
export_csv_by_table <- function(df, path) {
  df %>%
    group_by(horizon, gt_table) %>%
    summarise(
      mean_percent_correct_csv = mean(mean_percent_correct, na.rm = TRUE),
      median_percent_correct_csv = median(mean_percent_correct, na.rm = TRUE),
      mean_tp_csv = mean(tp, na.rm = TRUE),
      median_tp_csv = median(tp, na.rm = TRUE),
      mean_fp_csv = mean(fp, na.rm = TRUE),
      median_fp_csv = median(fp, na.rm = TRUE),
      mean_fn_csv = mean(fn, na.rm = TRUE),
      median_fn_csv = median(fn, na.rm = TRUE),
      mean_tn_csv = mean(tn, na.rm = TRUE),
      median_tn_csv = median(tn, na.rm = TRUE),
      mean_precision_csv = mean(precision, na.rm = TRUE),
      median_precision_csv = median(precision, na.rm = TRUE),
      mean_recall_csv    = mean(recall, na.rm = TRUE),
      median_recall_csv = median(recall, na.rm = TRUE),
      mean_f1_csv        = mean(f1, na.rm = TRUE),
      median_f1_csv   = median(f1, na.rm = TRUE),
      .groups='drop'
    ) %>%
    write_csv(path)

  # Write the raw data to a separate CSV file without summarization
  raw_data_path <- gsub("\\.csv$", "_raw.csv", path)
  df %>%
    select(horizon, iteration, gt_table, tp, fp, fn, tn, precision, recall, f1) %>%
    write_csv(raw_data_path)
}

horizon_labels <- c(
  "1" = "Horizon: 1",
  "2" = "Horizon: 2",
  "3" = "Horizon: 3",
  "4" = "Horizon: 4"
)

plot_precision_recall <- function(correct_by_table) {
  # Ensure correct_by_table is not empty and has the required columns
  if (nrow(correct_by_table) == 0 || 
      !all(c("precision", "recall", "f1", "horizon", "gt_table") %in% names(correct_by_table))) {
    warning("Insufficient data for plot_precision_recall. Returning empty plot.")
    return(ggplot() + theme_void() + labs(title = "Insufficient data for Precision-Recall Plot"))
  }
  
  # Remove rows where all metrics are NA to avoid issues with pivot_longer and plotting
  correct_by_table_filtered <- correct_by_table %>%
    filter(!(is.na(precision) & is.na(recall) & is.na(f1)))

  if (nrow(correct_by_table_filtered) == 0) {
    warning("No valid data after filtering NAs for plot_precision_recall. Returning empty plot.")
    return(ggplot() + theme_void() + labs(title = "No valid data for Precision-Recall Plot after NA filtering"))
  }

  correct_by_table_long <- correct_by_table_filtered %>%
    pivot_longer(
      cols = c("precision", "recall", "f1"),
      names_to = "metric",
      values_to = "value"
    ) %>%
    filter(!is.na(value)) # Ensure no NA values are passed to ggplot layers

  if (nrow(correct_by_table_long) == 0) {
    warning("No data to plot after pivoting and filtering NAs in plot_precision_recall. Returning empty plot.")
    return(ggplot() + theme_void() + labs(title = "No data to plot for Precision-Recall after pivoting and NA filtering"))
  }
  
  ggplot(correct_by_table_long, aes(x = horizon, y = value, fill = metric)) +
    geom_boxplot(
      position = position_dodge(width = 0.8),
      alpha = 0.5,
      na.rm = TRUE # Explicitly tell geom_boxplot to remove NAs
    ) +
    stat_summary(
      aes(group = metric, color = metric),
      fun = mean,
      geom = "line",
      position = position_dodge(width = 0.8),
      na.rm = TRUE
    ) +
    stat_summary(
      aes(group = metric, color = metric),
      fun = mean,
      geom = "point",
      position = position_dodge(width = 0.8),
      na.rm = TRUE
    ) +
    facet_wrap(~ gt_table, nrow = 2) +
    scale_fill_discrete(
      name = "Metric",
      labels = c("precision" = "Precision", "recall" = "Recall", "f1" = "F1")
    ) +
    scale_color_discrete(
      name = "Metric",
      labels = c("precision" = "Precision", "recall" = "Recall", "f1" = "F1")
    ) +
    scale_y_continuous(limits = c(0, 1)) +
    labs(
      x = "Horizon",
      y = "Metric Value"
    ) +
    theme_light() +
    theme(legend.position = "top")
}


plot_accuracy_over_time <- function(predictions_cur, num_bins = 40) {
  correct <- predictions_cur %>%
    group_by(
      horizon, iteration, unique_id,
      in_lock_start_time
    ) %>%
    summarise(
      is_correct = all(is_correct),
      .groups = 'drop'
    )
  
  correct$is_correct_num <- as.numeric(correct$is_correct)
  correct$in_lock_start_time_rel <- as.numeric(correct$in_lock_start_time - min(correct$in_lock_start_time))
  
  x_range <- range(correct$in_lock_start_time_rel, na.rm = TRUE)
  binwidth <- (x_range[2] - x_range[1]) / num_bins
  
  p <- ggplot(correct, aes(x = in_lock_start_time_rel, y = is_correct_num)) +
    stat_summary_bin(
      fun = "mean", geom = "col", binwidth = binwidth
    ) +
    facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
    labs(
      x = "Relative Lock End Time (nanoseconds)",
      y = "Percent Correct"
    ) +
    scale_x_continuous(labels = scales::scientific) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_light()
  
  return(p)
}
