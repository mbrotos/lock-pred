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

plot_accuracy_over_time_list <- function(dfs, df_names, num_bins = 40) {
  stopifnot(length(dfs) == length(df_names))
  all_data_list <- list()
  
  for (i in seq_along(dfs)) {
    predictions_cur <- dfs[[i]]
    dataset_name <- df_names[[i]]
    correct <- predictions_cur %>%
      group_by(horizon, iteration, unique_id, in_lock_start_time) %>%
      summarise(is_correct = all(is_correct), .groups = 'drop')
    correct$is_correct_num <- as.numeric(correct$is_correct)
    correct$in_lock_start_time_rel <- as.numeric(correct$in_lock_start_time - min(correct$in_lock_start_time))
    correct$dataset_name <- dataset_name
    all_data_list[[i]] <- correct
  }
  

  all_data <- dplyr::bind_rows(all_data_list)
  all_data$dataset_name <- factor(all_data$dataset_name, levels = df_names)
  
  print(df_names)
  
  x_range <- range(all_data$in_lock_start_time_rel, na.rm = TRUE)
  binwidth <- (x_range[2] - x_range[1]) / num_bins
  
  p <- ggplot(all_data, aes(x = in_lock_start_time_rel, y = is_correct_num, color = dataset_name)) +
    stat_summary_bin(fun = "mean", geom = "line", binwidth = binwidth, linewidth = 1) +
    facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
    labs(
      x = "Relative Lock End Time (nanoseconds)",
      y = "Percent Correct",
      color = "Model"
    ) +
    scale_x_continuous(labels = scales::scientific) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_light() +
    theme(
     # strip.background = element_blank(),      # Removes the gray background
      strip.text = element_text(size = 11, face = "bold")  # Optional: style the text
    ) +
    theme(
      legend.position = "top",
      legend.title = element_text(face = "bold"),
      legend.text = element_text(size = 11),
      axis.title = element_text(size = 13),
      axis.text = element_text(size = 11),
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5)
    ) +
    guides(color = guide_legend(override.aes = list(linewidth  = 4.5)))
  
  return(p)
}

# --- New Helper Functions ---

save_plot <- function(plot_object, file_path, width = 8, height = 6, units = "in", dpi = 300) {
  ggsave(
    filename = file_path,
    plot = plot_object,
    width = width,
    height = height,
    units = units,
    dpi = dpi
  )
  print(paste("Saved plot to:", file_path))
}

construct_output_path <- function(base_folder, 
                                  experiment_subdir, 
                                  filename) {
  dir_path <- file.path(base_folder, experiment_subdir)
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE)
    print(paste("Created directory:", dir_path))
  }
  return(file.path(dir_path, filename))
}


# --- New Refactored Plotting Functions ---

plot_horizon_performance_2_models <- function(data_model1, name_model1, data_model2, name_model2, 
                                            color_model1 = "black", color_model2 = "red", 
                                            file_path, plot_title = NULL, base_width = 8, base_height = 6) {
  p <- ggplot() +
    geom_boxplot(
      data = data_model1,
      aes(x = horizon, y = mean_percent_correct, color = name_model1),
      alpha = 0.5
    ) +
    geom_point(
      data = data_model2,
      aes(x = horizon, y = mean_percent_correct, color = name_model2),
      size = 2
    ) +
    labs(
      title = plot_title,
      x = "Horizon",
      y = "Percent Correct",
      color = "Legend"
    ) +
    scale_color_manual(values = setNames(c(color_model1, color_model2), c(name_model1, name_model2))) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_light() +
    theme(legend.position = "top") 
  
  save_plot(p, file_path, width = base_width, height = base_height)
}

plot_horizon_performance_3_models <- function(data_model1, name_model1, data_model2, name_model2, data_model3, name_model3, 
                                            color_model1 = "black", color_model2 = "blue", color_model3 = "red", 
                                            file_path, plot_title = NULL, base_width = 8, base_height = 6) {
  p <- ggplot() +
    geom_boxplot(
      data = data_model1,
      aes(x = horizon, y = mean_percent_correct, color = name_model1),
      alpha = 0.5
    ) +
    geom_boxplot(
      data = data_model2,
      aes(x = horizon, y = mean_percent_correct, color = name_model2),
      alpha = 0.5
    ) +
    geom_point(
      data = data_model3,
      aes(x = horizon, y = mean_percent_correct, color = name_model3),
      size = 2
    ) +
    labs(
      title = plot_title,
      x = "Horizon",
      y = "Percent Correct",
      color = "Legend"
    ) +
    scale_color_manual(values = setNames(c(color_model1, color_model2, color_model3), c(name_model1, name_model2, name_model3))) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_light() +
    theme(legend.position = "top") 
  
  save_plot(p, file_path, width = base_width, height = base_height)
}

plot_single_model_performance_by_table <- function(data_model_by_table, file_path, plot_title = NULL, base_width = 8, base_height = 6) {
  p <- ggplot(data_model_by_table, aes(x = gt_table, y = mean_percent_correct, fill = horizon)) +
    geom_boxplot(alpha = 0.5) +
    labs(
      title = plot_title,
      x = "Table",
      y = "Percent Correct",
      fill = "Horizon"
    ) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_light() +
    theme(legend.position = "top") 
  
  save_plot(p, file_path, width = base_width, height = base_height)
}

plot_table_performance_faceted_2_models <- function(data_model1_by_table, name_model1, data_model2_by_table, name_model2, 
                                                  color_model1 = "black", color_model2 = "red", 
                                                  file_path, plot_title = NULL, base_width = 15, base_height = 6) {
  p <- ggplot() +
    geom_boxplot(
      data = data_model1_by_table,
      aes(x = gt_table, y = mean_percent_correct, color = name_model1),
      alpha = 0.5
    ) +
    geom_point(
      data = data_model2_by_table,
      aes(x = gt_table, y = mean_percent_correct, color = name_model2),
      size = 2
    ) +
    facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
    labs(
      title = plot_title,
      x = "Table",
      y = "Percent Correct",
      color = "Model"
    ) +
    scale_color_manual(values = setNames(c(color_model1, color_model2), c(name_model1, name_model2))) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_light() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "top") 
  
  save_plot(p, file_path, width = base_width, height = base_height)
}

RQ2_plot_table_performance_faceted_3_models <- function(data_model1_by_table, name_model1, data_model2_by_table, name_model2, data_model3_by_table, name_model3, 
                                                  color_model1 = "black", color_model2 = "blue", color_model3 = "red", 
                                                  file_path, plot_title = NULL, base_width = 15, base_height = 6) {
  
  data_model1_by_table$Model <-  name_model1 
  data_model2_by_table$Model <-  name_model2
  data_model3_by_table$Model <-  name_model3

  # Add 'type' column to differentiate between boxplot and point
  data_model1_by_table$plot_type <- "box"
  data_model2_by_table$plot_type <- "box"
  data_model3_by_table$plot_type <- "box"
  # Combine all into one dataframe
  data_all <- rbind(data_model1_by_table, data_model2_by_table, data_model3_by_table)
  levels_models <- c(name_model1, name_model2, name_model3)  # ✅ Match exactly
  data_all$Model <- factor(data_all$Model, levels = levels_models)
  
  p <- ggplot(data_all, aes(x = gt_table, y = mean_percent_correct, color = Model)) +
    geom_boxplot(data = subset(data_all, plot_type == "box"), alpha = 0.5) +
    facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
    labs(
      title = plot_title,
      x = "Table",
      y = "Percent Correct",
      color = "Model"
    ) +
    scale_color_manual(values = setNames(c(color_model1, color_model2, color_model3), c(name_model1, name_model2, name_model3))) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_light() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "top")+
    theme(
      # strip.background = element_blank(),      # Removes the gray background
      strip.text = element_text(size = 11, face = "bold")  # Optional: style the text
    ) +
    theme(
    legend.position = "top",
    legend.title = element_text(face = "bold"),
    legend.text = element_text(size = 11),
    axis.title = element_text(size = 13),
    axis.text = element_text(size = 11),
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5)
  ) +
    guides(color = guide_legend(override.aes = list(linewidth  = 4.5)))
  
  save_plot(p, file_path, width = base_width, height = base_height)
}

generate_and_save_precision_recall_plot <- function(data_model_by_table, file_path, base_width = 10, base_height = 6) {
  p <- plot_precision_recall(data_model_by_table) 
  save_plot(p, file_path, width = base_width, height = base_height)
}

plot_comparison_by_table_faceted <- function(data_model1_by_table, name_model1, data_model2_by_table, name_model2, 
                                           file_path, plot_title = NULL, base_width = 15, base_height = 6) {
  data_model1_by_table$Model <- name_model1
  data_model2_by_table$Model <- name_model2
  
  combined_data <- bind_rows(data_model1_by_table, data_model2_by_table)
  
  p <- ggplot(combined_data, aes(x = gt_table, y = mean_percent_correct, fill = Model)) +
    geom_boxplot(alpha = 0.5) +
    facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
    labs(
      title = plot_title,
      x = "Table",
      y = "Percent Correct",
      fill = "Model"
    ) +
    theme_light() +
    scale_y_continuous(limits = c(0, 1)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "top") 
  
  save_plot(p, file_path, width = base_width, height = base_height)
}

#######
#######
#######
# Lets look at "cut" global results:
message("--- Starting 'cut' Experiment Analysis ---")
experiment_subdir_cut <- "cut"

#basepath = "C:/TMU/postdoc-TMU/deep-rediscovery/deeptable-analysis/results/"

setwd("C:/TMU/postdoc-TMU/lock-pred/")


predictions <- load_parquet("analysis/data/exp-39-tranformer-rounded-cut-row-locks/predictions.parquet")
head(predictions)
check_iterations(predictions)
predictions_transformer <-predictions

predictions_lstm <- load_parquet("analysis/data/exp-40-lstm-rounded-cut-row-locks/predictions.parquet")
check_iterations(predictions_lstm)

predictions_naive <- load_parquet("analysis/data/exp-41-naive-rounded-cut-row-locks/predictions.parquet") %>%
  filter(data == "data/fixed/row_locks.csv")

# Accuracy over time plots
p_global_acc_time <- plot_accuracy_over_time_list(
  list(predictions_transformer, predictions_lstm, predictions_naive),
  c("Global Transformer", "Global LSTM", "Global Naive Baseline")
)

save_plot(p_global_acc_time, 
          construct_output_path("analysis/plots", experiment_subdir_cut, "global_transformer_lstm_naive_baseline_accuracy_over_time.pdf"), 
          width = 10, height = 6)

# Transformer accuracy overtime
p_transformer_acc_time <- plot_accuracy_over_time(predictions_transformer)
save_plot(p_transformer_acc_time, 
          construct_output_path("analysis/plots", experiment_subdir_cut, "global_transformer_accuracy_over_time.pdf"), 
          width = 10, height = 6)

# Naive accuracy overtime
p_naive_acc_time <- plot_accuracy_over_time(predictions_naive)
save_plot(p_naive_acc_time, 
          construct_output_path("analysis/plots", experiment_subdir_cut, "global_naive_baseline_accuracy_over_time.pdf"), 
          width = 10, height = 6)

# Performance metrics
correct <- horizon_iteration_performance(predictions_transformer)
correct_by_table <- horizon_iteration_performance_by_table(predictions_transformer)
correct_lstm <- horizon_iteration_performance(predictions_lstm)
correct_by_table_lstm <- horizon_iteration_performance_by_table(predictions_lstm)
correct_naive <- horizon_iteration_performance(predictions_naive)
correct_naive_by_table <- horizon_iteration_performance_by_table(predictions_naive)

rm(predictions_transformer, predictions_lstm, predictions_naive); gc()

 
RQ2_plot_table_performance_faceted_3_models(correct_by_table, "Global Transformer", correct_by_table_lstm, "Global LSTM", correct_naive_by_table, "Global Naive Baseline",
    file_path = construct_output_path("analysis/plots", experiment_subdir_cut, "global_transformer_lstm_vs_naive_baseline_by_table.pdf"))



# Lets look at local for cut
predictions_local <- load_parquet("analysis/data/exp-44-transformer-local-rounded-cut/predictions.parquet")
check_iterations(predictions_local)
predictions_local_lstm <- load_parquet("analysis/data/exp-43-lstm-local-rounded-cut/predictions.parquet")
check_iterations(predictions_local_lstm)
predictions_naive_local <- load_parquet("analysis/data/exp-42-naive-local-rounded-cut-row-locks/predictions.parquet")

# Accuracy over time for local models
p_local_acc_time <- plot_accuracy_over_time_list(
  list(predictions_local, predictions_local_lstm, predictions_naive_local),
  c("Local Transformer", "Local LSTM", "Local Naive Baseline")
)

correct_local <- horizon_iteration_performance(predictions_local)
correct_local_by_table <- horizon_iteration_performance_by_table(predictions_local)
correct_local_lstm <- horizon_iteration_performance(predictions_local_lstm)
correct_local_by_table_lstm <- horizon_iteration_performance_by_table(predictions_local_lstm)
correct_naive_local <- horizon_iteration_performance(predictions_naive_local) 
correct_naive_local_by_table <- horizon_iteration_performance_by_table(predictions_naive_local)

RQ2_plot_table_performance_faceted_3_models(correct_local_by_table, "Local Transformer", correct_local_by_table_lstm, "Local LSTM", correct_naive_local_by_table, "Local Naive Baseline",
                                        file_path = construct_output_path("analysis/plots", experiment_subdir_cut, "local_transformer_lstm_vs_local_naive_baseline_by_table.pdf"))

 