require(tidyverse)
require(arrow)  # to read parquet
library(dplyr)
library(ggplot2)
library(stringr) # For str_remove and str_trim

# --- Core Data Processing Functions (largely unchanged) ---
is_correct <- function(df, is_table_lock=FALSE) {
  if (is_table_lock) {
    correct <- (df$gt_table == df$pred_table)
  } else {
    correct <- (df$gt_table == df$pred_table) &
      (df$gt_pageid == df$pred_pageid)
    df$gt_pageid <- NULL
    df$pred_pageid <- NULL
  }
  correct[is.na(correct)] <- FALSE
  return(correct)
}

load_parquet_data <- function(path, is_table_lock=FALSE, filter_tail_ns=3e11) {
  predictions <- read_parquet(path, col_select = c("in_lock_sequences_id", "in_lock_start_time", "data", "horizon", "unique_id", "horizon_position", "gt_table", "gt_pageid", "pred_table", "pred_pageid", "iteration")) %>%
    filter(iteration <= 10 & gt_table != "warehouse") # Retained original filter
  predictions$is_correct <- is_correct(predictions, is_table_lock)
  predictions$horizon <- as.factor(predictions$horizon)

  if (filter_tail_ns > 0 && nrow(predictions) > 0 && "in_lock_start_time" %in% colnames(predictions)) {
      max_time <- max(predictions$in_lock_start_time, na.rm = TRUE)
      if (!is.infinite(max_time)) {
        predictions <- predictions %>%
          filter(in_lock_start_time < max_time - filter_tail_ns)
      } else {
        warning(paste("Max 'in_lock_start_time' is infinite for path:", path, "- skipping tail filter."))
      }
    } else if (filter_tail_ns > 0) {
      warning(paste("Skipping tail filter for path:", path, "due to empty data or missing 'in_lock_start_time' column."))
    }
  return(predictions)
}

check_iterations <- function(df, df_name = "current_df") {
  if (nrow(df) == 0) {
    warning(paste("DataFrame '", df_name, "' is empty. Skipping iteration check.", sep=""))
    return()
  }
  horizon_iteration_counts <- df %>%
    group_by(horizon, iteration) %>%
    summarise(n = n(), .groups='drop') %>%
    group_by(horizon) %>%
    summarise(n_iterations = n(), .groups='drop') # Renamed to n_iterations for clarity

  if (any(horizon_iteration_counts$n_iterations < 10)) {
    print(paste("Iteration check for:", df_name))
    print(horizon_iteration_counts)
    # stop(paste("Some horizons in", df_name, "have less than 10 iterations")) # Disabled stop for broader script execution
    warning(paste("Some horizons in", df_name, "have less than 10 iterations. Execution will continue."))
  } else {
    print(paste("All horizons in", df_name, "have at least 10 iterations."))
  }
}

horizon_iteration_performance <- function(predictions) {
  if (nrow(predictions) == 0) return(tibble(horizon=factor(), iteration=integer(), mean_percent_correct=double()))
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

horizon_iteration_performance_by_table <- function(predictions) {
  if (nrow(predictions) == 0) {
    return(tibble(
      horizon = factor(), iteration = integer(), gt_table = character(),
      mean_percent_correct = double(), tp = integer(), fp = integer(), fn = integer(),
      precision = double(), recall = double(), f1 = double()
    ))
  }
  correct_by_table <- predictions %>%
    group_by(horizon, iteration, unique_id, gt_table) %>%
    summarise(is_correct = all(is_correct), .groups='drop') %>%
    group_by(horizon, iteration, gt_table) %>%
    summarise(mean_percent_correct = mean(is_correct), .groups='drop')

  print(correct_by_table %>%
    group_by(horizon, gt_table) %>%
    summarise(percent_correct = mean(mean_percent_correct), .groups='drop'))

  conf <- predictions %>%
    group_by(horizon, iteration, gt_table, pred_table) %>%
    summarise(n = n(), .groups='drop')

  all_tables <- union(conf$gt_table, conf$pred_table)
  
  # Ensure all_combos is not empty if conf is empty
  if(nrow(conf) == 0) {
     all_combos <- tibble(horizon = factor(), iteration = integer(), table = character())
  } else {
     all_combos <- conf %>%
        distinct(horizon, iteration) %>%
        tidyr::crossing(table = all_tables)
  }


  metrics <- all_combos %>%
    left_join(
      conf %>% filter(gt_table == pred_table) %>% rename(table = gt_table) %>% select(horizon, iteration, table, n),
      by = c("horizon", "iteration", "table")
    ) %>%
    rename(tp = n) %>%
    mutate(tp = tidyr::replace_na(tp, 0)) %>%
    left_join(
      conf %>% group_by(horizon, iteration, pred_table) %>% summarise(fp = sum(ifelse(gt_table != pred_table, n, 0)), .groups='drop') %>% rename(table = pred_table),
      by = c("horizon", "iteration", "table")
    ) %>%
    mutate(fp = tidyr::replace_na(fp, 0)) %>%
    left_join(
      conf %>% group_by(horizon, iteration, gt_table) %>% summarise(fn = sum(ifelse(gt_table != pred_table, n, 0)), .groups='drop') %>% rename(table = gt_table),
      by = c("horizon", "iteration", "table")
    ) %>%
    mutate(fn = tidyr::replace_na(fn, 0)) %>%
    mutate(
      precision = ifelse(tp + fp == 0, NA_real_, tp / (tp + fp)),
      recall    = ifelse(tp + fn == 0, NA_real_, tp / (tp + fn)),
      f1        = ifelse(!is.na(precision) & !is.na(recall) & (precision + recall > 0), 2 * precision * recall / (precision + recall), NA_real_)
    )  %>%
    rename(gt_table = table)
  
  # Ensure metrics has the correct columns even if empty
  if (nrow(metrics) == 0 && nrow(all_combos) > 0) {
     metrics <- all_combos %>% rename(gt_table = table) %>%
       mutate(tp = 0, fp = 0, fn = 0, precision = NA_real_, recall = NA_real_, f1 = NA_real_)
  } else if (nrow(metrics) == 0 && nrow(all_combos) == 0) {
     metrics <- tibble(horizon=factor(), iteration=integer(), gt_table=character(), tp=integer(), fp=integer(), fn=integer(), precision=double(), recall=double(), f1=double())
  }


  return(correct_by_table %>% left_join(metrics, by = c("horizon", "iteration", "gt_table")))
}

export_csv <- function(df, path) {
  if (nrow(df) == 0) {
    write_csv(tibble(horizon=factor(), mean_percent_correct_csv=double(), median_percent_correct_csv=double()), path)
    return()
  }
  df %>%
    group_by(horizon) %>%
    summarise(
      mean_percent_correct_csv = mean(mean_percent_correct, na.rm = TRUE),
      median_percent_correct_csv = median(mean_percent_correct, na.rm = TRUE),
      .groups='drop'
    ) %>%
    write_csv(path)
}

export_csv_by_table <- function(df, path) {
  if (nrow(df) == 0) {
    write_csv(tibble(
      horizon=factor(), gt_table=character(), mean_percent_correct_csv=double(), median_percent_correct_csv=double(),
      mean_precision_csv=double(), median_precision_csv=double(), mean_recall_csv=double(), median_recall_csv=double(),
      mean_f1_csv=double(), median_f1_csv=double()
    ), path)
    return()
  }
  df %>%
    group_by(horizon, gt_table) %>%
    summarise(
      mean_percent_correct_csv = mean(mean_percent_correct, na.rm = TRUE),
      median_percent_correct_csv = median(mean_percent_correct, na.rm = TRUE),
      mean_precision_csv = mean(precision, na.rm = TRUE),
      median_precision_csv = median(precision, na.rm = TRUE),
      mean_recall_csv    = mean(recall, na.rm = TRUE),
      median_recall_csv = median(recall, na.rm = TRUE),
      mean_f1_csv        = mean(f1, na.rm = TRUE),
      median_f1_csv   = median(f1, na.rm = TRUE),
      .groups='drop'
    ) %>%
    write_csv(path)
}

horizon_labels <- c("1" = "Horizon: 1", "2" = "Horizon: 2", "3" = "Horizon: 3", "4" = "Horizon: 4")

# --- Refactored Plotting Functions ---

plot_accuracy_over_time_single <- function(predictions_cur, num_bins = 40, plot_title_suffix = "") {
  if (nrow(predictions_cur) == 0) {
    warning(paste("No data to plot for accuracy over time", plot_title_suffix))
    return(ggplot() + labs(title = paste("No data for", plot_title_suffix)))
  }
  correct <- predictions_cur %>%
    group_by(horizon, iteration, unique_id, in_lock_start_time) %>%
    summarise(is_correct = all(is_correct), .groups = 'drop')
  
  correct$is_correct_num <- as.numeric(correct$is_correct)
  min_time <- min(correct$in_lock_start_time, na.rm = TRUE)
  if(is.infinite(min_time)) min_time <- 0 # Handle case where all times might be NA or no data

  correct$in_lock_start_time_rel <- as.numeric(correct$in_lock_start_time - min_time)
  
  x_range <- range(correct$in_lock_start_time_rel, na.rm = TRUE)
  binwidth <- if(x_range[2] == x_range[1] || is.na(x_range[2]) || is.na(x_range[1])) 1 else (x_range[2] - x_range[1]) / num_bins
  if(binwidth == 0) binwidth <- 1 # Avoid zero binwidth

  p <- ggplot(correct, aes(x = in_lock_start_time_rel, y = is_correct_num)) +
    stat_summary_bin(fun = "mean", geom = "col", binwidth = binwidth, fill = "steelblue", alpha = 0.7) +
    facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
    labs(
      title = paste("Accuracy Over Time", plot_title_suffix),
      x = "Relative Lock End Time (nanoseconds)",
      y = "Percent Correct"
    ) +
    scale_x_continuous(labels = scales::scientific) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_light()
  return(p)
}

plot_accuracy_over_time_multiple <- function(dfs, df_names, num_bins = 40, plot_title_suffix = "") {
  stopifnot(length(dfs) == length(df_names))
  all_data_list <- list()

  for (i in seq_along(dfs)) {
    predictions_cur <- dfs[[i]]
    dataset_name <- df_names[[i]]
    if (nrow(predictions_cur) == 0) next # Skip empty dataframes

    correct <- predictions_cur %>%
      group_by(horizon, iteration, unique_id, in_lock_start_time) %>%
      summarise(is_correct = all(is_correct), .groups = 'drop')
    correct$is_correct_num <- as.numeric(correct$is_correct)
    min_time <- min(correct$in_lock_start_time, na.rm = TRUE)
    if(is.infinite(min_time)) min_time <- 0

    correct$in_lock_start_time_rel <- as.numeric(correct$in_lock_start_time - min_time)
    correct$dataset_name <- dataset_name
    all_data_list[[i]] <- correct
  }

  if (length(all_data_list) == 0) {
    warning(paste("No data to plot for multi-accuracy over time", plot_title_suffix))
    return(ggplot() + labs(title = paste("No data for", plot_title_suffix)))
  }
  all_data <- dplyr::bind_rows(all_data_list)
  if (nrow(all_data) == 0) return(ggplot() + labs(title = paste("No combined data for", plot_title_suffix)))


  x_range <- range(all_data$in_lock_start_time_rel, na.rm = TRUE)
  binwidth <- if(x_range[2] == x_range[1] || is.na(x_range[2]) || is.na(x_range[1])) 1 else (x_range[2] - x_range[1]) / num_bins
   if(binwidth == 0) binwidth <- 1

  p <- ggplot(all_data, aes(x = in_lock_start_time_rel, y = is_correct_num, color = dataset_name)) +
    stat_summary_bin(fun = "mean", geom = "line", binwidth = binwidth, aes(group = dataset_name)) +
    facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
    labs(
      title = paste("Accuracy Over Time Comparison", plot_title_suffix),
      x = "Relative Lock End Time (nanoseconds)",
      y = "Percent Correct",
      color = "Dataset"
    ) +
    scale_x_continuous(labels = scales::scientific) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_light() +
    theme(legend.position = "top")
  return(p)
}

plot_precision_recall_f1 <- function(correct_by_table_df, plot_title_suffix = "") {
  if (nrow(correct_by_table_df) == 0 || !all(c("precision", "recall", "f1") %in% names(correct_by_table_df))) {
     warning(paste("Insufficient data/columns for Precision-Recall plot", plot_title_suffix))
     return(ggplot() + labs(title = paste("No data for P-R-F1", plot_title_suffix)))
  }
  correct_by_table_long <- correct_by_table_df %>%
    pivot_longer(cols = c("precision", "recall", "f1"), names_to = "metric", values_to = "value")

  ggplot(correct_by_table_long, aes(x = horizon, y = value, fill = metric)) +
    geom_boxplot(position = position_dodge(width = 0.8), alpha = 0.5, outlier.shape = NA) + # Added outlier.shape = NA
    stat_summary(aes(group = metric, color = metric), fun = mean, geom = "line", position = position_dodge(width = 0.8), na.rm = TRUE) +
    stat_summary(aes(group = metric, color = metric), fun = mean, geom = "point", position = position_dodge(width = 0.8), na.rm = TRUE) +
    facet_wrap(~ gt_table, nrow = 2) +
    scale_fill_discrete(name = "Metric", labels = c("precision" = "Precision", "recall" = "Recall", "f1" = "F1")) +
    scale_color_discrete(name = "Metric", labels = c("precision" = "Precision", "recall" = "Recall", "f1" = "F1")) +
    scale_y_continuous(limits = c(0, 1)) +
    labs(title = paste("Precision, Recall, F1-Score", plot_title_suffix), x = "Horizon", y = "Metric Value") +
    theme_light() +
    theme(legend.position = "top")
}

# Generic plot for comparing two models (boxplot vs points) on overall accuracy vs horizon
plot_overall_accuracy_comparison <- function(df_boxplot, df_points, name_boxplot, name_points, plot_title_suffix = "") {
  if (nrow(df_boxplot) == 0 && nrow(df_points) == 0) {
    warning(paste("No data for overall accuracy comparison", plot_title_suffix))
    return(ggplot() + labs(title = paste("No data for", plot_title_suffix)))
  }
  
  p <- ggplot()
  if (nrow(df_boxplot) > 0) {
    p <- p + geom_boxplot(data = df_boxplot, aes(x = horizon, y = mean_percent_correct, color = name_boxplot), alpha = 0.5, outlier.shape = NA)
  }
  if (nrow(df_points) > 0) {
    p <- p + geom_point(data = df_points, aes(x = horizon, y = mean_percent_correct, color = name_points), size = 2)
  }
  
  p + labs(
      title = paste("Horizon vs. Percent Correct", plot_title_suffix),
      x = "Horizon", y = "Percent Correct", color = "Model"
    ) +
    scale_color_manual(values = setNames(c("black", "red"), c(name_boxplot, name_points))) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_light() +
    theme(legend.position = "top")
}

# Generic plot for comparing multiple (e.g., three) models on overall accuracy vs horizon
plot_overall_accuracy_multi_model <- function(dfs_boxplot, df_points, names_boxplot, name_points, colors, plot_title_suffix = "") {
  if (all(sapply(dfs_boxplot, nrow) == 0) && nrow(df_points) == 0) {
    warning(paste("No data for multi-model overall accuracy comparison", plot_title_suffix))
    return(ggplot() + labs(title = paste("No data for", plot_title_suffix)))
  }

  p <- ggplot()
  for (i in seq_along(dfs_boxplot)) {
    if (nrow(dfs_boxplot[[i]]) > 0) {
      p <- p + geom_boxplot(data = dfs_boxplot[[i]], aes(x = horizon, y = mean_percent_correct, color = names_boxplot[i]), alpha = 0.5, outlier.shape = NA)
    }
  }
  if (nrow(df_points) > 0) {
    p <- p + geom_point(data = df_points, aes(x = horizon, y = mean_percent_correct, color = name_points), size = 2)
  }

  p + labs(
      title = paste("Horizon vs. Percent Correct", plot_title_suffix),
      x = "Horizon", y = "Percent Correct", color = "Model"
    ) +
    scale_color_manual(values = colors) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_light() +
    theme(legend.position = "top")
}


# Generic plot for single model accuracy by table, faceted by horizon
plot_accuracy_by_table_single <- function(df_correct_by_table, plot_title_suffix = "") {
  if (nrow(df_correct_by_table) == 0) {
    warning(paste("No data for single model accuracy by table", plot_title_suffix))
    return(ggplot() + labs(title = paste("No data for", plot_title_suffix)))
  }
  ggplot(df_correct_by_table, aes(x = gt_table, y = mean_percent_correct, fill = horizon)) +
    geom_boxplot(alpha = 0.5, outlier.shape = NA) +
    labs(
      title = paste("Table vs. Percent Correct by Horizon", plot_title_suffix),
      x = "Table", y = "Percent Correct", fill = "Horizon"
    ) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_light() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "top")
}

# Generic plot for comparing two models by table (boxplot vs points), faceted by horizon
plot_accuracy_by_table_comparison <- function(df_boxplot, df_points, name_boxplot, name_points, plot_title_suffix = "") {
  if (nrow(df_boxplot) == 0 && nrow(df_points) == 0) {
     warning(paste("No data for by-table accuracy comparison", plot_title_suffix))
    return(ggplot() + labs(title = paste("No data for", plot_title_suffix)))
  }
  p <- ggplot()
  if (nrow(df_boxplot) > 0) {
    p <- p + geom_boxplot(data = df_boxplot, aes(x = gt_table, y = mean_percent_correct, color = name_boxplot), alpha = 0.5, outlier.shape = NA)
  }
  if (nrow(df_points) > 0) {
    p <- p + geom_point(data = df_points, aes(x = gt_table, y = mean_percent_correct, color = name_points), size = 2)
  }
  
  p + facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
    labs(
      title = paste("Table vs. Percent Correct by Horizon", plot_title_suffix),
      x = "Table", y = "Percent Correct", color = "Model"
    ) +
    scale_color_manual(values = setNames(c("black", "red"), c(name_boxplot, name_points))) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_light() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "top")
}

# Generic plot for comparing multiple (e.g., three) models by table, faceted by horizon
plot_accuracy_by_table_multi_model <- function(dfs_boxplot, df_points, names_boxplot, name_points, colors, plot_title_suffix = "") {
  if (all(sapply(dfs_boxplot, nrow) == 0) && nrow(df_points) == 0) {
    warning(paste("No data for multi-model by-table accuracy comparison", plot_title_suffix))
    return(ggplot() + labs(title = paste("No data for", plot_title_suffix)))
  }
  p <- ggplot()
  for (i in seq_along(dfs_boxplot)) {
     if (nrow(dfs_boxplot[[i]]) > 0) {
      p <- p + geom_boxplot(data = dfs_boxplot[[i]], aes(x = gt_table, y = mean_percent_correct, color = names_boxplot[i]), alpha = 0.5, outlier.shape = NA)
    }
  }
  if (nrow(df_points) > 0) {
    p <- p + geom_point(data = df_points, aes(x = gt_table, y = mean_percent_correct, color = name_points), size = 2)
  }

  p + facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
    labs(
      title = paste("Table vs. Percent Correct by Horizon", plot_title_suffix),
      x = "Table", y = "Percent Correct", color = "Model"
    ) +
    scale_color_manual(values = colors) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_light() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "top")
}


# Generic plot for comparing two models using boxplots, overall accuracy
plot_overall_accuracy_two_boxplots <- function(df1, df2, name1, name2, colors, plot_title_suffix = "") {
  if (nrow(df1) == 0 && nrow(df2) == 0) {
    warning(paste("No data for two-boxplot overall accuracy comparison", plot_title_suffix))
    return(ggplot() + labs(title = paste("No data for", plot_title_suffix)))
  }
  p <- ggplot()
  if (nrow(df1) > 0) {
    p <- p + geom_boxplot(data = df1, aes(x = horizon, y = mean_percent_correct, color = name1), alpha = 0.5, outlier.shape = NA)
  }
  if (nrow(df2) > 0) {
    p <- p + geom_boxplot(data = df2, aes(x = horizon, y = mean_percent_correct, color = name2), alpha = 0.5, outlier.shape = NA)
  }
  p + labs(
      title = paste("Horizon vs. Percent Correct", plot_title_suffix),
      x = "Horizon", y = "Percent Correct", color = "Model"
    ) +
    scale_color_manual(values = colors) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_light() +
    theme(legend.position = "top")
}

# Generic plot for comparing two models using boxplots, by table, faceted by horizon
plot_accuracy_by_table_two_boxplots <- function(df1_by_table, df2_by_table, name1, name2, colors, plot_title_suffix = "") {
   if (nrow(df1_by_table) == 0 && nrow(df2_by_table) == 0) {
    warning(paste("No data for two-boxplot by-table accuracy comparison", plot_title_suffix))
    return(ggplot() + labs(title = paste("No data for", plot_title_suffix)))
  }
  
  # Add model column for ggplot aes
  if (nrow(df1_by_table) > 0) df1_by_table$Model <- name1
  if (nrow(df2_by_table) > 0) df2_by_table$Model <- name2
  
  combined_df <- bind_rows(df1_by_table, df2_by_table)
  if (nrow(combined_df) == 0) return(ggplot() + labs(title = paste("No combined data for", plot_title_suffix)))


  ggplot(combined_df, aes(x = gt_table, y = mean_percent_correct, fill = Model)) +
    geom_boxplot(alpha = 0.5, outlier.shape = NA, position = position_dodge(preserve = "single")) + # preserve single to avoid issues with one model having no data for a table
    facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
    labs(
      title = paste("Table vs. Percent Correct by Horizon", plot_title_suffix),
      x = "Table", y = "Percent Correct", fill = "Model"
    ) +
    scale_fill_manual(values = colors) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_light() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "top")
}


# --- Helper to save plots ---
save_plot <- function(plot_obj, path, width = 8, height = 6, units = "in", dpi = 300) {
  if (!is.null(plot_obj) && inherits(plot_obj, "ggplot")) {
    ggsave(filename = path, plot = plot_obj, width = width, height = height, units = units, dpi = dpi)
    print(paste("Saved plot to:", path))
  } else {
    warning(paste("Plot object is NULL or not a ggplot object. Skipping save for:", path))
  }
}

# --- Orchestration Function for a Standard Analysis Block ---
run_analysis_suite <- function(
    predictions_list, # A named list of dataframes: list(model_name = df, ...)
    naive_baseline_df = NULL, # Optional: a single dataframe for naive baseline (points)
    naive_baseline_name = "Naive Baseline",
    experiment_prefix = "exp", # e.g., "global_transformer", "local_lstm_sorted"
    output_dir = "analysis",   # Base output directory
    is_table_lock_experiment = FALSE,
    filter_csv_data = NULL # Optional: string to filter specific CSV data if naive is from a combined source
  ) {

  # Create directories if they don't exist
  plot_path_base <- file.path(output_dir, "plots", experiment_prefix)
  table_path_base <- file.path(output_dir, "tables", experiment_prefix)
  dir.create(plot_path_base, recursive = TRUE, showWarnings = FALSE)
  dir.create(table_path_base, recursive = TRUE, showWarnings = FALSE)

  # Process each model prediction dataset
  processed_data <- list()
  for (model_name in names(predictions_list)) {
    current_preds <- predictions_list[[model_name]]
    model_suffix <- gsub(" ", "_", tolower(model_name)) # e.g., global_transformer

    print(paste("--- Processing:", experiment_prefix, "-", model_name, "---"))
    if (is.null(current_preds) || nrow(current_preds) == 0) {
        warning(paste("Predictions data for", model_name, "is NULL or empty. Skipping."))
        processed_data[[model_name]] <- list(preds = NULL, correct = NULL, correct_by_table = NULL)
        next
    }
    
    check_iterations(current_preds, paste(experiment_prefix, model_name))

    correct_overall <- horizon_iteration_performance(current_preds)
    correct_by_tbl <- horizon_iteration_performance_by_table(current_preds)

    export_csv(correct_overall, file.path(table_path_base, paste0(model_suffix, "_performance.csv")))
    export_csv_by_table(correct_by_tbl, file.path(table_path_base, paste0(model_suffix, "_performance_by_table.csv")))
    
    # Plot accuracy over time for this model
    p_acc_time <- plot_accuracy_over_time_single(current_preds, plot_title_suffix = paste(experiment_prefix, model_name))
    save_plot(p_acc_time, file.path(plot_path_base, paste0(model_suffix, "_accuracy_over_time.pdf")), width=10)

    # Plot precision/recall for this model
    p_pr <- plot_precision_recall_f1(correct_by_tbl, plot_title_suffix = paste(experiment_prefix, model_name))
    save_plot(p_pr, file.path(plot_path_base, paste0(model_suffix, "_precision_recall.pdf")), width=10)
    
    # Store processed data
    processed_data[[model_name]] <- list(preds = current_preds, correct = correct_overall, correct_by_table = correct_by_tbl)
  }
  
  # Process Naive Baseline if provided
  processed_naive <- NULL
  if (!is.null(naive_baseline_df) && nrow(naive_baseline_df) > 0) {
    print(paste("--- Processing Naive Baseline for:", experiment_prefix, "---"))
    
    # Apply filtering if naive baseline data comes from a combined source (e.g. exp-10, exp-18)
    if (!is.null(filter_csv_data) && "data" %in% colnames(naive_baseline_df)) {
        naive_baseline_df_filtered <- naive_baseline_df %>% filter(data == filter_csv_data)
        if (nrow(naive_baseline_df_filtered) == 0) {
            warning(paste("Naive baseline data for", experiment_prefix, "is empty after filtering for", filter_csv_data))
            naive_baseline_df <- naive_baseline_df_filtered # assign empty df
        } else {
            naive_baseline_df <- naive_baseline_df_filtered
        }
    }

    if (nrow(naive_baseline_df) > 0) { # Check again after potential filtering
        check_iterations(naive_baseline_df, paste(experiment_prefix, naive_baseline_name))
        correct_naive_overall <- horizon_iteration_performance(naive_baseline_df)
        correct_naive_by_tbl <- horizon_iteration_performance_by_table(naive_baseline_df)

        export_csv(correct_naive_overall, file.path(table_path_base, paste0(tolower(gsub(" ", "_", naive_baseline_name)), "_performance.csv")))
        export_csv_by_table(correct_naive_by_tbl, file.path(table_path_base, paste0(tolower(gsub(" ", "_", naive_baseline_name)), "_performance_by_table.csv")))
        
        p_acc_time_naive <- plot_accuracy_over_time_single(naive_baseline_df, plot_title_suffix = paste(experiment_prefix, naive_baseline_name))
        save_plot(p_acc_time_naive, file.path(plot_path_base, paste0(tolower(gsub(" ", "_", naive_baseline_name)), "_accuracy_over_time.pdf")), width=10)
        
        p_pr_naive <- plot_precision_recall_f1(correct_naive_by_tbl, plot_title_suffix = paste(experiment_prefix, naive_baseline_name))
        save_plot(p_pr_naive, file.path(plot_path_base, paste0(tolower(gsub(" ", "_", naive_baseline_name)), "_precision_recall.pdf")), width=10)
        
        processed_naive <- list(preds = naive_baseline_df, correct = correct_naive_overall, correct_by_table = correct_naive_by_tbl)
    } else {
        warning(paste("Naive baseline data for", experiment_prefix, "is empty. Skipping naive baseline processing."))
    }
  }

  # Combined plots for accuracy over time (all models in predictions_list + naive)
  all_dfs_for_multitimeplot <- lapply(names(processed_data), function(name) processed_data[[name]]$preds)
  all_names_for_multitimeplot <- names(processed_data)
  if (!is.null(processed_naive) && !is.null(processed_naive$preds)) {
    all_dfs_for_multitimeplot <- c(all_dfs_for_multitimeplot, list(processed_naive$preds))
    all_names_for_multitimeplot <- c(all_names_for_multitimeplot, naive_baseline_name)
  }
  
  # Remove NULL entries if any model had no data
  valid_indices <- !sapply(all_dfs_for_multitimeplot, is.null)
  all_dfs_for_multitimeplot <- all_dfs_for_multitimeplot[valid_indices]
  all_names_for_multitimeplot <- all_names_for_multitimeplot[valid_indices]

  if (length(all_dfs_for_multitimeplot) > 1) {
    p_acc_time_multi <- plot_accuracy_over_time_multiple(all_dfs_for_multitimeplot, all_names_for_multitimeplot, plot_title_suffix = experiment_prefix)
    save_plot(p_acc_time_multi, file.path(plot_path_base, "all_models_accuracy_over_time.pdf"), width=10)
  }


  # --- Comparative plots ---
  # Assuming the first model in predictions_list is the primary one for comparison plots
  # This logic might need adjustment depending on how many models you want to compare and in what combinations.
  
  model_names <- names(processed_data)
  
  # Compare first model with naive baseline (if naive exists)
  if (length(model_names) >= 1 && !is.null(processed_data[[model_names[1]]]$correct) && !is.null(processed_naive) && !is.null(processed_naive$correct)) {
    primary_model_name <- model_names[1]
    p_overall_comp <- plot_overall_accuracy_comparison(
      processed_data[[primary_model_name]]$correct, processed_naive$correct,
      primary_model_name, naive_baseline_name,
      plot_title_suffix = paste(experiment_prefix, primary_model_name, "vs", naive_baseline_name)
    )
    save_plot(p_overall_comp, file.path(plot_path_base, paste0(tolower(gsub(" ", "_", primary_model_name)), "_vs_", tolower(gsub(" ", "_", naive_baseline_name)), ".pdf")))

    p_by_table_comp <- plot_accuracy_by_table_comparison(
      processed_data[[primary_model_name]]$correct_by_table, processed_naive$correct_by_table,
      primary_model_name, naive_baseline_name,
      plot_title_suffix = paste(experiment_prefix, primary_model_name, "vs", naive_baseline_name)
    )
    save_plot(p_by_table_comp, file.path(plot_path_base, paste0(tolower(gsub(" ", "_", primary_model_name)), "_vs_", tolower(gsub(" ", "_", naive_baseline_name)), "_by_table.pdf")), width=15)
  }

  # Compare first two models from predictions_list (if at least two exist)
  if (length(model_names) >= 2 && !is.null(processed_data[[model_names[1]]]$correct) && !is.null(processed_data[[model_names[2]]]$correct)) {
    model1_name <- model_names[1]
    model2_name <- model_names[2]
    
    colors_two_model <- setNames(c("black", "blue"), c(model1_name, model2_name))

    p_overall_two_model <- plot_overall_accuracy_two_boxplots(
        processed_data[[model1_name]]$correct, processed_data[[model2_name]]$correct,
        model1_name, model2_name, colors_two_model,
        plot_title_suffix = paste(experiment_prefix, model1_name, "vs", model2_name)
    )
    save_plot(p_overall_two_model, file.path(plot_path_base, paste0(tolower(gsub(" ", "_", model1_name)), "_vs_", tolower(gsub(" ", "_", model2_name)), ".pdf")))

    p_by_table_two_model <- plot_accuracy_by_table_two_boxplots(
        processed_data[[model1_name]]$correct_by_table, processed_data[[model2_name]]$correct_by_table,
        model1_name, model2_name, colors_two_model,
        plot_title_suffix = paste(experiment_prefix, model1_name, "vs", model2_name)
    )
    save_plot(p_by_table_two_model, file.path(plot_path_base, paste0(tolower(gsub(" ", "_", model1_name)), "_vs_", tolower(gsub(" ", "_", model2_name)), "_by_table.pdf")), width=15)
  }

  # Compare first two models with naive baseline (if naive and at least two models exist)
  if (length(model_names) >= 2 && !is.null(processed_data[[model_names[1]]]$correct) && !is.null(processed_data[[model_names[2]]]$correct) && !is.null(processed_naive) && !is.null(processed_naive$correct)) {
    model1_name <- model_names[1]
    model2_name <- model_names[2]
    
    boxplot_dfs <- list(processed_data[[model1_name]]$correct, processed_data[[model2_name]]$correct)
    boxplot_names <- c(model1_name, model2_name)
    point_df <- processed_naive$correct
    point_name <- naive_baseline_name
    
    colors_three_model <- setNames(c("black", "blue", "red"), c(model1_name, model2_name, naive_baseline_name))

    p_overall_three <- plot_overall_accuracy_multi_model(
      boxplot_dfs, point_df, boxplot_names, point_name, colors_three_model,
      plot_title_suffix = paste(experiment_prefix, model1_name, model2_name, "vs", naive_baseline_name)
    )
    save_plot(p_overall_three, file.path(plot_path_base, paste0(tolower(gsub(" ", "_", model1_name)), "_", tolower(gsub(" ", "_", model2_name)), "_vs_", tolower(gsub(" ", "_", naive_baseline_name)), ".pdf")))

    boxplot_dfs_by_table <- list(processed_data[[model1_name]]$correct_by_table, processed_data[[model2_name]]$correct_by_table)
    point_df_by_table <- processed_naive$correct_by_table

    p_by_table_three <- plot_accuracy_by_table_multi_model(
      boxplot_dfs_by_table, point_df_by_table, boxplot_names, point_name, colors_three_model,
      plot_title_suffix = paste(experiment_prefix, model1_name, model2_name, "vs", naive_baseline_name)
    )
    save_plot(p_by_table_three, file.path(plot_path_base, paste0(tolower(gsub(" ", "_", model1_name)), "_", tolower(gsub(" ", "_", model2_name)), "_vs_", tolower(gsub(" ", "_", naive_baseline_name)), "_by_table.pdf")), width=15)
  }
  
  # Clean up memory for large datasets
  rm(list = names(predictions_list)) # Remove the original large dataframes
  if (!is.null(naive_baseline_df)) rm(naive_baseline_df)
  gc()
  print(paste("--- Finished Analysis Suite for:", experiment_prefix, "---"))
}


# --- Data Analysis Section ---

# Original Row Lock Analysis (Example)
# predictions_global_transformer_row <- load_parquet_data("analysis/data/exp-6-row-locks/predictions.parquet")
# predictions_global_naive_row <- load_parquet_data("analysis/data/exp-10/predictions.parquet")

# run_analysis_suite(
#   predictions_list = list("Global Transformer" = predictions_global_transformer_row),
#   naive_baseline_df = predictions_global_naive_row,
#   naive_baseline_name = "Global Naive Baseline",
#   experiment_prefix = "original_row_locks_global_tf_vs_naive",
#   output_dir = "analysis/refactored_output",
#   filter_csv_data = "data/fixed/row_locks.csv" # Specify the CSV if naive data is mixed
# )

# predictions_local_transformer_row <- load_parquet_data("analysis/data/row_sep/exp-11-row-locks/predictions.parquet")
# predictions_local_naive_row <- load_parquet_data("analysis/data/row_sep/exp-11-row-locks-naive/predictions.parquet")
# predictions_local_transformer_rowid <- load_parquet_data("analysis/data/row_sep/exp-11-row-locks-row_id/predictions.parquet")

# run_analysis_suite(
#   predictions_list = list("Local Transformer" = predictions_local_transformer_row),
#   naive_baseline_df = predictions_local_naive_row, # Using local naive here
#   naive_baseline_name = "Local Naive Baseline",
#   experiment_prefix = "original_row_locks_local_tf_vs_local_naive",
#   output_dir = "analysis/refactored_output"
# )
# # Comparison between Global Transformer and Local Transformer
# if (!is.null(predictions_global_transformer_row) && !is.null(predictions_local_transformer_row)) {
#   run_analysis_suite(
#     predictions_list = list("Global Transformer" = predictions_global_transformer_row, "Local Transformer" = predictions_local_transformer_row),
#     experiment_prefix = "original_row_locks_global_vs_local_tf",
#     output_dir = "analysis/refactored_output"
#   )
# }
# # Comparison between Local Transformer and Local Transformer w/ Row ID
# if (!is.null(predictions_local_transformer_row) && !is.null(predictions_local_transformer_rowid)) {
#   run_analysis_suite(
#     predictions_list = list("Local Transformer" = predictions_local_transformer_row, "Local Transformer RowID" = predictions_local_transformer_rowid),
#     experiment_prefix = "original_row_locks_local_tf_vs_rowid",
#     output_dir = "analysis/refactored_output"
#   )
# }


# Original Table Lock Analysis (Example)
# predictions_global_transformer_tbl <- load_parquet_data("analysis/data/exp-6-table-locks/predictions.parquet", is_table_lock = TRUE)
# predictions_global_naive_tbl <- load_parquet_data("analysis/data/exp-10/predictions.parquet", is_table_lock = TRUE)

# run_analysis_suite(
#   predictions_list = list("Global Transformer" = predictions_global_transformer_tbl),
#   naive_baseline_df = predictions_global_naive_tbl,
#   naive_baseline_name = "Global Naive Baseline",
#   experiment_prefix = "original_table_locks_global_tf_vs_naive",
#   output_dir = "analysis/refactored_output",
#   is_table_lock_experiment = TRUE,
#   filter_csv_data = "data/fixed/table_locks.csv"
# )

# --- SORTED DATA Analysis ---
print("##########################################")
print("--- Starting SORTED Data Analysis ---")
print("##########################################")

# Global models for sorted row locks
preds_sorted_global_tf <- load_parquet_data("analysis/data/exp-17-transformer-sorted-row-locks/predictions.parquet")
preds_sorted_global_lstm <- load_parquet_data("analysis/data/exp-21-lstm-sorted-row-locks/predictions.parquet")
preds_sorted_global_naive <- load_parquet_data("analysis/data/exp-18-naive-sorted-row-locks/predictions.parquet")

run_analysis_suite(
  predictions_list = list("Global Transformer" = preds_sorted_global_tf, "Global LSTM" = preds_sorted_global_lstm),
  naive_baseline_df = preds_sorted_global_naive,
  naive_baseline_name = "Global Naive Baseline",
  experiment_prefix = "sorted_row_locks_global",
  output_dir = "analysis/refactored_output/sorted",
  filter_csv_data = "data/fixed/row_locks.csv"
)

# CLear memory
rm(preds_sorted_global_tf, preds_sorted_global_lstm, preds_sorted_global_naive)

# Local models for sorted row locks
preds_sorted_local_tf <- load_parquet_data("analysis/data/exp-19-transformer-sorted-row-locks/predictions.parquet")
preds_sorted_local_lstm <- load_parquet_data("analysis/data/exp-20-lstm-sorted-row-locks/predictions.parquet")
preds_sorted_local_naive <- load_parquet_data("analysis/data/exp-19-naive-sorted-row-locks/predictions.parquet") # Assuming path, update if different

run_analysis_suite(
  predictions_list = list("Local Transformer" = preds_sorted_local_tf, "Local LSTM" = preds_sorted_local_lstm),
  naive_baseline_df = preds_sorted_local_naive,
  naive_baseline_name = "Local Naive Baseline",
  experiment_prefix = "sorted_row_locks_local",
  output_dir = "analysis/refactored_output/sorted"
  # No filter_csv_data needed if local naive is specific to row_locks already
)

# Local models with Row ID for sorted row locks
preds_sorted_local_tf_rowid <- load_parquet_data("analysis/data/exp-25-transformer-row-id-local-sorted-row-locks/predictions.parquet")
preds_sorted_local_lstm_rowid <- load_parquet_data("analysis/data/exp-23-lstm-row-id-local-sorted-row-locks/predictions.parquet")

# Compare Local TF vs Local TF RowID (Sorted)
if (!is.null(preds_sorted_local_tf) && !is.null(preds_sorted_local_tf_rowid)) {
  run_analysis_suite(
    predictions_list = list("Local Transformer" = preds_sorted_local_tf, "Local Transformer RowID" = preds_sorted_local_tf_rowid),
    experiment_prefix = "sorted_row_locks_local_tf_vs_rowid",
    output_dir = "analysis/refactored_output/sorted"
  )
}
# Compare Local LSTM vs Local LSTM RowID (Sorted)
if (!is.null(preds_sorted_local_lstm) && !is.null(preds_sorted_local_lstm_rowid)) {
  run_analysis_suite(
    predictions_list = list("Local LSTM" = preds_sorted_local_lstm, "Local LSTM RowID" = preds_sorted_local_lstm_rowid),
    experiment_prefix = "sorted_row_locks_local_lstm_vs_rowid",
    output_dir = "analysis/refactored_output/sorted"
  )
}

# Clear Memory
rm(preds_sorted_local_tf, preds_sorted_local_lstm, preds_sorted_local_naive, preds_sorted_local_tf_rowid, preds_sorted_local_lstm_rowid)

# Global models for sorted table locks
preds_sorted_tbl_global_tf <- load_parquet_data("analysis/data/exp-17-transformer-sorted-table-locks/predictions.parquet", is_table_lock = TRUE)
preds_sorted_tbl_global_lstm <- load_parquet_data("analysis/data/exp-21-lstm-sorted-table-locks/predictions.parquet", is_table_lock = TRUE)
preds_sorted_tbl_global_naive <- load_parquet_data("analysis/data/exp-18-naive-sorted-table-locks/predictions.parquet", is_table_lock = TRUE)

run_analysis_suite(
  predictions_list = list("Global Transformer" = preds_sorted_tbl_global_tf, "Global LSTM" = preds_sorted_tbl_global_lstm),
  naive_baseline_df = preds_sorted_tbl_global_naive,
  naive_baseline_name = "Global Naive Baseline",
  experiment_prefix = "sorted_table_locks_global",
  output_dir = "analysis/refactored_output/sorted",
  is_table_lock_experiment = TRUE,
  filter_csv_data = "data/fixed/table_locks.csv"
)

rm(preds_sorted_tbl_global_tf, preds_sorted_tbl_global_lstm, preds_sorted_tbl_global_naive)

# --- DEDUPED Data Analysis ---
print("##########################################")
print("--- Starting DEDUPED Data Analysis ---")
print("##########################################")

# Global models for deduped row locks
preds_deduped_global_tf <- load_parquet_data("analysis/data/exp-26-tranformer-dedupe-row-locks/predictions.parquet")
preds_deduped_global_lstm <- load_parquet_data("analysis/data/exp-28-lstm-dedupe-row-locks/predictions.parquet")
preds_deduped_global_naive <- load_parquet_data("analysis/data/exp-27-naive-dedupe-row-locks/predictions.parquet")

run_analysis_suite(
  predictions_list = list("Global Transformer" = preds_deduped_global_tf, "Global LSTM" = preds_deduped_global_lstm),
  naive_baseline_df = preds_deduped_global_naive,
  naive_baseline_name = "Global Naive Baseline",
  experiment_prefix = "deduped_row_locks_global",
  output_dir = "analysis/refactored_output/deduped",
  filter_csv_data = "data/fixed/row_locks.csv"
)

rm(preds_deduped_global_tf, preds_deduped_global_lstm, preds_deduped_global_naive)

# Local models for deduped row locks (Paths are for sorted in original, assuming these are the deduped local)
preds_deduped_local_tf <- load_parquet_data("analysis/data/exp-32-transformer-sorted-row-locks/predictions.parquet") # Path might be exp-XX-transformer-deduped-local
preds_deduped_local_lstm <- load_parquet_data("analysis/data/exp-31-lstm-sorted-row-locks/predictions.parquet")   # Path might be exp-XX-lstm-deduped-local
preds_deduped_local_naive <- load_parquet_data("analysis/data/exp-30-naive-sorted-row-locks/predictions.parquet") # Path might be exp-XX-naive-deduped-local

run_analysis_suite(
  predictions_list = list("Local Transformer" = preds_deduped_local_tf, "Local LSTM" = preds_deduped_local_lstm),
  naive_baseline_df = preds_deduped_local_naive,
  naive_baseline_name = "Local Naive Baseline",
  experiment_prefix = "deduped_row_locks_local", # Changed from "sorted"
  output_dir = "analysis/refactored_output/deduped"
)

rm(preds_deduped_local_tf, preds_deduped_local_lstm, preds_deduped_local_naive)

# Global models for deduped table locks
preds_deduped_tbl_global_tf <- load_parquet_data("analysis/data/exp-26-tranformer-dedupe-table-locks/predictions.parquet", is_table_lock = TRUE)
preds_deduped_tbl_global_lstm <- load_parquet_data("analysis/data/exp-28-lstm-dedupe-table-locks/predictions.parquet", is_table_lock = TRUE)
preds_deduped_tbl_global_naive <- load_parquet_data("analysis/data/exp-27-naive-dedupe-table-locks/predictions.parquet", is_table_lock = TRUE)

run_analysis_suite(
  predictions_list = list("Global Transformer" = preds_deduped_tbl_global_tf, "Global LSTM" = preds_deduped_tbl_global_lstm),
  naive_baseline_df = preds_deduped_tbl_global_naive,
  naive_baseline_name = "Global Naive Baseline",
  experiment_prefix = "deduped_table_locks_global",
  output_dir = "analysis/refactored_output/deduped",
  is_table_lock_experiment = TRUE,
  filter_csv_data = "data/fixed/table_locks.csv"
)

rm(preds_deduped_tbl_global_tf, preds_deduped_tbl_global_lstm, preds_deduped_tbl_global_naive)

# --- ROUNDED Data Analysis (Randomized Page IDs) ---
print("##########################################")
print("--- Starting ROUNDED Data Analysis ---") # Renamed from DEDUPED to ROUNDED as per exp names
print("##########################################")

# Global models for rounded row locks
preds_rounded_global_tf <- load_parquet_data("analysis/data/exp-33-tranformer-random-row-locks/predictions.parquet")
preds_rounded_global_lstm <- load_parquet_data("analysis/data/exp-34-lstm-random-row-locks/predictions.parquet")
preds_rounded_global_naive <- load_parquet_data("analysis/data/exp-35-naive-rounded-row-locks/predictions.parquet")

run_analysis_suite(
  predictions_list = list("Global Transformer" = preds_rounded_global_tf, "Global LSTM" = preds_rounded_global_lstm),
  naive_baseline_df = preds_rounded_global_naive,
  naive_baseline_name = "Global Naive Baseline",
  experiment_prefix = "rounded_row_locks_global",
  output_dir = "analysis/refactored_output/rounded",
  filter_csv_data = "data/fixed/row_locks.csv"
)

rm(preds_rounded_global_tf, preds_rounded_global_lstm, preds_rounded_global_naive)

# Local models for rounded row locks
preds_rounded_local_tf <- load_parquet_data("analysis/data/exp-38-transformer-local-rounded/predictions.parquet")
preds_rounded_local_lstm <- load_parquet_data("analysis/data/exp-37-lstm-local-rounded/predictions.parquet")
preds_rounded_local_naive <- load_parquet_data("analysis/data/exp-36-naive-local-rounded-row-locks/predictions.parquet")

run_analysis_suite(
  predictions_list = list("Local Transformer" = preds_rounded_local_tf, "Local LSTM" = preds_rounded_local_lstm),
  naive_baseline_df = preds_rounded_local_naive,
  naive_baseline_name = "Local Naive Baseline",
  experiment_prefix = "rounded_row_locks_local",
  output_dir = "analysis/refactored_output/rounded"
)

rm(preds_rounded_local_tf, preds_rounded_local_lstm, preds_rounded_local_naive)

# Global models for rounded table locks
preds_rounded_tbl_global_tf <- load_parquet_data("analysis/data/exp-33-tranformer-random-table-locks/predictions.parquet", is_table_lock = TRUE)
preds_rounded_tbl_global_lstm <- load_parquet_data("analysis/data/exp-34-lstm-random-table-locks/predictions.parquet", is_table_lock = TRUE)
preds_rounded_tbl_global_naive <- load_parquet_data("analysis/data/exp-35-naive-rounded-table-locks/predictions.parquet", is_table_lock = TRUE)

run_analysis_suite(
  predictions_list = list("Global Transformer" = preds_rounded_tbl_global_tf, "Global LSTM" = preds_rounded_tbl_global_lstm),
  naive_baseline_df = preds_rounded_tbl_global_naive,
  naive_baseline_name = "Global Naive Baseline",
  experiment_prefix = "rounded_table_locks_global",
  output_dir = "analysis/refactored_output/rounded",
  is_table_lock_experiment = TRUE,
  filter_csv_data = "data/fixed/table_locks.csv"
)

rm(preds_rounded_tbl_global_tf, preds_rounded_tbl_global_lstm, preds_rounded_tbl_global_naive)

# --- CUT Data Analysis (Binned Page IDs) ---
print("##########################################")
print("--- Starting CUT Data Analysis ---")
print("##########################################")

# Global models for cut row locks
preds_cut_global_tf <- load_parquet_data("analysis/data/exp-39-tranformer-rounded-cut-row-locks/predictions.parquet")
preds_cut_global_lstm <- load_parquet_data("analysis/data/exp-40-lstm-rounded-cut-row-locks/predictions.parquet")
preds_cut_global_naive <- load_parquet_data("analysis/data/exp-41-naive-rounded-cut-row-locks/predictions.parquet")

run_analysis_suite(
  predictions_list = list("Global Transformer" = preds_cut_global_tf, "Global LSTM" = preds_cut_global_lstm),
  naive_baseline_df = preds_cut_global_naive,
  naive_baseline_name = "Global Naive Baseline",
  experiment_prefix = "cut_row_locks_global",
  output_dir = "analysis/refactored_output/cut",
  filter_csv_data = "data/fixed/row_locks.csv"
)

rm(preds_cut_global_tf, preds_cut_global_lstm, preds_cut_global_naive)

# Local models for cut row locks
preds_cut_local_tf <- load_parquet_data("analysis/data/exp-44-transformer-local-rounded-cut/predictions.parquet")
preds_cut_local_lstm <- load_parquet_data("analysis/data/exp-43-lstm-local-rounded-cut/predictions.parquet")
preds_cut_local_naive <- load_parquet_data("analysis/data/exp-42-naive-local-rounded-cut-row-locks/predictions.parquet")

run_analysis_suite(
  predictions_list = list("Local Transformer" = preds_cut_local_tf, "Local LSTM" = preds_cut_local_lstm),
  naive_baseline_df = preds_cut_local_naive,
  naive_baseline_name = "Local Naive Baseline",
  experiment_prefix = "cut_row_locks_local",
  output_dir = "analysis/refactored_output/cut"
)

rm(preds_cut_local_tf, preds_cut_local_lstm, preds_cut_local_naive)

# Global models for cut table locks
preds_cut_tbl_global_tf <- load_parquet_data("analysis/data/exp-39-tranformer-rounded-cut-table-locks/predictions.parquet", is_table_lock = TRUE)
preds_cut_tbl_global_lstm <- load_parquet_data("analysis/data/exp-40-lstm-rounded-cut-table-locks/predictions.parquet", is_table_lock = TRUE)
preds_cut_tbl_global_naive <- load_parquet_data("analysis/data/exp-41-naive-rounded-cut-table-locks/predictions.parquet", is_table_lock = TRUE)

run_analysis_suite(
  predictions_list = list("Global Transformer" = preds_cut_tbl_global_tf, "Global LSTM" = preds_cut_tbl_global_lstm),
  naive_baseline_df = preds_cut_tbl_global_naive,
  naive_baseline_name = "Global Naive Baseline",
  experiment_prefix = "cut_table_locks_global",
  output_dir = "analysis/refactored_output/cut",
  is_table_lock_experiment = TRUE,
  filter_csv_data = "data/fixed/table_locks.csv"
)

rm(preds_cut_tbl_global_tf, preds_cut_tbl_global_lstm, preds_cut_tbl_global_naive)

# --- QCUT Data Analysis (Quantile Binned Page IDs) ---
print("##########################################")
print("--- Starting QCUT Data Analysis ---")
print("##########################################")

# Global models for qcut row locks
preds_qcut_global_tf <- load_parquet_data("analysis/data/exp-45-tranformer-rounded-qcut-row-locks/predictions.parquet")
preds_qcut_global_lstm <- load_parquet_data("analysis/data/exp-46-lstm-rounded-qcut-row-locks/predictions.parquet")
preds_qcut_global_naive <- load_parquet_data("analysis/data/exp-47-naive-rounded-qcut-row-locks/predictions.parquet")

run_analysis_suite(
  predictions_list = list("Global Transformer" = preds_qcut_global_tf, "Global LSTM" = preds_qcut_global_lstm),
  naive_baseline_df = preds_qcut_global_naive,
  naive_baseline_name = "Global Naive Baseline",
  experiment_prefix = "qcut_row_locks_global",
  output_dir = "analysis/refactored_output/qcut",
  filter_csv_data = "data/fixed/row_locks.csv"
)

rm(preds_qcut_global_tf, preds_qcut_global_lstm, preds_qcut_global_naive)

# Local models for qcut row locks
preds_qcut_local_tf <- load_parquet_data("analysis/data/exp-50-transformer-local-rounded-qcut/predictions.parquet")
preds_qcut_local_lstm <- load_parquet_data("analysis/data/exp-49-lstm-local-rounded-qcut/predictions.parquet")
preds_qcut_local_naive <- load_parquet_data("analysis/data/exp-48-naive-local-rounded-qcut-row-locks/predictions.parquet")

run_analysis_suite(
  predictions_list = list("Local Transformer" = preds_qcut_local_tf, "Local LSTM" = preds_qcut_local_lstm),
  naive_baseline_df = preds_qcut_local_naive,
  naive_baseline_name = "Local Naive Baseline",
  experiment_prefix = "qcut_row_locks_local",
  output_dir = "analysis/refactored_output/qcut"
)

rm(preds_qcut_local_tf, preds_qcut_local_lstm, preds_qcut_local_naive)

# Global models for qcut table locks
preds_qcut_tbl_global_tf <- load_parquet_data("analysis/data/exp-45-tranformer-rounded-qcut-table-locks/predictions.parquet", is_table_lock = TRUE)
preds_qcut_tbl_global_lstm <- load_parquet_data("analysis/data/exp-46-lstm-rounded-qcut-table-locks/predictions.parquet", is_table_lock = TRUE)
preds_qcut_tbl_global_naive <- load_parquet_data("analysis/data/exp-47-naive-rounded-qcut-table-locks/predictions.parquet", is_table_lock = TRUE)

run_analysis_suite(
  predictions_list = list("Global Transformer" = preds_qcut_tbl_global_tf, "Global LSTM" = preds_qcut_tbl_global_lstm),
  naive_baseline_df = preds_qcut_tbl_global_naive,
  naive_baseline_name = "Global Naive Baseline",
  experiment_prefix = "qcut_table_locks_global",
  output_dir = "analysis/refactored_output/qcut",
  is_table_lock_experiment = TRUE,
  filter_csv_data = "data/fixed/table_locks.csv"
)

rnm(preds_qcut_tbl_global_tf, preds_qcut_tbl_global_lstm, preds_qcut_tbl_global_naive)

# --- CUT-100 Data Analysis (Binned Page IDs with 100 bins) ---
print("##########################################")
print("--- Starting CUT-100 Data Analysis ---")
print("##########################################")

# Global models for cut-100 row locks
preds_cut100_global_tf <- load_parquet_data("analysis/data/exp-39-tranformer-rounded-cut-row-locks_100/predictions.parquet")
preds_cut100_global_lstm <- load_parquet_data("analysis/data/exp-40-lstm-rounded-cut-row-locks_100/predictions.parquet")
preds_cut100_global_naive <- load_parquet_data("analysis/data/exp-41-naive-rounded-cut-row-locks_100/predictions.parquet")

run_analysis_suite(
  predictions_list = list("Global Transformer" = preds_cut100_global_tf, "Global LSTM" = preds_cut100_global_lstm),
  naive_baseline_df = preds_cut100_global_naive,
  naive_baseline_name = "Global Naive Baseline",
  experiment_prefix = "cut100_row_locks_global",
  output_dir = "analysis/refactored_output/cut-100",
  filter_csv_data = "data/fixed/row_locks.csv"
)

rm(preds_cut100_global_tf, preds_cut100_global_lstm, preds_cut100_global_naive)

# Local models for cut-100 row locks
preds_cut100_local_tf <- load_parquet_data("analysis/data/exp-44-transformer-local-rounded-cut_100/predictions.parquet")
preds_cut100_local_lstm <- load_parquet_data("analysis/data/exp-43-lstm-local-rounded-cut_100/predictions.parquet")
preds_cut100_local_naive <- load_parquet_data("analysis/data/exp-42-naive-local-rounded-cut-row-locks_100/predictions.parquet")

run_analysis_suite(
  predictions_list = list("Local Transformer" = preds_cut100_local_tf, "Local LSTM" = preds_cut100_local_lstm),
  naive_baseline_df = preds_cut100_local_naive,
  naive_baseline_name = "Local Naive Baseline",
  experiment_prefix = "cut100_row_locks_local",
  output_dir = "analysis/refactored_output/cut-100"
)

rm(preds_cut100_local_tf, preds_cut100_local_lstm, preds_cut100_local_naive)

# Global models for cut-100 table locks
preds_cut100_tbl_global_tf <- load_parquet_data("analysis/data/exp-39-tranformer-rounded-cut-table-locks_100/predictions.parquet", is_table_lock = TRUE)
preds_cut100_tbl_global_lstm <- load_parquet_data("analysis/data/exp-40-lstm-rounded-cut-table-locks_100/predictions.parquet", is_table_lock = TRUE)
preds_cut100_tbl_global_naive <- load_parquet_data("analysis/data/exp-41-naive-rounded-cut-table-locks_100/predictions.parquet", is_table_lock = TRUE)

run_analysis_suite(
  predictions_list = list("Global Transformer" = preds_cut100_tbl_global_tf, "Global LSTM" = preds_cut100_tbl_global_lstm),
  naive_baseline_df = preds_cut100_tbl_global_naive,
  naive_baseline_name = "Global Naive Baseline",
  experiment_prefix = "cut100_table_locks_global",
  output_dir = "analysis/refactored_output/cut-100",
  is_table_lock_experiment = TRUE,
  filter_csv_data = "data/fixed/table_locks.csv"
)

rm(preds_cut100_tbl_global_tf, preds_cut100_tbl_global_lstm, preds_cut100_tbl_global_naive)


# --- Dataset Analysis (Original Code, slightly adapted) ---
print("##########################################")
print("--- Dataset characteristics ---")
print("##########################################")
analyze_dataset_characteristics <- function(output_dir = "analysis/refactored_output/dataset_stats") {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
    
    data_row_locks_raw <- read_csv("data/fixed/row_locks.csv")
    data_table_locks_raw <- read_csv("data/fixed/table_locks.csv")

    # Prep function
    prep_data <- function(df) {
      df %>%
        mutate(TABNAME = str_trim(str_remove(TABNAME, "_")),
               TABSCHEMA = str_trim(str_remove(TABSCHEMA, "_"))) %>%
        filter(TABSCHEMA != "SYSIBM") # Ensure SYSIBM is filtered
    }

    data_row_locks <- prep_data(data_row_locks_raw)
    data_table_locks <- prep_data(data_table_locks_raw)

    print("Summary for Row Locks Data:")
    summary(data_row_locks)
    print("Summary for Table Locks Data:")
    summary(data_table_locks)

    tab_counts_row_lock <- data_row_locks %>%
      group_by(TABNAME) %>%
      summarise(count = n(), .groups = 'drop') %>%
      arrange(desc(count))
    print("Row Lock Table Counts:")
    print(tab_counts_row_lock)
    write_csv(tab_counts_row_lock, file.path(output_dir, "row_lock_table_counts.csv"))

    tab_counts_table_lock <- data_table_locks %>%
      group_by(TABNAME) %>%
      summarise(count = n(), .groups = 'drop') %>%
      arrange(desc(count))
    print("Table Lock Table Counts:")
    print(tab_counts_table_lock)
    write_csv(tab_counts_table_lock, file.path(output_dir, "table_lock_table_counts.csv"))

    # Page ID counts for WAREHOUS (if exists and has PAGEID)
    if ("WAREHOUS" %in% data_row_locks$TABNAME && "PAGEID" %in% colnames(data_row_locks)) {
        page_counts_warehouse_row_lock <- data_row_locks %>%
            filter(TABNAME == "WAREHOUS") %>%
            group_by(TABNAME, PAGEID) %>%
            summarise(count = n(), .groups = 'drop') %>%
            arrange(desc(count))
        print("PageID Counts for WAREHOUS table in Row Locks:")
        print(page_counts_warehouse_row_lock)
        write_csv(page_counts_warehouse_row_lock, file.path(output_dir, "row_lock_warehouse_pageid_counts.csv"))
    } else {
        print("WAREHOUS table or PAGEID column not found in row_locks data for PageID count analysis.")
    }
}

analyze_dataset_characteristics()