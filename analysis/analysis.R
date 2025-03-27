require(tidyverse)
require(arrow)  # to read parquet
library(dplyr)
library(ggplot2)

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
  
  # Filter out the last filter_tail_ns nanoseconds of predictions
  # Default is 3e11 nanoseconds (5 minutes) 
  if (filter_tail_ns > 0) {
    predictions <- predictions %>%
      filter(in_lock_start_time < max(in_lock_start_time) - filter_tail_ns)
  }
  
  return(predictions)
}

# Lets make a function that takes in a df and groups by horizon and iteration
# and makes sure that no horizon has less than 10 iterations
# if it does, it will print out the horizons that have less than 10 iterations
# and throw an error
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

horizon_iteration_performance_by_table <- function(predictions) {
  correct_by_table <- predictions %>%
    group_by(horizon, iteration, unique_id, gt_table) %>%
    summarise(is_correct = all(is_correct), .groups='drop') %>%
    group_by(horizon, iteration, gt_table) %>%
    summarise(mean_percent_correct = mean(is_correct), .groups='drop')

  print(correct_by_table %>%
    group_by(horizon, gt_table) %>%
    summarise(percent_correct = mean(mean_percent_correct), .groups='drop'))
  
  # Count how many rows for each (horizon, iteration, ground_truth, prediction)
  conf <- predictions %>%
    group_by(horizon, iteration, gt_table, pred_table) %>%
    summarise(n = n(), .groups='drop')
  
  # Collect all unique table names that appear as either gt or prediction
  all_tables <- union(conf$gt_table, conf$pred_table)
  
  # We want to produce results for every horizon, iteration, and table
  all_combos <- conf %>%
    distinct(horizon, iteration) %>%
    tidyr::crossing(table = all_tables)
  
  # Join to get TP, FP, FN counts for each "table" in all_combos
  metrics <- all_combos %>%
    # 1) True Positives: gt_table == pred_table == table
    left_join(
      conf %>%
        filter(gt_table == pred_table) %>%
        rename(table = gt_table) %>%
        select(horizon, iteration, table, n),
      by = c("horizon", "iteration", "table")
    ) %>%
    rename(tp = n) %>%
    mutate(tp = tidyr::replace_na(tp, 0)) %>%
    
    # 2) False Positives: gt_table != t but pred_table == t
    left_join(
      conf %>%
        group_by(horizon, iteration, pred_table) %>%
        summarise(fp = sum(ifelse(gt_table != pred_table, n, 0)),
                  .groups='drop') %>%
        rename(table = pred_table),
      by = c("horizon", "iteration", "table")
    ) %>%
    mutate(fp = tidyr::replace_na(fp, 0)) %>%
    
    # 3) False Negatives: gt_table == t but pred_table != t
    left_join(
      conf %>%
        group_by(horizon, iteration, gt_table) %>%
        summarise(fn = sum(ifelse(gt_table != pred_table, n, 0)),
                  .groups='drop') %>%
        rename(table = gt_table),
      by = c("horizon", "iteration", "table")
    ) %>%
    mutate(fn = tidyr::replace_na(fn, 0)) %>%
    
    # Compute Precision, Recall, F1
    mutate(
      precision = ifelse(tp + fp == 0, NA, tp / (tp + fp)),
      recall    = ifelse(tp + fn == 0, NA, tp / (tp + fn)),
      f1        = ifelse(
        !is.na(precision) & !is.na(recall) & (precision + recall > 0),
        2 * precision * recall / (precision + recall),
        NA
      )
    )  %>%
    # Rename the 'table' column to 'gt_table' so that it matches
    # the name in `correct_by_table`
    rename(gt_table = table)
  
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

export_csv_by_table <- function(df, path) {
  df %>%
    group_by(horizon, gt_table) %>%
    summarise(
      mean_percent_correct_csv = mean(mean_percent_correct),
      median_percent_correct_csv = median(mean_percent_correct),
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

horizon_labels <- c(
  "1" = "Horizon: 1",
  "2" = "Horizon: 2",
  "3" = "Horizon: 3",
  "4" = "Horizon: 4"
)


plot_precision_recall <- function(correct_by_table) {
  
  # Pivot to long format
  correct_by_table_long <- correct_by_table %>%
    pivot_longer(
      cols = c("precision", "recall", "f1"),
      names_to = "metric",
      values_to = "value"
    )
  
  # Build the plot
  ggplot(correct_by_table_long, aes(x = horizon, y = value, fill = metric)) +
    # Boxplot (dodged)
    geom_boxplot(
      position = position_dodge(width = 0.8),
      alpha = 0.5
    ) +
    # Trend lines for each metric
    stat_summary(
      aes(group = metric, color = metric),
      fun = mean,
      geom = "line",
      position = position_dodge(width = 0.8)
    ) +
    # Points for the mean values
    stat_summary(
      aes(group = metric, color = metric),
      fun = mean,
      geom = "point",
      position = position_dodge(width = 0.8)
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

library(ggplot2)
library(dplyr)

plot_accuracy_over_time <- function(predictions_cur, num_bins = 40) {
  # Process the data as before
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
  
  # Calculate the bin width based on the number of bins
  x_range <- range(correct$in_lock_start_time_rel, na.rm = TRUE)  # Range of x-axis values
  binwidth <- (x_range[2] - x_range[1]) / num_bins  # Divide the range by number of bins
  
  # Create the plot
  p <- ggplot(correct, aes(x = in_lock_start_time_rel, y = is_correct_num)) +
    # stat_summary_bin with dynamically calculated binwidth
    stat_summary_bin(
      fun = "mean",        # how to aggregate y-values in each bin
      geom = "col",        # use columns (like a histogram)
      binwidth = binwidth  # Use the calculated binwidth
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
  # Ensure we have the same number of names as data frames
  stopifnot(length(dfs) == length(df_names))
  
  # Prepare a list to collect processed data for each data frame
  all_data_list <- list()
  
  # Process each data frame individually, then add a column for dataset name
  for (i in seq_along(dfs)) {
    predictions_cur <- dfs[[i]]
    dataset_name <- df_names[[i]]
    
    # Summarize correctness across iteration, unique_id, horizon, and in_lock_start_time
    correct <- predictions_cur %>%
      group_by(
        horizon, iteration, unique_id,
        in_lock_start_time
      ) %>%
      summarise(
        is_correct = all(is_correct),
        .groups = 'drop'
      )
    
    # Convert logical to numeric for plotting
    correct$is_correct_num <- as.numeric(correct$is_correct)
    
    # Make lock start time relative to the earliest time in this particular data set
    correct$in_lock_start_time_rel <- as.numeric(
      correct$in_lock_start_time - min(correct$in_lock_start_time)
    )
    
    # Add a column for the dataset name (for legend)
    correct$dataset_name <- dataset_name
    
    # Store
    all_data_list[[i]] <- correct
  }
  
  # Combine all processed data frames
  all_data <- dplyr::bind_rows(all_data_list)
  
  # Calculate bin width based on the overall range (across all data)
  x_range <- range(all_data$in_lock_start_time_rel, na.rm = TRUE)
  binwidth <- (x_range[2] - x_range[1]) / num_bins
  
  # Create the plot
  p <- ggplot(all_data, aes(
    x = in_lock_start_time_rel,
    y = is_correct_num,
    color = dataset_name
  )) +
    stat_summary_bin(
      fun = "mean",       # aggregate y-values in each bin by mean
      geom = "line",      # use line plots
      binwidth = binwidth
    ) +
    facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
    labs(
      x = "Relative Lock End Time (nanoseconds)",
      y = "Percent Correct",
      color = "Dataset"   # legend title
    ) +
    scale_x_continuous(labels = scales::scientific) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_light()
  
  return(p)
}



# Command to get prediction parquets:
# rsync -aR --prune-empty-dirs --include="*/" --include="*/predictions.parquet" --exclude="*" . ../../analysis/data


#####################
#####################

# Lets look at Row lock performance

predictions <- load_parquet("analysis/data/exp-6-row-locks/predictions.parquet")
check_iterations(predictions)

predictions_naive <- load_parquet("analysis/data/exp-10/predictions.parquet") %>%
  filter(data == "data/fixed/row_locks.csv")


p <- plot_accuracy_over_time(predictions)
print(p)

ggsave(
  "analysis/plots/global_transformer_accuracy_over_time.pdf",
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)

p <- plot_accuracy_over_time(predictions_naive)
print(p)

ggsave(
  "analysis/plots/global_naive_baseline_accuracy_over_time.pdf",
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)



correct <- horizon_iteration_performance(predictions)
correct_by_table <- horizon_iteration_performance_by_table(predictions)

correct_naive <- horizon_iteration_performance(predictions_naive)
correct_naive_by_table <- horizon_iteration_performance_by_table(predictions_naive)

# Offload predictions to free up memory
rm(predictions)
rm(predictions_naive)
gc()

export_csv(correct, "analysis/tables/global_transformer_performance.csv")
export_csv(correct_naive, "analysis/tables/global_naive_baseline_performance.csv")
export_csv_by_table(correct_by_table, "analysis/tables/global_transformer_performance_by_table.csv")
export_csv_by_table(correct_naive_by_table, "analysis/tables/global_naive_baseline_performance_by_table.csv")


# Box plot w/ correct and scatter plot w/ correct_naive
ggplot() +
  geom_boxplot(
    data = correct,
    aes(x = horizon, y = mean_percent_correct, color = "Global Transformer"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive,
    aes(x = horizon, y = mean_percent_correct, color = "Global Naive Baseline"),
    size = 2
  ) +
  labs(
#    title = "Global, Transformer and Naive Baseline Performance: Horizon vs. Percent Correct",
    x = "Horizon",
    y = "Percent Correct",
    color = "Legend"
  ) +  # Ensures only one legend title
  scale_color_manual(values = c(
    "Global Transformer" = "black",
    "Global Naive Baseline" = "red"
  )) +
  theme_light()

ggsave(
  "analysis/plots/global_transformer_vs_naive_baseline.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)

# Box plot w/ correct showing Global, Transformer: GT_table vs Percent Correct by Horizon

ggplot() +
  geom_boxplot(
    data = correct_by_table,
    aes(x = gt_table, y = mean_percent_correct, fill=horizon),
    alpha = 0.5
  ) +
  labs(
#    title = "Global, Transformer Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    fill = "Horizon"
  ) +
  theme_light()

ggsave(
  "analysis/plots/global_transformer_by_table.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)




ggplot() +
  geom_boxplot(
    data = correct_by_table,
    aes(x = gt_table, y = mean_percent_correct, color = "Global Transformer"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive_by_table,
    aes(x = gt_table, y = mean_percent_correct, color = "Global Naive Baseline"),
    size = 2,
  ) +
  facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
  labs(
#    title = "Global, Transformer vs Naive Baseline Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    color = "Model"
  ) +
  scale_color_manual(values = c("Global Transformer" = "black", "Global Naive Baseline" = "red")) + # Red for scatter plot
  theme_light() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ggsave(
  "analysis/plots/global_transformer_vs_naive_baseline_by_table.pdf",
  width = 15,
  height = 6,
  units = "in",
  dpi = 300
)

# Lets load the data for local transformer models
predictions_local <- load_parquet("analysis/data/row_sep/exp-11-row-locks/predictions.parquet")
check_iterations(predictions_local)
correct_local <- horizon_iteration_performance(predictions_local)
correct_local_by_table <- horizon_iteration_performance_by_table(predictions_local)


correct_local <- horizon_iteration_performance(
  predictions_local %>%
    filter(!(as.numeric(as.character(horizon))>1 & gt_table == "orderline"))
)
correct_local_by_table <- horizon_iteration_performance_by_table(
  predictions_local %>%
    filter(!(as.numeric(as.character(horizon))>1 & gt_table == "orderline"))
)

# Offload predictions to free up memory
rm(predictions_local)
gc()


export_csv(correct_local, "analysis/tables/local_transformer_performance.csv")
export_csv_by_table(correct_local_by_table, "analysis/tables/local_transformer_performance_by_table.csv")
export_csv(correct_local, "analysis/tables/local_transformer_no_orderline_gt_h1_performance.csv")
export_csv_by_table(correct_local_by_table, "analysis/tables/local_transformer_no_orderline_gt_h1_performance_by_table.csv")

ggplot() +
  geom_boxplot(
    data = correct_local_by_table,
    aes(x = gt_table, y = mean_percent_correct, fill=horizon),
    alpha = 0.5
  ) +
  labs(
#    title = "Local Transformer Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    fill = "Horizon"
  ) +
  theme_light()

ggsave(
  "analysis/plots/local_transformer_by_table.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)

# Lets combine the Global and Local Transformer plots so that each table shows the performance of both models

# Combine the data
correct_by_table$Model <- "Global Transformer"
correct_local_by_table$Model <- "Local Transformer"

combined_correct_by_table <- bind_rows(correct_by_table, correct_local_by_table)

# Plot with both Global and Local Transformer performances
ggplot(combined_correct_by_table, aes(x = gt_table, y = mean_percent_correct, fill = Model)) +
  geom_boxplot(alpha = 0.5) +
  facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
  labs(
#    title = "Global vs Local Transformer Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    fill = "Model"
  ) +
  theme_light() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ggsave(
  "analysis/plots/global_vs_local_transformer_by_table.pdf",
  width = 15,
  height = 6,
  units = "in",
  dpi = 300
)

# Lets plot local transformer vs global Naive baseline (i.e., correct_naive)
# The plots should be Horizon vs Percent Correct, where local transformer is a box plot and global naive is a scatter plot

ggplot() +
  geom_boxplot(
    data = correct_local,
    aes(x = horizon, y = mean_percent_correct, color = "Local Transformer"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive,
    aes(x = horizon, y = mean_percent_correct, color = "Global Naive Baseline"),
    size = 2
  ) +
  labs(
#    title = "Local Transformer vs Global Naive Baseline Performance: Horizon vs. Percent Correct",
    x = "Horizon",
    y = "Percent Correct",
    color = "Legend"
  ) +  # Ensures only one legend title
  scale_color_manual(values = c(
    "Local Transformer" = "black",
    "Global Naive Baseline" = "red"
  )) +
  theme_light()

ggsave(
  "analysis/plots/local_transformer_vs_global_naive_baseline.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)

predictions_naive_local <- load_parquet("analysis/data/row_sep/exp-11-row-locks-naive/predictions.parquet")

correct_naive_local <- horizon_iteration_performance(predictions_naive_local)
correct_naive_local_by_table <- horizon_iteration_performance_by_table(predictions_naive_local)

rm(predictions_naive_local)
gc()

export_csv(correct_naive_local, "analysis/tables/local_naive_baseline_performance.csv")
export_csv_by_table(correct_naive_local_by_table, "analysis/tables/local_naive_baseline_performance_by_table.csv")

ggplot() +
  geom_boxplot(
    data = correct_local,
    aes(x = horizon, y = mean_percent_correct, color = "Local Transformer"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive_local,
    aes(x = horizon, y = mean_percent_correct, color = "Local Naive Baseline"),
    size = 2
  ) +
  labs(
#    title = "Local Transformer vs Local Naive Baseline Performance: Horizon vs. Percent Correct",
    x = "Horizon",
    y = "Percent Correct",
    color = "Legend"
  ) +  # Ensures only one legend title
  scale_color_manual(values = c(
    "Local Transformer" = "black",
    "Local Naive Baseline" = "red"
  )) +
  theme_light()

ggsave(
  "analysis/plots/local_transformer_vs_local_naive_baseline.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)

correct_naive_local_by_table$Model <- "Local Naive Baseline"

# Modify the plot: Local Transformer as a box plot, Local Naive Baseline as a scatter plot
ggplot() +
  geom_boxplot(
    data = correct_local_by_table,
    aes(x = gt_table, y = mean_percent_correct, color = "Local Transformer"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive_local_by_table,
    aes(x = gt_table, y = mean_percent_correct, color = "Local Naive Baseline"),
    size = 2,
  ) +
  facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
  labs(
#    title = "Local Transformer vs Local Naive Baseline Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    color = "Model"
  ) +
  scale_color_manual(values = c("Local Transformer" = "black", "Local Naive Baseline" = "red")) + # Red for scatter plot
  theme_light() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# Save the updated plot
ggsave(
  "analysis/plots/local_transformer_vs_local_naive_baseline_by_table.pdf",
  width = 15,
  height = 6,
  units = "in",
  dpi = 300
)


predictions_local_rowid <- load_parquet("analysis/data/row_sep/exp-11-row-locks-row_id/predictions.parquet")
check_iterations(predictions_local_rowid)

correct_local_rowid <- horizon_iteration_performance(predictions_local_rowid)
correct_local_rowid_by_table <- horizon_iteration_performance_by_table(predictions_local_rowid)

rm(predictions_local_rowid)
gc()

export_csv(correct_local_rowid, "analysis/tables/local_transformer_rowid_performance.csv")
export_csv_by_table(correct_local_rowid_by_table, "analysis/tables/local_transformer_rowid_performance_by_table.csv")

ggplot() +
  geom_boxplot(
    data = correct_local,
    aes(x = horizon, y = mean_percent_correct, color = "Local Transformer"),
    alpha = 0.5
  ) +
  geom_boxplot(
    data = correct_local_rowid,
    aes(x = horizon, y = mean_percent_correct, color = "Local Transformer with Row ID"),
    alpha = 0.5
  ) +
  labs(
#    title = "Local Transformer w/ & w/o Row ID Performance (Excludes 'orderline' table for Horizon>1): Horizon vs. Percent Correct",
    x = "Horizon",
    y = "Percent Correct",
    color = "Legend"
  ) +  # Ensures only one legend title
  scale_color_manual(values = c(
    "Local Transformer" = "black",
    "Local Transformer with Row ID" = "red"
  )) +
  theme_light()


ggsave(
  "analysis/plots/local_transformer_vs_local_transformer_rowid.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)


correct_local_rowid_by_table$Model <- "Local Transformer with Row ID"
correct_local_by_table$Model <- "Local Transformer"

combined_correct_local_by_table <- bind_rows(
  correct_local_by_table, correct_local_rowid_by_table)

# Plot with both Global and Local Transformer performances
ggplot(combined_correct_local_by_table, aes(x = gt_table, y = mean_percent_correct, fill = Model)) +
  geom_boxplot(alpha = 0.5) +
  facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
  labs(
#    title = "Local Transformer w/ & w/o Row ID Performance (Excludes 'orderline' table for Horizon>1): Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    fill = "Model"
  ) +
  theme_light() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ggsave(
  "analysis/plots/local_transformer_vs_local_transformer_rowid_by_table.pdf",
  width = 15,
  height = 6,
  units = "in",
  dpi = 300
)


#####################
#####################

# Lets look at table lock performance

predictions_table <- load_parquet("analysis/data/exp-6-table-locks/predictions.parquet", is_table_lock=TRUE)
check_iterations(predictions_table)

predictions_naive_table <- load_parquet("analysis/data/exp-10/predictions.parquet", is_table_lock=TRUE) %>%
  filter(data == "data/fixed/table_locks.csv")

correct_table <- horizon_iteration_performance(predictions_table)
correct_table_by_table <- horizon_iteration_performance_by_table(predictions_table)

correct_naive_table <- horizon_iteration_performance(predictions_naive_table)
correct_naive_table_by_table <- horizon_iteration_performance_by_table(predictions_naive_table)

rm(predictions_table)
rm(predictions_naive_table)
gc()

export_csv(correct_table, "analysis/tables/table-lock_transformer_performance.csv")
export_csv(correct_naive_table, "analysis/tables/table-lock_naive_baseline_performance.csv")
export_csv_by_table(correct_table_by_table, "analysis/tables/table-lock_transformer_performance_by_table.csv")
export_csv_by_table(correct_naive_table_by_table, "analysis/tables/table-lock_naive_baseline_performance_by_table.csv")


# Box plot w/ correct and scatter plot w/ correct_naive
ggplot() +
  geom_boxplot(
    data = correct_table,
    aes(x = horizon, y = mean_percent_correct, color = "Model Predictions"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive_table,
    aes(x = horizon, y = mean_percent_correct, color = "Naive Baseline"),
    size = 2
  ) +
  labs(
#    title = "Table Lock, Transformer and Naive Baseline Performance: Horizon vs. Percent Correct",
    x = "Horizon",
    y = "Percent Correct",
    color = "Legend"
  ) +  # Ensures only one legend title
  scale_color_manual(values = c(
    "Model Predictions" = "black",
    "Naive Baseline" = "red"
  )) +
  theme_light()

ggsave(
  "analysis/plots/table-lock_transformer_vs_naive_baseline.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)


ggplot() +
  geom_boxplot(
    data = correct_table_by_table,
    aes(x = gt_table, y = mean_percent_correct, color = "Transformer"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive_table_by_table,
    aes(x = gt_table, y = mean_percent_correct, color = "Naive Baseline"),
    size = 2,
  ) +
  facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
  labs(
#    title = "Table Lock, Transformer vs Naive Baseline Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    color = "Model"
  ) +
  scale_color_manual(values = c("Transformer" = "black", "Naive Baseline" = "red")) + # Red for scatter plot
  theme_light() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ggsave(
  "analysis/plots/table-lock_transformer_vs_naive_baseline_by_table.pdf",
  width = 15,
  height = 6,
  units = "in",
  dpi = 300
)

p <- plot_precision_recall(correct_table_by_table)
print(p)

ggsave(
  "analysis/plots/table-lock_transformer_precision_recall.pdf",
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)

p <- plot_precision_recall(correct_naive_table_by_table)
print(p)

ggsave(
  "analysis/plots/table-lock_naive_baseline_precision_recall.pdf",
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)

#####################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
#####################






# SORTED DATA






#####################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
##########################################
#####################



# Lets look at Row lock performance

predictions <- load_parquet("analysis/data/exp-17-transformer-sorted-row-locks/predictions.parquet")
check_iterations(predictions)

predictions_lstm <- load_parquet("analysis/data/exp-21-lstm-sorted-row-locks/predictions.parquet")
check_iterations(predictions_lstm)

predictions_naive <- load_parquet("analysis/data/exp-18-naive-sorted-row-locks/predictions.parquet") %>%
  filter(data == "data/fixed/row_locks.csv")


p <- plot_accuracy_over_time_list(
  list(predictions, predictions_lstm, predictions_naive),
  c("Global Transformer", "Global LSTM", "Global Naive Baseline")
)
print(p)

ggsave(
  "analysis/plots/sorted/global_transformer_lstm_naive_baseline_accuracy_over_time.pdf",
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)


p <- plot_accuracy_over_time(predictions)
print(p)

ggsave(
  "analysis/plots/sorted/global_transformer_accuracy_over_time.pdf",
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)

p <- plot_accuracy_over_time(predictions_naive)
print(p)

ggsave(
  "analysis/plots/sorted/global_naive_baseline_accuracy_over_time.pdf",
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)



correct <- horizon_iteration_performance(predictions)
correct_by_table <- horizon_iteration_performance_by_table(predictions)

correct_lstm <- horizon_iteration_performance(predictions_lstm)
correct_by_table_lstm <- horizon_iteration_performance_by_table(predictions_lstm)

correct_naive <- horizon_iteration_performance(predictions_naive)
correct_naive_by_table <- horizon_iteration_performance_by_table(predictions_naive)

# Offload predictions to free up memory
rm(predictions)
rm(predictions_lstm)
rm(predictions_naive)
gc()

export_csv(correct, "analysis/tables/sorted/global_transformer_performance.csv")
export_csv(correct_lstm, "analysis/tables/sorted/global_lstm_performance.csv")
export_csv(correct_naive, "analysis/tables/sorted/global_naive_baseline_performance.csv")
export_csv_by_table(correct_by_table, "analysis/tables/sorted/global_transformer_performance_by_table.csv")
export_csv_by_table(correct_by_table_lstm, "analysis/tables/sorted/global_lstm_performance_by_table.csv")
export_csv_by_table(correct_naive_by_table, "analysis/tables/sorted/global_naive_baseline_performance_by_table.csv")


# Box plot w/ correct and scatter plot w/ correct_naive
ggplot() +
  geom_boxplot(
    data = correct,
    aes(x = horizon, y = mean_percent_correct, color = "Global Transformer"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive,
    aes(x = horizon, y = mean_percent_correct, color = "Global Naive Baseline"),
    size = 2
  ) +
  labs(
    #    title = "Global, Transformer and Naive Baseline Performance: Horizon vs. Percent Correct",
    x = "Horizon",
    y = "Percent Correct",
    color = "Legend"
  ) +  # Ensures only one legend title
  scale_color_manual(values = c(
    "Global Transformer" = "black",
    "Global Naive Baseline" = "red"
  )) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_light()

ggsave(
  "analysis/plots/sorted/global_transformer_vs_naive_baseline.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)

ggplot() +
  geom_boxplot(
    data = correct,
    aes(x = horizon, y = mean_percent_correct, color = "Global Transformer"),
    alpha = 0.5
  ) +
  geom_boxplot(
    data = correct_lstm,
    aes(x = horizon, y = mean_percent_correct, color = "Global LSTM"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive,
    aes(x = horizon, y = mean_percent_correct, color = "Global Naive Baseline"),
    size = 2
  ) +
  labs(
    #    title = "Global, Transformer and Naive Baseline Performance: Horizon vs. Percent Correct",
    x = "Horizon",
    y = "Percent Correct",
    color = "Legend"
  ) +  # Ensures only one legend title
  scale_color_manual(values = c(
    "Global Transformer" = "black",
    "Global LSTM" = "blue",
    "Global Naive Baseline" = "red"
  )) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_light()

ggsave(
  "analysis/plots/sorted/global_transformer_lstm_vs_naive_baseline.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)

# Box plot w/ correct showing Global, Transformer: GT_table vs Percent Correct by Horizon

ggplot() +
  geom_boxplot(
    data = correct_by_table,
    aes(x = gt_table, y = mean_percent_correct, fill=horizon),
    alpha = 0.5
  ) +
  labs(
    #    title = "Global, Transformer Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    fill = "Horizon"
  ) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_light()

ggsave(
  "analysis/plots/sorted/global_transformer_by_table.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)


ggplot() +
  geom_boxplot(
    data = correct_by_table_lstm,
    aes(x = gt_table, y = mean_percent_correct, fill=horizon),
    alpha = 0.5
  ) +
  labs(
    #    title = "Global, LSTM Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    fill = "Horizon"
  ) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_light()

ggsave(
  "analysis/plots/sorted/global_lstm_by_table.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)


ggplot() +
  geom_boxplot(
    data = correct_by_table,
    aes(x = gt_table, y = mean_percent_correct, color = "Global Transformer"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive_by_table,
    aes(x = gt_table, y = mean_percent_correct, color = "Global Naive Baseline"),
    size = 2,
  ) +
  facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
  labs(
    #    title = "Global, Transformer vs Naive Baseline Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    color = "Model"
  ) +
  scale_color_manual(values = c("Global Transformer" = "black", "Global Naive Baseline" = "red")) + # Red for scatter plot
  scale_y_continuous(limits = c(0, 1)) +
  theme_light() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ggsave(
  "analysis/plots/sorted/global_transformer_vs_naive_baseline_by_table.pdf",
  width = 15,
  height = 6,
  units = "in",
  dpi = 300
)


ggplot() +
  geom_boxplot(
    data = correct_by_table,
    aes(x = gt_table, y = mean_percent_correct, color = "Global Transformer"),
    alpha = 0.5
  ) +
  geom_boxplot(
    data = correct_by_table_lstm,
    aes(x = gt_table, y = mean_percent_correct, color = "Global LSTM"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive_by_table,
    aes(x = gt_table, y = mean_percent_correct, color = "Global Naive Baseline"),
    size = 2,
  ) +
  facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
  labs(
    #    title = "Global, Transformer vs Naive Baseline Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    color = "Model"
  ) +
  scale_color_manual(values = c(
    "Global Transformer" = "black",
    "Global LSTM" = "blue",
    "Global Naive Baseline" = "red"
  )) + # Red for scatter plot
  scale_y_continuous(limits = c(0, 1)) +
  theme_light() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ggsave(
  "analysis/plots/sorted/global_transformer_lstm_vs_naive_baseline_by_table.pdf",
  width = 15,
  height = 6,
  units = "in",
  dpi = 300
)

# Lets load the data for local transformer models
predictions_local <- load_parquet("analysis/data/exp-19-transformer-sorted-row-locks/predictions.parquet")
check_iterations(predictions_local)

predictions_local_lstm <- load_parquet("analysis/data/exp-20-lstm-sorted-row-locks/predictions.parquet")
check_iterations(predictions_local_lstm)

correct_local <- horizon_iteration_performance(predictions_local)
correct_local_by_table <- horizon_iteration_performance_by_table(predictions_local)

correct_local_lstm <- horizon_iteration_performance(predictions_local_lstm)
correct_local_by_table_lstm <- horizon_iteration_performance_by_table(predictions_local_lstm)


correct_local_no_orderline_gt_h1 <- horizon_iteration_performance(
  predictions_local %>%
    filter(!(as.numeric(as.character(horizon))>1 & gt_table == "orderline"))
)
correct_local_no_orderline_gt_h1_by_table <- horizon_iteration_performance_by_table(
  predictions_local %>%
    filter(!(as.numeric(as.character(horizon))>1 & gt_table == "orderline"))
)

# Offload predictions to free up memory
rm(predictions_local)
rm(predictions_local_lstm)
gc()


export_csv(correct_local, "analysis/tables/sorted/local_transformer_performance.csv")
export_csv_by_table(correct_local_by_table, "analysis/tables/sorted/local_transformer_performance_by_table.csv")
export_csv(correct_local_lstm, "analysis/tables/sorted/local_lstm_performance.csv")
export_csv_by_table(correct_local_by_table_lstm, "analysis/tables/sorted/local_lstm_performance_by_table.csv")
export_csv(correct_local_no_orderline_gt_h1, "analysis/tables/sorted/local_transformer_no_orderline_gt_h1_performance.csv")
export_csv_by_table(correct_local_no_orderline_gt_h1_by_table, "analysis/tables/sorted/local_transformer_no_orderline_gt_h1_performance_by_table.csv")

ggplot() +
  geom_boxplot(
    data = correct_local_by_table,
    aes(x = gt_table, y = mean_percent_correct, fill=horizon),
    alpha = 0.5
  ) +
  labs(
    #    title = "Local Transformer Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    fill = "Horizon"
  ) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_light()

ggsave(
  "analysis/plots/sorted/local_transformer_by_table.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)

ggplot() +
  geom_boxplot(
    data = correct_local_by_table_lstm,
    aes(x = gt_table, y = mean_percent_correct, fill=horizon),
    alpha = 0.5
  ) +
  labs(
    #    title = "Local Transformer Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    fill = "Horizon"
  ) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_light()

ggsave(
  "analysis/plots/sorted/local_lstm_by_table.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)

# Lets combine the Global and Local Transformer plots so that each table shows the performance of both models

# Combine the data
correct_by_table$Model <- "Global Transformer"
correct_local_by_table$Model <- "Local Transformer"

combined_correct_by_table <- bind_rows(correct_by_table, correct_local_by_table)

# Plot with both Global and Local Transformer performances
ggplot(combined_correct_by_table, aes(x = gt_table, y = mean_percent_correct, fill = Model)) +
  geom_boxplot(alpha = 0.5) +
  facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
  labs(
    #    title = "Global vs Local Transformer Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    fill = "Model"
  ) +
  theme_light() +
  scale_y_continuous(limits = c(0, 1)) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ggsave(
  "analysis/plots/sorted/global_vs_local_transformer_by_table.pdf",
  width = 15,
  height = 6,
  units = "in",
  dpi = 300
)

correct_by_table_lstm$Model <- "Global LSTM"
correct_local_by_table_lstm$Model <- "Local LSTM"

combined_correct_by_table_lstm <- bind_rows(correct_by_table_lstm, correct_local_by_table_lstm)

# Plot with both Global and Local Transformer performances
ggplot(combined_correct_by_table_lstm, aes(x = gt_table, y = mean_percent_correct, fill = Model)) +
  geom_boxplot(alpha = 0.5) +
  facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
  labs(
    #    title = "Global vs Local Transformer Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    fill = "Model"
  ) +
  theme_light() +
  scale_y_continuous(limits = c(0, 1)) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# Lets plot local transformer vs global Naive baseline (i.e., correct_naive)
# The plots should be Horizon vs Percent Correct, where local transformer is a box plot and global naive is a scatter plot

ggplot() +
  geom_boxplot(
    data = correct_local,
    aes(x = horizon, y = mean_percent_correct, color = "Local Transformer"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive,
    aes(x = horizon, y = mean_percent_correct, color = "Global Naive Baseline"),
    size = 2
  ) +
  labs(
    #    title = "Local Transformer vs Global Naive Baseline Performance: Horizon vs. Percent Correct",
    x = "Horizon",
    y = "Percent Correct",
    color = "Legend"
  ) +  # Ensures only one legend title
  scale_color_manual(values = c(
    "Local Transformer" = "black",
    "Global Naive Baseline" = "red"
  )) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_light()

ggsave(
  "analysis/plots/sorted/local_transformer_vs_global_naive_baseline.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)

ggplot() +
  geom_boxplot(
    data = correct_local,
    aes(x = horizon, y = mean_percent_correct, color = "Local Transformer"),
    alpha = 0.5
  ) +
  geom_boxplot(
    data = correct_local_lstm,
    aes(x = horizon, y = mean_percent_correct, color = "Local LSTM"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive,
    aes(x = horizon, y = mean_percent_correct, color = "Global Naive Baseline"),
    size = 2
  ) +
  labs(
    #    title = "Local Transformer vs Global Naive Baseline Performance: Horizon vs. Percent Correct",
    x = "Horizon",
    y = "Percent Correct",
    color = "Legend"
  ) +  # Ensures only one legend title
  scale_color_manual(values = c(
    "Local Transformer" = "black",
    "Global Naive Baseline" = "red",
    "Local LSTM" = "blue"
  )) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_light()

ggsave(
  "analysis/plots/sorted/local_transformer_lstm_vs_global_naive_baseline.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)

predictions_naive_local <- load_parquet("analysis/data/exp-19-naive-sorted-row-locks/predictions.parquet")

correct_naive_local <- horizon_iteration_performance(predictions_naive_local)
correct_naive_local_by_table <- horizon_iteration_performance_by_table(predictions_naive_local)

rm(predictions_naive_local)
gc()

export_csv(correct_naive_local, "analysis/tables/sorted/local_naive_baseline_performance.csv")
export_csv_by_table(correct_naive_local_by_table, "analysis/tables/sorted/local_naive_baseline_performance_by_table.csv")

ggplot() +
  geom_boxplot(
    data = correct_local,
    aes(x = horizon, y = mean_percent_correct, color = "Local Transformer"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive_local,
    aes(x = horizon, y = mean_percent_correct, color = "Local Naive Baseline"),
    size = 2
  ) +
  labs(
    #    title = "Local Transformer vs Local Naive Baseline Performance: Horizon vs. Percent Correct",
    x = "Horizon",
    y = "Percent Correct",
    color = "Legend"
  ) +  # Ensures only one legend title
  scale_color_manual(values = c(
    "Local Transformer" = "black",
    "Local Naive Baseline" = "red"
  )) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_light()

ggsave(
  "analysis/plots/sorted/local_transformer_vs_local_naive_baseline.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)

ggplot() +
  geom_boxplot(
    data = correct_local,
    aes(x = horizon, y = mean_percent_correct, color = "Local Transformer"),
    alpha = 0.5
  ) +
  geom_boxplot(
    data = correct_local_lstm,
    aes(x = horizon, y = mean_percent_correct, color = "Local LSTM"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive_local,
    aes(x = horizon, y = mean_percent_correct, color = "Local Naive Baseline"),
    size = 2
  ) +
  labs(
    #    title = "Local Transformer vs Local Naive Baseline Performance: Horizon vs. Percent Correct",
    x = "Horizon",
    y = "Percent Correct",
    color = "Legend"
  ) +  # Ensures only one legend title
  scale_color_manual(values = c(
    "Local Transformer" = "black",
    "Local Naive Baseline" = "red",
    "Local LSTM" = "blue"
  )) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_light()

ggsave(
  "analysis/plots/sorted/local_transformer_lstm_vs_local_naive_baseline.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)

correct_naive_local_by_table$Model <- "Local Naive Baseline"

# Modify the plot: Local Transformer as a box plot, Local Naive Baseline as a scatter plot
ggplot() +
  geom_boxplot(
    data = correct_local_by_table,
    aes(x = gt_table, y = mean_percent_correct, color = "Local Transformer"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive_local_by_table,
    aes(x = gt_table, y = mean_percent_correct, color = "Local Naive Baseline"),
    size = 2,
  ) +
  facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
  labs(
    #    title = "Local Transformer vs Local Naive Baseline Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    color = "Model"
  ) +
  scale_color_manual(values = c("Local Transformer" = "black", "Local Naive Baseline" = "red")) + # Red for scatter plot
  theme_light() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

# Save the updated plot
ggsave(
  "analysis/plots/sorted/local_transformer_vs_local_naive_baseline_by_table.pdf",
  width = 15,
  height = 6,
  units = "in",
  dpi = 300
)

ggplot() +
  geom_boxplot(
    data = correct_local_by_table,
    aes(x = gt_table, y = mean_percent_correct, color = "Local Transformer"),
    alpha = 0.5
  ) +
  geom_boxplot(
    data = correct_local_by_table_lstm,
    aes(x = gt_table, y = mean_percent_correct, color = "Local LSTM"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive_local_by_table,
    aes(x = gt_table, y = mean_percent_correct, color = "Local Naive Baseline"),
    size = 2,
  ) +
  facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
  labs(
    #    title = "Local Transformer vs Local Naive Baseline Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    color = "Model"
  ) +
  scale_color_manual(values = c(
    "Local Transformer" = "black",
    "Local Naive Baseline" = "red",
    "Local LSTM" = "blue"
  )) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_light() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ggsave(
  "analysis/plots/sorted/local_transformer_lstm_vs_local_naive_baseline_by_table.pdf",
  width = 15,
  height = 6,
  units = "in",
  dpi = 300
)

predictions_local_rowid <- load_parquet("analysis/data/exp-25-transformer-row-id-local-sorted-row-locks/predictions.parquet")
check_iterations(predictions_local_rowid)

predictions_local_rowid_lstm <- load_parquet("analysis/data/exp-23-lstm-row-id-local-sorted-row-locks/predictions.parquet")
check_iterations(predictions_local_rowid_lstm)

correct_local_rowid <- horizon_iteration_performance(predictions_local_rowid)
correct_local_rowid_by_table <- horizon_iteration_performance_by_table(predictions_local_rowid)

correct_local_rowid_lstm <- horizon_iteration_performance(predictions_local_rowid_lstm)
correct_local_rowid_by_table_lstm <- horizon_iteration_performance_by_table(predictions_local_rowid_lstm)

rm(predictions_local_rowid)
rm(predictions_local_rowid_lstm)
gc()

export_csv(correct_local_rowid, "analysis/tables/sorted/local_transformer_rowid_performance.csv")
export_csv_by_table(correct_local_rowid_by_table, "analysis/tables/sorted/local_transformer_rowid_performance_by_table.csv")
export_csv(correct_local_rowid_lstm, "analysis/tables/sorted/local_lstm_rowid_performance.csv")
export_csv_by_table(correct_local_rowid_by_table_lstm, "analysis/tables/sorted/local_lstm_rowid_performance_by_table.csv")


ggplot() +
  geom_boxplot(
    data = correct_local,
    aes(x = horizon, y = mean_percent_correct, color = "Local Transformer"),
    alpha = 0.5
  ) +
  geom_boxplot(
    data = correct_local_rowid,
    aes(x = horizon, y = mean_percent_correct, color = "Local Transformer with Row ID"),
    alpha = 0.5
  ) +
  labs(
    #    title = "Local Transformer w/ & w/o Row ID Performance (Excludes 'orderline' table for Horizon>1): Horizon vs. Percent Correct",
    x = "Horizon",
    y = "Percent Correct",
    color = "Legend"
  ) +  # Ensures only one legend title
  scale_color_manual(values = c(
    "Local Transformer" = "black",
    "Local Transformer with Row ID" = "red"
  )) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_light()


ggsave(
  "analysis/plots/sorted/local_transformer_vs_local_transformer_rowid.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)

ggplot() +
  geom_boxplot(
    data = correct_local_lstm,
    aes(x = horizon, y = mean_percent_correct, color = "Local LSTM"),
    alpha = 0.5
  ) +
  geom_boxplot(
    data = correct_local_rowid_lstm,
    aes(x = horizon, y = mean_percent_correct, color = "Local LSTM with Row ID"),
    alpha = 0.5
  ) +
  labs(
    #    title = "Local LSTM w/ & w/o Row ID Performance (Excludes 'orderline' table for Horizon>1): Horizon vs. Percent Correct",
    x = "Horizon",
    y = "Percent Correct",
    color = "Legend"
  ) +  # Ensures only one legend title
  scale_color_manual(values = c(
    "Local LSTM" = "black",
    "Local LSTM with Row ID" = "red"
  )) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_light()

ggsave(
  "analysis/plots/sorted/local_lstm_vs_local_lstm_rowid.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)

correct_local_rowid_by_table$Model <- "Local Transformer with Row ID"
correct_local_by_table$Model <- "Local Transformer"

combined_correct_local_by_table <- bind_rows(
  correct_local_by_table, correct_local_rowid_by_table)

# Plot with both Global and Local Transformer performances
ggplot(combined_correct_local_by_table, aes(x = gt_table, y = mean_percent_correct, fill = Model)) +
  geom_boxplot(alpha = 0.5) +
  facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
  labs(
    #    title = "Local Transformer w/ & w/o Row ID Performance (Excludes 'orderline' table for Horizon>1): Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    fill = "Model"
  ) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_light() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ggsave(
  "analysis/plots/sorted/local_transformer_vs_local_transformer_rowid_by_table.pdf",
  width = 15,
  height = 6,
  units = "in",
  dpi = 300
)

correct_local_rowid_by_table_lstm$Model <- "Local LSTM with Row ID"
correct_local_by_table_lstm$Model <- "Local LSTM"

combined_correct_local_by_table_lstm <- bind_rows(
  correct_local_by_table_lstm, correct_local_rowid_by_table_lstm)

# Plot with both Global and Local Transformer performances
ggplot(combined_correct_local_by_table_lstm, aes(x = gt_table, y = mean_percent_correct, fill = Model)) +
  geom_boxplot(alpha = 0.5) +
  facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
  labs(
    #    title = "Local LSTM w/ & w/o Row ID Performance (Excludes 'orderline' table for Horizon>1): Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    fill = "Model"
  ) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_light() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ggsave(
  "analysis/plots/sorted/local_lstm_vs_local_lstm_rowid_by_table.pdf",
  width = 15,
  height = 6,
  units = "in",
  dpi = 300
)


#####################
#####################

# Lets look at table lock performance

predictions_table <- load_parquet("analysis/data/exp-17-transformer-sorted-table-locks/predictions.parquet", is_table_lock=TRUE)
check_iterations(predictions_table)

predictions_table_lstm <- load_parquet("analysis/data/exp-21-lstm-sorted-table-locks/predictions.parquet", is_table_lock=TRUE)
check_iterations(predictions_table_lstm)

predictions_naive_table <- load_parquet("analysis/data/exp-18-naive-sorted-table-locks/predictions.parquet", is_table_lock=TRUE) %>%
  filter(data == "data/fixed/table_locks.csv")

correct_table <- horizon_iteration_performance(predictions_table)
correct_table_by_table <- horizon_iteration_performance_by_table(predictions_table)

correct_table_lstm <- horizon_iteration_performance(predictions_table_lstm)
correct_table_by_table_lstm <- horizon_iteration_performance_by_table(predictions_table_lstm)

correct_naive_table <- horizon_iteration_performance(predictions_naive_table)
correct_naive_table_by_table <- horizon_iteration_performance_by_table(predictions_naive_table)


p <- plot_accuracy_over_time(predictions_table)
print(p)

ggsave(
  "analysis/plots/sorted/table-lock_global_transformer_accuracy_over_time.pdf",
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)

p <- plot_accuracy_over_time(predictions_naive_table)
print(p)

ggsave(
  "analysis/plots/sorted/table-lock_global_naive_baseline_accuracy_over_time.pdf",
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)

p <- plot_accuracy_over_time_list(
  list(predictions_table, predictions_table_lstm, predictions_naive_table),
  c("Global Transformer", "Global LSTM", "Global Naive Baseline")
)
print(p)

ggsave(
  "analysis/plots/sorted/table-lock_global_transformer_lstm_naive_baseline_accuracy_over_time.pdf",
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)


rm(predictions_table)
rm(predictions_table_lstm)
rm(predictions_naive_table)
gc()

export_csv(correct_table, "analysis/tables/sorted/table-lock_transformer_performance.csv")
export_csv(correct_naive_table, "analysis/tables/sorted/table-lock_naive_baseline_performance.csv")
export_csv(correct_table_lstm, "analysis/tables/sorted/table-lock_lstm_performance.csv")
export_csv_by_table(correct_table_by_table, "analysis/tables/sorted/table-lock_transformer_performance_by_table.csv")
export_csv_by_table(correct_naive_table_by_table, "analysis/tables/sorted/table-lock_naive_baseline_performance_by_table.csv")
export_csv_by_table(correct_table_by_table_lstm, "analysis/tables/sorted/table-lock_lstm_performance_by_table.csv")


# Box plot w/ correct and scatter plot w/ correct_naive
ggplot() +
  geom_boxplot(
    data = correct_table,
    aes(x = horizon, y = mean_percent_correct, color = "Transformer"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive_table,
    aes(x = horizon, y = mean_percent_correct, color = "Naive Baseline"),
    size = 2
  ) +
  labs(
    #    title = "Table Lock, Transformer and Naive Baseline Performance: Horizon vs. Percent Correct",
    x = "Horizon",
    y = "Percent Correct",
    color = "Legend"
  ) +  # Ensures only one legend title
  scale_color_manual(values = c(
    "Transformer" = "black",
    "Naive Baseline" = "red"
  )) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_light()

ggsave(
  "analysis/plots/sorted/table-lock_transformer_vs_naive_baseline.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)

ggplot() +
  geom_boxplot(
    data = correct_table,
    aes(x = horizon, y = mean_percent_correct, color = "Transformer"),
    alpha = 0.5
  ) +
  geom_boxplot(
    data = correct_table_lstm,
    aes(x = horizon, y = mean_percent_correct, color = "LSTM"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive_table,
    aes(x = horizon, y = mean_percent_correct, color = "Naive Baseline"),
    size = 2
  ) +
  labs(
    #    title = "Table Lock, Transformer and Naive Baseline Performance: Horizon vs. Percent Correct",
    x = "Horizon",
    y = "Percent Correct",
    color = "Legend"
  ) +  # Ensures only one legend title
  scale_color_manual(values = c(
    "Transformer" = "black",
    "Naive Baseline" = "red",
    "LSTM" = "blue"
  )) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_light()

ggsave(
  "analysis/plots/sorted/table-lock_transformer_lstm_vs_naive_baseline.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)


ggplot() +
  geom_boxplot(
    data = correct_table_by_table,
    aes(x = gt_table, y = mean_percent_correct, color = "Transformer"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive_table_by_table,
    aes(x = gt_table, y = mean_percent_correct, color = "Naive Baseline"),
    size = 2,
  ) +
  facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
  labs(
    #    title = "Table Lock, Transformer vs Naive Baseline Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    color = "Model"
  ) +
  scale_color_manual(values = c("Transformer" = "black", "Naive Baseline" = "red")) + # Red for scatter plot
  scale_y_continuous(limits = c(0, 1)) +
  theme_light() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ggsave(
  "analysis/plots/sorted/table-lock_transformer_vs_naive_baseline_by_table.pdf",
  width = 15,
  height = 6,
  units = "in",
  dpi = 300
)

ggplot() +
  geom_boxplot(
    data = correct_table_by_table,
    aes(x = gt_table, y = mean_percent_correct, color = "Transformer"),
    alpha = 0.5
  ) +
  geom_boxplot(
    data = correct_table_by_table_lstm,
    aes(x = gt_table, y = mean_percent_correct, color = "LSTM"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive_table_by_table,
    aes(x = gt_table, y = mean_percent_correct, color = "Naive Baseline"),
    size = 2,
  ) +
  facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
  labs(
    #    title = "Table Lock, Transformer vs Naive Baseline Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    color = "Model"
  ) +
  scale_color_manual(values = c("Transformer" = "black", "Naive Baseline" = "red", "LSTM" = "blue")) + # Red for scatter plot
  scale_y_continuous(limits = c(0, 1)) +
  theme_light() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ggsave(
  "analysis/plots/sorted/table-lock_transformer_lstm_vs_naive_baseline_by_table.pdf",
  width = 15,
  height = 6,
  units = "in",
  dpi = 300
)


p <- plot_precision_recall(correct_table_by_table)
print(p)

ggsave(
  "analysis/plots/sorted/table-lock_transformer_precision_recall.pdf",
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)

p <- plot_precision_recall(correct_table_by_table_lstm)
print(p)

ggsave(
  "analysis/plots/sorted/table-lock_lstm_precision_recall.pdf",
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)

p <- plot_precision_recall(correct_naive_table_by_table)
print(p)

ggsave(
  "analysis/plots/sorted/table-lock_naive_baseline_precision_recall.pdf",
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)


#####################

# Data set analysis:

# Load the csv data
data_row_locks <- read_csv("data/fixed/row_locks.csv") 
data_table_locks <- read_csv("data/fixed/table_locks.csv")

# Lets look at the data
summary(data_row_locks)
summary(data_table_locks)

# prep
data_row_locks <- data_row_locks %>%
  mutate(TABNAME = str_remove(TABNAME, "_")) %>%
  mutate(TABNAME = str_trim(TABNAME)) %>%
  mutate(TABSCHEMA = str_remove(TABSCHEMA, "_")) %>%
  mutate(TABSCHEMA = str_trim(TABSCHEMA))

data_table_locks <- data_table_locks %>%
  mutate(TABNAME = str_remove(TABNAME, "_")) %>%
  mutate(TABNAME = str_trim(TABNAME)) %>%
  mutate(TABSCHEMA = str_remove(TABSCHEMA, "_")) %>%
  mutate(TABSCHEMA = str_trim(TABSCHEMA))

# filter out tabschema sysibm
data_row_locks <- data_row_locks %>%
  filter(TABSCHEMA != "SYSIBM")

data_table_locks <- data_table_locks %>%
  filter(TABSCHEMA != "SYSIBM")


# Lets look at the unique TABNAME's and their counts
tab_counts_row_lock <- data_row_locks %>%
  group_by(TABNAME) %>%
  summarise(count = n()) %>%
  arrange(desc(count))
tab_counts_row_lock
write_csv(tab_counts_row_lock, "analysis/tables/sorted/row_lock_table_counts.csv")

tab_counts_table_lock <- data_table_locks %>%
  group_by(TABNAME) %>%
  summarise(count = n()) %>%
  arrange(desc(count))
tab_counts_table_lock

write_csv(tab_counts_table_lock, "analysis/tables/sorted/table_lock_table_counts.csv")

# lets loko at the unique pageids in the row locks for Warehouse table, i want the coutns
page_counts_warehouse_row_lock <- data_row_locks %>%
  group_by(TABNAME, PAGEID) %>%
  summarise(count = n()) %>%
  arrange(desc(count))
page_counts_warehouse_row_lock











#######
#######
#######
# Lets look at deduped global results:

predictions <- load_parquet("analysis/data/exp-26-tranformer-dedupe-row-locks/predictions.parquet")
check_iterations(predictions)

predictions_lstm <- load_parquet("analysis/data/exp-28-lstm-dedupe-row-locks/predictions.parquet")
check_iterations(predictions_lstm)

predictions_naive <- load_parquet("analysis/data/exp-27-naive-dedupe-row-locks/predictions.parquet") %>%
  filter(data == "data/fixed/row_locks.csv")


p <- plot_accuracy_over_time_list(
  list(predictions, predictions_lstm, predictions_naive),
  c("Global Transformer", "Global LSTM", "Global Naive Baseline")
)
print(p)

ggsave(
  "analysis/plots/deduped/global_transformer_lstm_naive_baseline_accuracy_over_time.pdf",
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)


p <- plot_accuracy_over_time(predictions)
print(p)

ggsave(
  "analysis/plots/deduped/global_transformer_accuracy_over_time.pdf",
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)

p <- plot_accuracy_over_time(predictions_naive)
print(p)

ggsave(
  "analysis/plots/deduped/global_naive_baseline_accuracy_over_time.pdf",
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)



correct <- horizon_iteration_performance(predictions)
correct_by_table <- horizon_iteration_performance_by_table(predictions)

correct_lstm <- horizon_iteration_performance(predictions_lstm)
correct_by_table_lstm <- horizon_iteration_performance_by_table(predictions_lstm)

correct_naive <- horizon_iteration_performance(predictions_naive)
correct_naive_by_table <- horizon_iteration_performance_by_table(predictions_naive)

# Offload predictions to free up memory
rm(predictions)
rm(predictions_lstm)
rm(predictions_naive)
gc()

export_csv(correct, "analysis/tables/deduped/global_transformer_performance.csv")
export_csv(correct_lstm, "analysis/tables/deduped/global_lstm_performance.csv")
export_csv(correct_naive, "analysis/tables/deduped/global_naive_baseline_performance.csv")
export_csv_by_table(correct_by_table, "analysis/tables/deduped/global_transformer_performance_by_table.csv")
export_csv_by_table(correct_by_table_lstm, "analysis/tables/deduped/global_lstm_performance_by_table.csv")
export_csv_by_table(correct_naive_by_table, "analysis/tables/deduped/global_naive_baseline_performance_by_table.csv")


# Box plot w/ correct and scatter plot w/ correct_naive
ggplot() +
  geom_boxplot(
    data = correct,
    aes(x = horizon, y = mean_percent_correct, color = "Global Transformer"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive,
    aes(x = horizon, y = mean_percent_correct, color = "Global Naive Baseline"),
    size = 2
  ) +
  labs(
    #    title = "Global, Transformer and Naive Baseline Performance: Horizon vs. Percent Correct",
    x = "Horizon",
    y = "Percent Correct",
    color = "Legend"
  ) +  # Ensures only one legend title
  scale_color_manual(values = c(
    "Global Transformer" = "black",
    "Global Naive Baseline" = "red"
  )) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_light()

ggsave(
  "analysis/plots/deduped/global_transformer_vs_naive_baseline.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)

ggplot() +
  geom_boxplot(
    data = correct,
    aes(x = horizon, y = mean_percent_correct, color = "Global Transformer"),
    alpha = 0.5
  ) +
  geom_boxplot(
    data = correct_lstm,
    aes(x = horizon, y = mean_percent_correct, color = "Global LSTM"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive,
    aes(x = horizon, y = mean_percent_correct, color = "Global Naive Baseline"),
    size = 2
  ) +
  labs(
    #    title = "Global, Transformer and Naive Baseline Performance: Horizon vs. Percent Correct",
    x = "Horizon",
    y = "Percent Correct",
    color = "Legend"
  ) +  # Ensures only one legend title
  scale_color_manual(values = c(
    "Global Transformer" = "black",
    "Global LSTM" = "blue",
    "Global Naive Baseline" = "red"
  )) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_light()

ggsave(
  "analysis/plots/deduped/global_transformer_lstm_vs_naive_baseline.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)

# Box plot w/ correct showing Global, Transformer: GT_table vs Percent Correct by Horizon

ggplot() +
  geom_boxplot(
    data = correct_by_table,
    aes(x = gt_table, y = mean_percent_correct, fill=horizon),
    alpha = 0.5
  ) +
  labs(
    #    title = "Global, Transformer Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    fill = "Horizon"
  ) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_light()

ggsave(
  "analysis/plots/deduped/global_transformer_by_table.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)


ggplot() +
  geom_boxplot(
    data = correct_by_table_lstm,
    aes(x = gt_table, y = mean_percent_correct, fill=horizon),
    alpha = 0.5
  ) +
  labs(
    #    title = "Global, LSTM Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    fill = "Horizon"
  ) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_light()

ggsave(
  "analysis/plots/deduped/global_lstm_by_table.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)


ggplot() +
  geom_boxplot(
    data = correct_by_table,
    aes(x = gt_table, y = mean_percent_correct, color = "Global Transformer"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive_by_table,
    aes(x = gt_table, y = mean_percent_correct, color = "Global Naive Baseline"),
    size = 2,
  ) +
  facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
  labs(
    #    title = "Global, Transformer vs Naive Baseline Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    color = "Model"
  ) +
  scale_color_manual(values = c("Global Transformer" = "black", "Global Naive Baseline" = "red")) + # Red for scatter plot
  scale_y_continuous(limits = c(0, 1)) +
  theme_light() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ggsave(
  "analysis/plots/deduped/global_transformer_vs_naive_baseline_by_table.pdf",
  width = 15,
  height = 6,
  units = "in",
  dpi = 300
)


ggplot() +
  geom_boxplot(
    data = correct_by_table,
    aes(x = gt_table, y = mean_percent_correct, color = "Global Transformer"),
    alpha = 0.5
  ) +
  geom_boxplot(
    data = correct_by_table_lstm,
    aes(x = gt_table, y = mean_percent_correct, color = "Global LSTM"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive_by_table,
    aes(x = gt_table, y = mean_percent_correct, color = "Global Naive Baseline"),
    size = 2,
  ) +
  facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
  labs(
    #    title = "Global, Transformer vs Naive Baseline Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    color = "Model"
  ) +
  scale_color_manual(values = c(
    "Global Transformer" = "black",
    "Global LSTM" = "blue",
    "Global Naive Baseline" = "red"
  )) + # Red for scatter plot
  scale_y_continuous(limits = c(0, 1)) +
  theme_light() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ggsave(
  "analysis/plots/deduped/global_transformer_lstm_vs_naive_baseline_by_table.pdf",
  width = 15,
  height = 6,
  units = "in",
  dpi = 300
)


# Lets look at deduped results for table locks

predictions_table <- load_parquet("analysis/data/exp-26-tranformer-dedupe-table-locks/predictions.parquet", is_table_lock=TRUE)
check_iterations(predictions_table)

predictions_table_lstm <- load_parquet("analysis/data/exp-28-lstm-dedupe-table-locks/predictions.parquet", is_table_lock=TRUE)
check_iterations(predictions_table_lstm)

predictions_naive_table <- load_parquet("analysis/data/exp-27-naive-dedupe-table-locks/predictions.parquet", is_table_lock=TRUE) %>%
  filter(data == "data/fixed/table_locks.csv")

correct_table <- horizon_iteration_performance(predictions_table)
correct_table_by_table <- horizon_iteration_performance_by_table(predictions_table)

correct_table_lstm <- horizon_iteration_performance(predictions_table_lstm)
correct_table_by_table_lstm <- horizon_iteration_performance_by_table(predictions_table_lstm)

correct_naive_table <- horizon_iteration_performance(predictions_naive_table)
correct_naive_table_by_table <- horizon_iteration_performance_by_table(predictions_naive_table)


p <- plot_accuracy_over_time(predictions_table)
print(p)

ggsave(
  "analysis/plots/deduped/table-lock_global_transformer_accuracy_over_time.pdf",
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)

p <- plot_accuracy_over_time(predictions_naive_table)
print(p)

ggsave(
  "analysis/plots/deduped/table-lock_global_naive_baseline_accuracy_over_time.pdf",
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)

p <- plot_accuracy_over_time_list(
  list(predictions_table, predictions_table_lstm, predictions_naive_table),
  c("Global Transformer", "Global LSTM", "Global Naive Baseline")
)
print(p)

ggsave(
  "analysis/plots/deduped/table-lock_global_transformer_lstm_naive_baseline_accuracy_over_time.pdf",
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)


rm(predictions_table)
rm(predictions_table_lstm)
rm(predictions_naive_table)
gc()

export_csv(correct_table, "analysis/tables/deduped/table-lock_transformer_performance.csv")
export_csv(correct_naive_table, "analysis/tables/deduped/table-lock_naive_baseline_performance.csv")
export_csv(correct_table_lstm, "analysis/tables/deduped/table-lock_lstm_performance.csv")
export_csv_by_table(correct_table_by_table, "analysis/tables/deduped/table-lock_transformer_performance_by_table.csv")
export_csv_by_table(correct_naive_table_by_table, "analysis/tables/deduped/table-lock_naive_baseline_performance_by_table.csv")
export_csv_by_table(correct_table_by_table_lstm, "analysis/tables/deduped/table-lock_lstm_performance_by_table.csv")


# Box plot w/ correct and scatter plot w/ correct_naive
ggplot() +
  geom_boxplot(
    data = correct_table,
    aes(x = horizon, y = mean_percent_correct, color = "Transformer"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive_table,
    aes(x = horizon, y = mean_percent_correct, color = "Naive Baseline"),
    size = 2
  ) +
  labs(
    #    title = "Table Lock, Transformer and Naive Baseline Performance: Horizon vs. Percent Correct",
    x = "Horizon",
    y = "Percent Correct",
    color = "Legend"
  ) +  # Ensures only one legend title
  scale_color_manual(values = c(
    "Transformer" = "black",
    "Naive Baseline" = "red"
  )) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_light()

ggsave(
  "analysis/plots/deduped/table-lock_transformer_vs_naive_baseline.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)

ggplot() +
  geom_boxplot(
    data = correct_table,
    aes(x = horizon, y = mean_percent_correct, color = "Transformer"),
    alpha = 0.5
  ) +
  geom_boxplot(
    data = correct_table_lstm,
    aes(x = horizon, y = mean_percent_correct, color = "LSTM"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive_table,
    aes(x = horizon, y = mean_percent_correct, color = "Naive Baseline"),
    size = 2
  ) +
  labs(
    #    title = "Table Lock, Transformer and Naive Baseline Performance: Horizon vs. Percent Correct",
    x = "Horizon",
    y = "Percent Correct",
    color = "Legend"
  ) +  # Ensures only one legend title
  scale_color_manual(values = c(
    "Transformer" = "black",
    "Naive Baseline" = "red",
    "LSTM" = "blue"
  )) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_light()

ggsave(
  "analysis/plots/deduped/table-lock_transformer_lstm_vs_naive_baseline.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)


ggplot() +
  geom_boxplot(
    data = correct_table_by_table,
    aes(x = gt_table, y = mean_percent_correct, color = "Transformer"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive_table_by_table,
    aes(x = gt_table, y = mean_percent_correct, color = "Naive Baseline"),
    size = 2,
  ) +
  facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
  labs(
    #    title = "Table Lock, Transformer vs Naive Baseline Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    color = "Model"
  ) +
  scale_color_manual(values = c("Transformer" = "black", "Naive Baseline" = "red")) + # Red for scatter plot
  scale_y_continuous(limits = c(0, 1)) +
  theme_light() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ggsave(
  "analysis/plots/deduped/table-lock_transformer_vs_naive_baseline_by_table.pdf",
  width = 15,
  height = 6,
  units = "in",
  dpi = 300
)

ggplot() +
  geom_boxplot(
    data = correct_table_by_table,
    aes(x = gt_table, y = mean_percent_correct, color = "Transformer"),
    alpha = 0.5
  ) +
  geom_boxplot(
    data = correct_table_by_table_lstm,
    aes(x = gt_table, y = mean_percent_correct, color = "LSTM"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive_table_by_table,
    aes(x = gt_table, y = mean_percent_correct, color = "Naive Baseline"),
    size = 2,
  ) +
  facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
  labs(
    #    title = "Table Lock, Transformer vs Naive Baseline Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    color = "Model"
  ) +
  scale_color_manual(values = c("Transformer" = "black", "Naive Baseline" = "red", "LSTM" = "blue")) + # Red for scatter plot
  scale_y_continuous(limits = c(0, 1)) +
  theme_light() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

ggsave(
  "analysis/plots/deduped/table-lock_transformer_lstm_vs_naive_baseline_by_table.pdf",
  width = 15,
  height = 6,
  units = "in",
  dpi = 300
)


p <- plot_precision_recall(correct_table_by_table)
print(p)

ggsave(
  "analysis/plots/deduped/table-lock_transformer_precision_recall.pdf",
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)

p <- plot_precision_recall(correct_table_by_table_lstm)
print(p)

ggsave(
  "analysis/plots/deduped/table-lock_lstm_precision_recall.pdf",
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)

p <- plot_precision_recall(correct_naive_table_by_table)
print(p)

ggsave(
  "analysis/plots/deduped/table-lock_naive_baseline_precision_recall.pdf",
  width = 10,
  height = 6,
  units = "in",
  dpi = 300
)


