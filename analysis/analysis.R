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
  }
  correct[is.na(correct)] <- FALSE  # Explicitly set NA results to FALSE
  # NA's occur when the prediction output token length is less than the ground truth token length
  return(correct)
}

load_parquet <- function(path, is_table_lock=FALSE) {
  predictions <- read_parquet(path) %>% filter(iteration <= 10)
  predictions$is_correct <- is_correct(predictions, is_table_lock)
  predictions$horizon <- as.factor(predictions$horizon)
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
  
  return(correct_by_table)
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



correct <- horizon_iteration_performance(predictions)
correct_by_table <- horizon_iteration_performance_by_table(predictions)

correct_naive <- horizon_iteration_performance(predictions_naive)

# Offload predictions to free up memory
rm(predictions)
rm(predictions_naive)
gc()

# Box plot w/ correct and scatter plot w/ correct_naive
ggplot() +
  geom_boxplot(
    data = correct,
    aes(x = horizon, y = mean_percent_correct, color = "Model Predictions"),
    alpha = 0.5
  ) +
  geom_point(
    data = correct_naive,
    aes(x = horizon, y = mean_percent_correct, color = "Naive Baseline"),
    size = 2
  ) +
  labs(
    title = "Global, Transformer and Naive Baseline Performance: Horizon vs. Percent Correct",
    x = "Horizon",
    y = "Percent Correct",
    color = "Legend"
  ) +  # Ensures only one legend title
  scale_color_manual(values = c(
    "Model Predictions" = "black",
    "Naive Baseline" = "red"
  )) +
  theme_minimal()

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
    title = "Global, Transformer Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    fill = "Horizon"
  ) +
  theme_minimal()

ggsave(
  "analysis/plots/global_transformer_by_table.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)



# Lets load the data for local transformer models
predictions_local <- load_parquet("analysis/data/row_sep/exp-11-row-locks/predictions.parquet")
check_iterations(predictions_local)
correct_local <- horizon_iteration_performance(predictions_local)
correct_local_by_table <- horizon_iteration_performance_by_table(predictions_local)


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
gc()

ggplot() +
  geom_boxplot(
    data = correct_local_by_table,
    aes(x = gt_table, y = mean_percent_correct, fill=horizon),
    alpha = 0.5
  ) +
  labs(
    title = "Local Transformer Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    fill = "Horizon"
  ) +
  theme_minimal()

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

horizon_labels <- c(
  "1" = "Horizon: 1",
  "2" = "Horizon: 2",
  "3" = "Horizon: 3",
  "4" = "Horizon: 4"
)

# Plot with both Global and Local Transformer performances
ggplot(combined_correct_by_table, aes(x = gt_table, y = mean_percent_correct, fill = Model)) +
  geom_boxplot(alpha = 0.5) +
  facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
  labs(
    title = "Global vs Local Transformer Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    fill = "Model"
  ) +
  theme_minimal()

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
    title = "Local Transformer vs Global Naive Baseline Performance: Horizon vs. Percent Correct",
    x = "Horizon",
    y = "Percent Correct",
    color = "Legend"
  ) +  # Ensures only one legend title
  scale_color_manual(values = c(
    "Local Transformer" = "black",
    "Global Naive Baseline" = "red"
  )) +
  theme_minimal()

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
    title = "Local Transformer vs Local Naive Baseline Performance: Horizon vs. Percent Correct",
    x = "Horizon",
    y = "Percent Correct",
    color = "Legend"
  ) +  # Ensures only one legend title
  scale_color_manual(values = c(
    "Local Transformer" = "black",
    "Local Naive Baseline" = "red"
  )) +
  theme_minimal()

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
    title = "Local Transformer vs Local Naive Baseline Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    color = "Model"
  ) +
  scale_color_manual(values = c("Local Transformer" = "black", "Local Naive Baseline" = "red")) + # Red for scatter plot
  theme_minimal()

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

ggplot() +
  geom_boxplot(
    data = correct_local_no_orderline_gt_h1,
    aes(x = horizon, y = mean_percent_correct, color = "Local Transformer"),
    alpha = 0.5
  ) +
  geom_boxplot(
    data = correct_local_rowid,
    aes(x = horizon, y = mean_percent_correct, color = "Local Transformer with Row ID"),
    alpha = 0.5
  ) +
  labs(
    title = "Local Transformer w/ & w/o Row ID Performance (Excludes 'orderline' table for Horizon>1): Horizon vs. Percent Correct",
    x = "Horizon",
    y = "Percent Correct",
    color = "Legend"
  ) +  # Ensures only one legend title
  scale_color_manual(values = c(
    "Local Transformer" = "black",
    "Local Transformer with Row ID" = "red"
  )) +
  theme_minimal()


ggsave(
  "analysis/plots/local_transformer_vs_local_transformer_rowid.pdf",
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)


correct_local_rowid_by_table$Model <- "Local Transformer with Row ID"
correct_local_no_orderline_gt_h1_by_table$Model <- "Local Transformer"

combined_correct_local_no_orderline_gt_h1_by_table <- bind_rows(
  correct_local_no_orderline_gt_h1_by_table, correct_local_rowid_by_table)

# Plot with both Global and Local Transformer performances
ggplot(combined_correct_local_no_orderline_gt_h1_by_table, aes(x = gt_table, y = mean_percent_correct, fill = Model)) +
  geom_boxplot(alpha = 0.5) +
  facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
  labs(
    title = "Local Transformer w/ & w/o Row ID Performance (Excludes 'orderline' table for Horizon>1): Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    fill = "Model"
  ) +
  theme_minimal()

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
    title = "Table Lock, Transformer and Naive Baseline Performance: Horizon vs. Percent Correct",
    x = "Horizon",
    y = "Percent Correct",
    color = "Legend"
  ) +  # Ensures only one legend title
  scale_color_manual(values = c(
    "Model Predictions" = "black",
    "Naive Baseline" = "red"
  )) +
  theme_minimal()

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
    title = "Transformer vs Naive Baseline Performance: Table vs. Percent Correct by Horizon",
    x = "Table",
    y = "Percent Correct",
    color = "Model"
  ) +
  scale_color_manual(values = c("Transformer" = "black", "Naive Baseline" = "red")) + # Red for scatter plot
  theme_minimal()












