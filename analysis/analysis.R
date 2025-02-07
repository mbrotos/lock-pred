require(tidyverse)
require(arrow)  # to read parquet
library(dplyr)
library(ggplot2)

is_correct <- function(df) {
  correct <- (df$gt_table == df$pred_table) &
    (df$gt_pageid == df$pred_pageid)
  correct[is.na(correct)] <- FALSE  # Explicitly set NA results to FALSE
  return(correct)
}


predictions <- read_parquet("analysis/data/exp-6-row-locks/predictions.parquet")
predictions$is_correct <- is_correct(predictions)

predictions_naive <- read_parquet("analysis/data/exp-10/predictions.parquet") %>%
  filter(data == "data/fixed/row_locks.csv")
predictions_naive$is_correct <- is_correct(predictions_naive)


# Convert horizon to a factor
predictions$horizon <- as.factor(predictions$horizon)
predictions_naive$horizon <- as.factor(predictions_naive$horizon)


correct <- predictions %>%
  group_by(horizon, iteration, unique_id) %>%
  summarise(is_correct = all(is_correct)) %>%
  group_by(horizon, iteration) %>%
  summarise(mean_percent_correct = mean(is_correct))

correct %>%
  group_by(horizon) %>%
  summarise(percent_correct = mean(mean_percent_correct))

correct_naive <- predictions_naive %>%
  group_by(horizon, iteration, unique_id) %>%
  summarise(is_correct = all(is_correct)) %>%
  group_by(horizon, iteration) %>%
  summarise(mean_percent_correct = mean(is_correct))

correct_naive %>%
  group_by(horizon) %>%
  summarise(percent_correct = mean(mean_percent_correct))

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

# Bpx plot w/ correct showing Global, Transformer: GT_table vs Percent Correct by Horizon

correct_by_table <- predictions %>%
  group_by(horizon, iteration, unique_id, gt_table) %>%
  summarise(is_correct = all(is_correct)) %>%
  group_by(horizon, iteration, gt_table) %>%
  summarise(mean_percent_correct = mean(is_correct))

correct_by_table %>%
  group_by(horizon, gt_table) %>%
  summarise(percent_correct = mean(mean_percent_correct))

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


