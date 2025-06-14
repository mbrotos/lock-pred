require(tidyverse)
require(arrow)  # to read parquet
library(dplyr)
library(ggplot2)

source("analysis/common_functions.R")

# --- Test Dataframe with Realistic Ground Truth (gt) and Prediction (pred) data ---

test_predictions_detailed <- tribble(
  ~in_lock_start_time, ~unique_id, ~horizon, ~horizon_position, ~iteration, ~gt_table, ~pred_table, ~gt_pageid, ~pred_pageid,
  
  as.integer64("1734442181000994000"), "id_A", 1, 1, 1, "customer", "customer", 101, 101,
  as.integer64("1734442181000994000"), "id_B", 2, 1, 1, "district", "district", 202, 202,
  as.integer64("1734442181000994000"), "id_B", 2, 2, 1, "history", "orders",   303, 303,
  as.integer64("1734442181000994000"), "id_C", 3, 1, 1, "item", "item",     404, 404,
  as.integer64("1734442181000994000"), "id_C", 3, 2, 1, "new_order", "old_order", 505, 505,
  as.integer64("1734442181000994000"), "id_C", 3, 3, 1, "item", "item",     606, 606,
  
  as.integer64("1734442191000032000"), "id_D", 1, 1, 1, "customer", "customer", 111, 111,
  as.integer64("1734442191000032000"), "id_E", 2, 1, 1, "district", "district", 222, 222,
  as.integer64("1734442191000032000"), "id_E", 2, 2, 1, "history", "history",  333, 333,
  as.integer64("1734442191000032000"), "id_F", 3, 1, 1, "item", "item",     444, 444,
  as.integer64("1734442191000032000"), "id_F", 3, 2, 1, "new_order", "new_order", 555, 555,
  as.integer64("1734442191000032000"), "id_F", 3, 3, 1, "stock", "stock",    666, 666,

  as.integer64("1734442181000994000"), "id_G", 3, 1, 1, "item", "item", 404, 404,
  as.integer64("1734442181000994000"), "id_G", 3, 2, 1, "new_order", "old_order", 505, 505,
  as.integer64("1734442181000994000"), "id_G", 3, 3, 1, "item", "item", 606, 606,
  
  # --- Horizon 3: An INCOMPLETE but correct-so-far sequence ---
  # This sequence is missing its final step (horizon_position = 3).
  as.integer64("1734442201000032000"), "id_H", 3, 1, 1, "customer", "customer", 707, 707,
  as.integer64("1734442201000032000"), "id_H", 3, 2, 1, "district", "district", 808, 808,
  
  # More horizon 1
  as.integer64("1734442191000032000"), "id_I", 1, 1, 1, "stock", "customer", 111, 111,
  as.integer64("1734442191000032000"), "id_J", 1, 1, 1, "stock", "stock", 111, 111,
  as.integer64("1734442191000032000"), "id_K", 1, 1, 1, "customer", "item", 111, 111,
) %>%
  mutate(horizon = as.factor(horizon))



predictions_with_correctness <- test_predictions_detailed %>%
  mutate(is_correct = is_correct(.))


print("Dataframe after applying is_correct():")
print(predictions_with_correctness %>% arrange(horizon, iteration, unique_id, horizon_position))


print ("Testing cumall() and all() functions:")

test <- cumall(c(TRUE, TRUE, FALSE, TRUE)) 
test 
all(c(TRUE, TRUE, FALSE, TRUE))
print("cumall() and all() seem the same if u grab the last value of cumall()")

# 2. Now, run both analysis functions on the prepared data.
print("cumall() Results:")
cumulative_results <- horizon_iteration_cumulative_accuracy(predictions_with_correctness)

print("all() Results:")
performance_results <- horizon_iteration_performance(predictions_with_correctness)

# --- Check for Incomplete Sequences (Corrected) ---

incomplete_check <- predictions_with_correctness %>%
  group_by(unique_id) %>%
  summarise(
    # Convert the factor to a numeric value for comparison
    target_horizon = as.numeric(as.character(first(horizon))),
    max_position = max(horizon_position),
    .groups = 'drop'
  ) %>%
  filter(max_position != target_horizon)

if (nrow(incomplete_check) > 0) {
  cat("\n--- WARNING: Incomplete sequences found! ---\n")
  print(incomplete_check)
} else {
  cat("\n--- CHECK PASSED: No incomplete sequences found. ---\n")
}

# Incomplete sequence check from RQ1 on the "predictions_table" dataframe:
# --- WARNING: Incomplete sequences found! ---
#   # A tibble: 312,190 × 3
#   unique_id                            target_horizon max_position
# <chr>                                         <dbl>        <int>
#   1 0000c0ce-9101-4e1e-badf-5bd1bd19f9c5              4            3
# 2 0000c370-abce-45be-a2a8-656d5b1a2e5f              2            1
# 3 00012551-1bed-49fc-bf17-17b91b519c29              4            2
# 4 00015e25-a5e9-41ba-92d2-e865afd31cc9              3            2
# 5 00019851-49bb-4635-9a18-d837f83de8d2              4            3
# 6 00019856-77d1-49d3-b886-d8e890014bb0              3            2
# 7 0001e4d4-083d-442b-8f40-c3eda3d46643              4            3
# 8 0002021f-1e7b-4191-badd-696d1e29938f              3            2
# 9 00025130-a1bc-418c-a03f-96804692bcef              4            3
# 10 000251f9-da7c-4044-8b05-087db8b4b57d              3            2
# # ℹ 312,180 more rows

# Total rows: 7,957,220

# [1] "Breakdown of incomplete sequences by target horizon:"
# # A tibble: 3 × 2
# target_horizon number_of_incomplete_sequences
# <dbl>                          <int>
#   1              2                         102790
#   2              3                         104700
#   3               4                         104700

print("Looks like all() overestimates the performance when we have incomplete sequences. Otherwise its the same as cumall().")

print("RQ1: Precision and Recall:")

correct_table_by_table <- horizon_iteration_cumulative_performance_by_table(predictions_with_correctness)
print("Correctness by Table:")
print(correct_table_by_table)

print(predictions_with_correctness %>% arrange(horizon, iteration, unique_id, horizon_position))

print("Looks like there may be some descrepancies between the counts and what I expect for the candidate apporach.")

print("RQ2: Looks good w/ perhaps no incomplete sequences for local.")

print("RQ3: Looks good w/ perhaps no incomplete sequences for local.")

# > predictions_local <- load_parquet("analysis/data/exp-44-transformer-local-rounded-cut/predictions.parquet")
# > check_iterations(predictions_local)
# [1] "All horizons have at least 10 iterations"
# > incomplete_check <- predictions_local %>%
#   +     group_by(unique_id) %>%
#   +     summarise(
#     +         # Convert the factor to a numeric value for comparison
#       +         target_horizon = as.numeric(as.character(first(horizon))),
#     +         max_position = max(horizon_position),
#     +         .groups = 'drop'
#     +     ) %>%
#   +     filter(max_position < target_horizon)
# > 
#   > if (nrow(incomplete_check) > 0) {
#     +     cat("\n--- WARNING: Incomplete sequences found! ---\n")
#     +     print(incomplete_check)
#     + } else {
#       +     cat("\n--- CHECK PASSED: No incomplete sequences found. ---\n")
#       + }
# 
# --- CHECK PASSED: No incomplete sequences found. ---
# 


