setwd("C:/TMU/postdoc-TMU/lock-pred/")
source("analysis/common_functions.R") # Load common functions, vars, and libraries (Assumes running from project root)

RQ3_plot_accuracy_over_time_list <- function(dfs, df_names, num_bins = 40) {
  stopifnot(length(dfs) == length(df_names))
  all_data_list <- list()
  color_model1 ="#1b9e77"
  color_model2 ="#d95f02"
  color_model3 ="#7570b3"
  
  for (i in seq_along(dfs)) {
    predictions_cur <- dfs[[i]]
    dataset_name <- df_names[[i]]
    correct <- predictions_cur %>%
      filter(horizon == horizon_position) %>%
      group_by(horizon, iteration, unique_id, in_lock_start_time) %>%
      summarise(is_correct = all(cumulative_correct), .groups = 'drop')
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
    stat_summary_bin(fun = "mean", geom = "line", binwidth = binwidth, linewidth = 0.5) +
    facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
    labs(
      x = "Relative Lock End Time (nanoseconds)",
      y = "Accuracy",
      color = "Model"
    ) +
    scale_color_manual(values = setNames(c(color_model1, color_model2, color_model3), df_names)) +
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
      legend.box.margin = margin(b = -10),    
      axis.title = element_text(size = 13),
      axis.text = element_text(size = 11),
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5)
    ) +
    guides(color = guide_legend(override.aes = list(linewidth  = 6.5)))
  
  return(p)
}

 
RQ3_plot_lock_count_over_time_list <- function(dfs, df_names, num_bins = 15) {
  stopifnot(length(dfs) == length(df_names))
  
  all_data_list <- list()
  color_model1 <- "black"
  color_model2 <- "#d95f02"
  color_model3 <- "#7570b3"
  
  for (i in seq_along(dfs)) {
    predictions_cur <- dfs[[i]]
    dataset_name <- df_names[[i]]
    
    # Only horizon == 1
    filtered <- predictions_cur %>%
      filter(horizon == horizon_position  & horizon == 1)
    
    # Relative lock time
    filtered$in_lock_start_time_rel <- as.numeric(filtered$in_lock_start_time - min(filtered$in_lock_start_time))
    filtered$dataset_name <- dataset_name
    all_data_list[[i]] <- filtered
  }
  
  all_data <- dplyr::bind_rows(all_data_list)
  
  #print(now(all_data))
  print(head(all_data))
  
   
  all_data$dataset_name <- factor(all_data$dataset_name, levels = df_names)
  
  # Set binwidth for histogram
  x_range <- range(all_data$in_lock_start_time_rel, na.rm = TRUE)
  print(x_range)
  binwidth <- (x_range[2] - x_range[1]) / num_bins
  
  p <- ggplot(all_data, aes(x = in_lock_start_time_rel, fill = dataset_name)) +
    geom_histogram(binwidth = binwidth, position = "identity", alpha = 0.6, color = "black") +
   # facet_wrap(~ dataset_name, ncol = 3) +  # facet by dataset
    labs(
      x = "Relative Lock Start Time (nanoseconds)",
      y = "Number of Lock Observations",
      fill = "Model"
    ) +
    scale_fill_manual(values = setNames(c(color_model1, color_model2, color_model3), df_names)) +
    scale_x_continuous(labels = scales::scientific) +
    theme_light() +
    theme(
      #strip.text = element_text(size = 11, face = "bold"),
      legend.position = "none",
      legend.title = element_text(face = "bold"),
      legend.text = element_text(size = 11),
      axis.title = element_text(size = 13),
      axis.text = element_text(size = 11),
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5)
    ) +
    guides(fill = guide_legend(override.aes = list(alpha = 1)))
  
  return(p)
}

#######
#######
#######
# Lets look at "cut" global results:
message("--- Starting 'cut' Experiment Analysis ---")
experiment_subdir_cut <- "cut"

#basepath = "C:/TMU/postdoc-TMU/deep-rediscovery/deeptable-analysis/results/"


 

# Lets look at cut results for table locks
predictions_table <- load_parquet("analysis/data/exp-39-tranformer-rounded-cut-table-locks/predictions.parquet", is_table_lock=TRUE)
check_iterations(predictions_table)

nrow(predictions_table)


predictions_table_lstm <- load_parquet("analysis/data/exp-40-lstm-rounded-cut-table-locks/predictions.parquet", is_table_lock=TRUE)
check_iterations(predictions_table_lstm)

nrow(predictions_table_lstm)

predictions_naive_table <- load_parquet("analysis/data/exp-41-naive-rounded-cut-table-locks/predictions.parquet", is_table_lock=TRUE) %>%
  filter(data == "data/fixed/table_locks.csv")

nrow(predictions_naive_table)

#correct_table <- horizon_iteration_performance(predictions_table)
#correct_table_by_table <- horizon_iteration_performance_by_table(predictions_table)

#accuracy <- horizon_iteration_cumulative_accuracy(predictions_table)

#correct_table_by_table <- horizon_iteration_cumulative_performance_by_table(predictions_table)

#correct_table_lstm <- horizon_iteration_performance(predictions_table_lstm)
#correct_table_by_table_lstm <- horizon_iteration_performance_by_table(predictions_table_lstm)
#correct_table_by_table_lstm <- horizon_iteration_cumulative_performance_by_table(predictions_table_lstm)

#correct_naive_table <- horizon_iteration_performance(predictions_naive_table)
#correct_naive_table_by_table <- horizon_iteration_performance_by_table(predictions_naive_table)
#correct_naive_table_by_table <- horizon_iteration_cumulative_performance_by_table(predictions_naive_table)

 
predictions_cuml <- horizon_iteration_cumulative_calculation(predictions_table)
predictions_lstm_cuml <- horizon_iteration_cumulative_calculation(predictions_table_lstm)
predictions_naive_cuml <- horizon_iteration_cumulative_calculation(predictions_naive_table)
 

  

# Accuracy over time plots
p_global_acc_time <- RQ3_plot_accuracy_over_time_list(
  list(predictions_cuml, predictions_lstm_cuml, predictions_naive_cuml),
  c("Transformer", "LSTM", "Naive Baseline")
)




save_plot(p_global_acc_time, 
          construct_output_path("analysis/plots", experiment_subdir_cut, "RQ3_table_transformer_lstm_naive_baseline_accuracy_over_time_cuml.pdf"), 
          width = 10, height = 6)


p_global_locks_count_time <- RQ3_plot_lock_count_over_time_list(
  list(predictions_cuml),
  c("Transformer")
)



save_plot(p_global_locks_count_time, 
          construct_output_path("analysis/plots", experiment_subdir_cut, "RQ3_table_transformer_lstm_naive_baseline_locks_count_over_time_cuml.pdf"), 
          width = 6, height = 5)
 
















# Lets look at local for cut
predictions_local <- load_parquet("analysis/data/exp-44-transformer-local-rounded-cut/predictions.parquet")
check_iterations(predictions_local)
predictions_local_lstm <- load_parquet("analysis/data/exp-43-lstm-local-rounded-cut/predictions.parquet")
check_iterations(predictions_local_lstm)
predictions_naive_local <- load_parquet("analysis/data/exp-42-naive-local-rounded-cut-row-locks/predictions.parquet")

nrow(predictions_local)
nrow(predictions_local_lstm)
nrow(predictions_naive_local)

predictions_local_cuml <- horizon_iteration_cumulative_calculation(predictions_local)
predictions_local_lstm_cuml <- horizon_iteration_cumulative_calculation(predictions_local_lstm)
predictions_local_naive_cuml <- horizon_iteration_cumulative_calculation(predictions_naive_local)




# Accuracy over time for local models
p_local_acc_time <- RQ3_plot_accuracy_over_time_list(
  list(predictions_local_cuml, predictions_local_lstm_cuml, predictions_local_naive_cuml),
  c("Local Transformer", "Local LSTM", "Local Naive Baseline")
)

save_plot(p_local_acc_time, 
          construct_output_path("analysis/plots", experiment_subdir_cut, "RQ3_local_transformer_lstm_naive_baseline_accuracy_over_time_cuml.pdf"), 
          width = 10, height = 6)

 

subset_predictions_local_cuml <- predictions_local_cuml %>% filter(horizon == 1, iteration == 1)
nrow(subset_predictions_local_cuml)


subset_predictions_local_lstm_cuml <- predictions_local_lstm_cuml %>% filter(horizon == 1, iteration == 1)
nrow(subset_predictions_local_lstm_cuml)


subset_predictions_local_naive_cuml <- predictions_local_naive_cuml %>% filter(horizon == 1, iteration == 1)
nrow(subset_predictions_local_naive_cuml)


p_global_local_locks_count_time <- RQ3_plot_lock_count_over_time_list(
  list(subset_predictions_local_cuml),  c("Local Transformer") )



save_plot(p_global_local_locks_count_time, 
          construct_output_path("analysis/plots", experiment_subdir_cut, "RQ3_table_transformer_lstm_naive_baseline_local_locks_count_over_time_cuml.pdf"), 
          width = 6, height = 5)










# Lets look at global for cut
predictions_global <- load_parquet("analysis/data/exp-39-tranformer-rounded-cut-row-locks/predictions.parquet")
check_iterations(predictions_global)
predictions_global_lstm <- load_parquet("analysis/data/exp-40-lstm-rounded-cut-row-locks/predictions.parquet")
check_iterations(predictions_global_lstm)
predictions_naive_global <- load_parquet("analysis/data/exp-41-naive-rounded-cut-row-locks/predictions.parquet")



predictions_global_cuml <- horizon_iteration_cumulative_calculation(predictions_global)
predictions_global_lstm_cuml <- horizon_iteration_cumulative_calculation(predictions_global_lstm)
predictions_global_naive_cuml <- horizon_iteration_cumulative_calculation(predictions_naive_global)




# Accuracy over time for local models
p_global_acc_time <- RQ3_plot_accuracy_over_time_list(
  list(predictions_global_cuml, predictions_global_lstm_cuml, predictions_global_naive_cuml),
  c("Global Transformer", "Global LSTM", "Global Naive Baseline")
)

save_plot(p_global_acc_time, 
          construct_output_path("analysis/plots", experiment_subdir_cut, "RQ3_global_transformer_lstm_naive_baseline_accuracy_over_time_cuml.pdf"), 
          width = 10, height = 6)



p_global_global_locks_count_time <- RQ3_plot_lock_count_over_time_list(
  list(predictions_global_cuml, predictions_global_lstm_cuml, predictions_global_naive_cuml),
  c("Global Transformer", "Global LSTM", "Global Naive Baseline")
)



save_plot(p_global_global_locks_count_time, 
          construct_output_path("analysis/plots", experiment_subdir_cut, "RQ3_table_transformer_lstm_naive_baseline_global_locks_count_over_time_cuml.pdf"), 
          width = 10, height = 6)



nrow(predictions_table)

subset_table <- predictions_table %>% filter(horizon == 1, iteration == 1)
nrow(subset_table)

nrow(predictions_local)
subset_local <- predictions_local %>% filter(horizon == 1, iteration == 1)
nrow(subset_local)

nrow(predictions_global)
subset_global <- predictions_global %>% filter(horizon == 1, iteration == 1)
nrow(subset_global)
