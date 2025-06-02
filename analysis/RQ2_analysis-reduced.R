source("analysis/common_setup.R") # Load common functions, vars, and libraries (Assumes running from project root)

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

 