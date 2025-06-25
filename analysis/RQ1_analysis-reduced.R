require(tidyverse)
require(arrow)  # to read parquet
library(dplyr)
library(ggplot2)


setwd("C:/TMU/postdoc-TMU/lock-pred/")
source("analysis/common_functions.R")

rq1_plot_table_performance_faceted_3_models <- function(data_model1_by_table, name_model1, data_model2_by_table, name_model2, data_model3_by_table, name_model3, 
                                                    color_model1 = "#1b9e77", color_model2 = "#d95f02", color_model3 = "#7570b3", 
                                                    file_path, plot_title = NULL, base_width = 15, base_height = 6) {
  
  
  
  
  #levels_models <- c("Transformer", "LSTM", "Naive-Baseline")
  # Add 'Model' column to each dataframe
  name_model1 <- "Transformer"
  name_model2 <- "LSTM"
  name_model3 <- "Naive-Baseline"  # ✅ Use hyphen here consistently
  data_model1_by_table$Model <-  name_model1 
  data_model2_by_table$Model <-  name_model2
  data_model3_by_table$Model <-  name_model3
  
  # Add 'type' column to differentiate between boxplot and point
  data_model1_by_table$plot_type <- "box"
  data_model2_by_table$plot_type <- "box"
  data_model3_by_table$plot_type <- "box"
  
 
  # Combine all into one dataframe
  data_all <- rbind(data_model1_by_table, data_model2_by_table, data_model3_by_table)
  levels_models <- c("Transformer", "LSTM", "Naive-Baseline")  # ✅ Match exactly
  data_all$Model <- factor(data_all$Model, levels = levels_models)
  
  
  print(levels(data_all$Model))
  
  # Base ggplot
  p <- ggplot(data_all, aes(x = gt_table, y = mean_percent_correct, color = Model)) +
    geom_boxplot(data = subset(data_all, plot_type == "box"), alpha = 0.5) +
    #geom_point(data = subset(data_all, plot_type == "point"), size = 2) +
    facet_wrap(~ horizon, labeller = as_labeller(horizon_labels)) +
    labs(
      title = plot_title,
      x = "Table",
      y = "Percent Correct",
      color = "Model"
    ) +
    scale_color_manual(values = setNames(c("#1b9e77", "#d95f02", "#7570b3"), levels_models)) + 
    scale_y_continuous(limits = c(0, 1)) +
    theme_light() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
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


rq1_plot_horizon_performance_3_models <- function(data_model1, name_model1, data_model2, name_model2, data_model3, name_model3, 
                                              color_model1 = "#1b9e77", color_model2 = "#d95f02", color_model3 = "#7570b3", 
                                              file_path, plot_title = NULL, base_width = 5, base_height = 5, y_col = "accuracy", y_label = "Accuracy"  ) {
  
  
  data_model1$Model <-  name_model1 
  data_model2$Model <-  name_model2
  data_model3$Model <-  name_model3
  data_model1$plot_type <- "box"
  data_model2$plot_type <- "box"
  data_model3$plot_type <- "box"
  
  levels_models <- c(name_model1, name_model2, name_model3)  # ✅ Match exactly
  data_all <- rbind(data_model1, data_model2, data_model3)
  data_all$Model <- factor(data_all$Model, levels = levels_models)
  
  
  p <- ggplot(data_all, aes(x = horizon, y = .data[[y_col]], color = Model)) +
    geom_boxplot(data = subset(data_all, plot_type == "box"), alpha = 0.5) +
    labs(
      title = plot_title,
      x = "Horizon",
      y = y_label,
      color = "Model"
    ) +
    scale_color_manual(values = setNames(c(color_model1, color_model2, color_model3), levels_models)) + 
    scale_y_continuous(limits = c(0, 1)) +
    theme_light() +
    theme(
      legend.position = "top",
      legend.title = element_text(face = "bold"),
      legend.text = element_text(size = 11),
      legend.box.margin = margin(b = -10),    
      axis.title = element_text(size = 13),
      axis.text = element_text(size = 11),
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5)
    ) +
    guides(color = guide_legend(override.aes = list(linewidth  = 2.5)))
  
   # guides( fill = guide_legend(override.aes = list(shape = 22, size = 6)), shape="none")

  
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




# Lets look at cut results for table locks
predictions_table <- load_parquet("analysis/data/exp-39-tranformer-rounded-cut-table-locks/predictions.parquet", is_table_lock=TRUE)
check_iterations(predictions_table)
predictions_table_lstm <- load_parquet("analysis/data/exp-40-lstm-rounded-cut-table-locks/predictions.parquet", is_table_lock=TRUE)
check_iterations(predictions_table_lstm)
predictions_naive_table <- load_parquet("analysis/data/exp-41-naive-rounded-cut-table-locks/predictions.parquet", is_table_lock=TRUE) %>%
  filter(data == "data/fixed/table_locks.csv")

 

 

# Tabel level lock prediction
# Calculating Accuracy of the model overall across 10 or n iteration, and n horizions
correct_table <- horizon_iteration_performance(predictions_table)
correct_table_by_table <- horizon_iteration_performance_by_table(predictions_table)
accuracy_overall_table <- horizon_overall_accuracy(predictions_table)
# Candidate Approach
accuracy_cum_table <- horizon_overall_cumulative_accuracy(predictions_table)
accuracy_iter_cum_table <- horizon_iteration_cumulative_accuracy(predictions_table)

 
correct_table_lstm <- horizon_iteration_performance(predictions_table_lstm)
correct_table_by_table_lstm <- horizon_iteration_performance_by_table(predictions_table_lstm)
accuracy_overall_table_lstm <- horizon_overall_accuracy(predictions_table_lstm)
# Candidate Approach
accuracy_cum_table_lstm <- horizon_overall_cumulative_accuracy(predictions_table_lstm)
accuracy_iter_cum_table_lstm <- horizon_iteration_cumulative_accuracy(predictions_table_lstm)


correct_naive_table <- horizon_iteration_performance(predictions_naive_table)
correct_naive_table_by_table <- horizon_iteration_performance_by_table(predictions_naive_table)
accuracy_overall_table_naive <- horizon_overall_accuracy(predictions_naive_table)
# Candidate Approach
accuracy_cum_table_naive <- horizon_overall_cumulative_accuracy(predictions_naive_table)
accuracy_iter_cum_table_naive <- horizon_iteration_cumulative_accuracy(predictions_naive_table)



summary_by_horizon <- correct_table_by_table %>%
  group_by(horizon) %>%
  summarise(
    mean_accuracy = mean(mean_percent_correct, na.rm = TRUE),
    mean_precision = mean(precision, na.rm = TRUE),
    mean_recall = mean(recall, na.rm = TRUE),
    mean_f1 = mean(f1, na.rm = TRUE)
  )



summary_by_horizon_lstm <- correct_table_by_table_lstm %>%
  group_by(horizon) %>%
  summarise(
    mean_accuracy = mean(mean_percent_correct, na.rm = TRUE),
    mean_precision = mean(precision, na.rm = TRUE),
    mean_recall = mean(recall, na.rm = TRUE),
    mean_f1 = mean(f1, na.rm = TRUE)
  )

summary_by_horizon_naive <- correct_naive_table_by_table %>%
  group_by(horizon) %>%
  summarise(
    mean_accuracy = mean(mean_percent_correct, na.rm = TRUE),
    mean_precision = mean(precision, na.rm = TRUE),
    mean_recall = mean(recall, na.rm = TRUE),
    mean_f1 = mean(f1, na.rm = TRUE)
  )


summary_by_horizon$Model <- "Transformer"
summary_by_horizon_lstm$Model <- "LSTM"
summary_by_horizon_naive$Model <- "Naive"

combined_summary <- bind_rows(
  summary_by_horizon,
  summary_by_horizon_lstm,
  summary_by_horizon_naive
)


horizon_1_data <- combined_summary %>%
  filter(horizon == 1)



library(knitr)
library(kableExtra)


# Assuming horizon1_df is your filtered data
horizon1_df_no_horizon <- horizon_1_data %>% select(-horizon)


kable(horizon1_df_no_horizon, format = "latex", booktabs = TRUE, digits = 3,
      caption = "Summary Metrics for Horizon 1") %>%
  kable_styling(latex_options = c("hold_position", stripe_color = NA))


#rm(predictions_table, predictions_table_lstm, predictions_naive_table); gc()

 
# Plots for table locks
#plot_horizon_performance_2_models(correct_table, "Transformer", correct_naive_table, "Naive Baseline",
 #   file_path = construct_output_path("analysis/plots", experiment_subdir_cut, "table-lock_transformer_vs_naive_baseline.pdf"))
#plot_horizon_performance_3_models(correct_table, "Transformer", correct_table_lstm, "LSTM", correct_naive_table, "Naive Baseline",
 #   file_path = construct_output_path("analysis/plots", experiment_subdir_cut, "table-lock_transformer_lstm_vs_naive_baseline.pdf"))
#plot_table_performance_faceted_2_models(correct_table_by_table, "Transformer", correct_naive_table_by_table, "Naive Baseline",
  #  file_path = construct_output_path("analysis/plots", experiment_subdir_cut, "table-lock_transformer_vs_naive_baseline_by_table.pdf"))
#plot_table_performance_faceted_3_models(correct_table_by_table, "Transformer", correct_table_by_table_lstm, "LSTM", correct_naive_table_by_table, "Naive Baseline",
 #   file_path = construct_output_path("analysis/plots", experiment_subdir_cut, "table-lock_transformer_lstm_vs_naive_baseline_by_table.pdf"))
 

 
  horizon_1_data_accuracy <- accuracy_iter_cum_table %>%
  filter(horizon == 1)  
  mean(horizon_1_data_accuracy$accuracy)
  
  horizon_1_data_accuracy <- accuracy_iter_cum_table_lstm %>%
    filter(horizon == 1)  
  mean(horizon_1_data_accuracy$accuracy)
 
  horizon_1_data_accuracy <- accuracy_iter_cum_table_naive %>%
    filter(horizon == 1)  
  mean(horizon_1_data_accuracy$accuracy)
  
  
rq1_plot_horizon_performance_3_models(accuracy_iter_cum_table, "Transformer", accuracy_iter_cum_table_lstm, "LSTM", accuracy_iter_cum_table_naive, "Naive Baseline",
                                  file_path = construct_output_path("analysis/plots", experiment_subdir_cut, "RQ1_table-lock_transformer_lstm_vs_naive_baseline_accuracy_cuml.pdf") , y_col = "accuracy", y_label = "Accuracy")



#Currently not in the paper
#rq1_plot_table_performance_faceted_3_models(correct_table_by_table, "Transformer", correct_table_by_table_lstm, "LSTM", correct_naive_table_by_table, "Naive-Baseline",
                                           # file_path = construct_output_path("analysis/plots", experiment_subdir_cut, "table-lock_transformer_lstm_vs_naive_baseline_by_table_cuml.pdf"))



# Precision/Recall plots for table locks
#generate_and_save_precision_recall_plot(correct_table_by_table,
          #                              file_path = construct_output_path("analysis/plots", experiment_subdir_cut, "table-lock_transformer_precision_recall.pdf"))
#generate_and_save_precision_recall_plot(correct_table_by_table_lstm,
   #                                     file_path = construct_output_path("analysis/plots", experiment_subdir_cut, "table-lock_lstm_precision_recall.pdf"))
#generate_and_save_precision_recall_plot(correct_naive_table_by_table,
                     #                   file_path = construct_output_path("analysis/plots", experiment_subdir_cut, "table-lock_naive_baseline_precision_recall.pdf"))


 