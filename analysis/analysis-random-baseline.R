# This scripts will compute the number of unique values for table and pageid prediction
# These metrics will support the addition of a random baseline in the paper.
require(tidyverse)
require(arrow)  # to read parquet
library(dplyr)
library(ggplot2)


#setwd("C:/TMU/postdoc-TMU/lock-pred/")
source("analysis/common_functions.R")

#######
#######
#######

#basepath = "C:/TMU/postdoc-TMU/deep-rediscovery/deeptable-analysis/results/"

# Lets load the table- and page-lock predictions

## We will just use the navie model for this since it only has a single iteration

## We will also only use horizon=1 since it will make calculations easier and
## the total number of unique values will be the same for higher horizon values.

predictions_naive_table <- load_parquet("analysis/data/exp-41-naive-rounded-cut-table-locks/predictions.parquet", is_table_lock=TRUE) %>%
  filter(horizon==1)

predictions_naive <- load_parquet("analysis/data/exp-41-naive-rounded-cut-row-locks/predictions.parquet") %>%
  filter(horizon==1)

# Lets first look at table-level and calculate the number of unique table names in our test dataset
unique_tables <- predictions_naive_table %>%
  distinct(gt_table) %>%
  nrow()
print(paste("Number of unique tables in test set:", unique_tables))

table_counts <- predictions_naive_table %>%
  group_by(gt_table) %>%
  summarise(count=n()) %>%
  arrange(desc(count))
print("Table counts:")
print(as.data.frame(table_counts))


# Now lets look at the pageid and calculate the number of unique pageids in our test dataset

unique_pageids <- predictions_naive %>%
  distinct(gt_pageid) %>%
  nrow()
print(paste("Number of unique pageids in test set:", unique_pageids))

pageid_counts <- predictions_naive %>%
  group_by(gt_pageid) %>%
  summarise(count=n()) %>%
  arrange(desc(count))
print("Pageid counts:")
print(as.data.frame(pageid_counts))

min_max_pageid <- predictions_naive %>%
  summarise(min_pageid = min(gt_pageid), max_pageid = max(gt_pageid))
print(paste("Min pageid:", min_max_pageid$min_pageid))
print(paste("Max pageid:", min_max_pageid$max_pageid))
