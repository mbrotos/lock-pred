#!/bin/bash

# NOTE: You must run this script from the root of the repository.
# Alternatively, if you are using RStudio, make sure to set the working directory to the root of the repository.
# Also, make sure the directory 'analysis/data' exists and contains the parquet files.
# See gDrive for data: https://drive.google.com/open?id=1pmV-jmh35HMGpcIWniGUFs1supIWzkdd&usp=drive_fs

Rscript -e "if (!require('tidyverse')) install.packages('tidyverse', repos='https://cloud.r-project.org/')"
Rscript -e "if (!require('arrow')) install.packages('arrow', repos='https://cloud.r-project.org/')"
Rscript -e "if (!require('dplyr')) install.packages('dplyr', repos='https://cloud.r-project.org/')"
Rscript -e "if (!require('ggplot2')) install.packages('ggplot2', repos='https://cloud.r-project.org/')"


Rscript analysis/analysis-reduced.R
