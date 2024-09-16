#! /bin/bash

experiments=("exp-1" "exp-2" "exp-3" "exp-4" "exp-5" "exp-5-large")
for experiment in "${experiments[@]}"; do
    zip -r "results/$experiment.zip" "results/$experiment"
done

experiments=("exp-1" "exp-2" "exp-3" "exp-4" "exp-5" "exp-5-large")
for experiment in "${experiments[@]}"; do
    unzip "$experiment.zip"
done
