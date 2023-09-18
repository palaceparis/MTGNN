#!/bin/bash

# List of all the scripts to run
scripts=(
  "run_experiments_cn_1.sh"
  "run_experiments_cn_3.sh"
  "run_experiments_eu_3.sh"
  "run_experiments.sh"
  "run_experiments_us_1.sh"
  "run_experiments_us_3.sh"
)

# Loop over each script and run it
for script in "${scripts[@]}"; do
  # Check if the script file exists and is executable
  if [[ -x "$script" ]]; then
    echo "Running script: $script"
    ./$script
  else
    echo "Script $script not found or not executable"
  fi
done
