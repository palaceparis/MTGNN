#!/bin/bash

# Define the configurations to be used in the training
configs=(
    "data/emissions.csv 2"
    "data/emissions.csv 4"
    "data/eu_emi.csv 2"
    "data/eu_emi.csv 4"
    "data/us_emi.csv 2"
    "data/us_emi.csv 4"
)

# Create or clear the log file
> training_errors.log

# Loop through each configuration and run the training script
for config in "${configs[@]}"; do
    # Split the config into csv_filepath and y_offset_end_value
    read -r csv_filepath y_offset_end_value <<< "$config"
    
    # Run the Python script with the current configuration
    python train_multi_step.py --csv_filepath $csv_filepath --y_offset_end_value $y_offset_end_value 2>> training_errors.log
    
    # Check if the script executed successfully
    if [ $? -eq 0 ]; then
        echo "Training successful with config: csv_filepath=$csv_filepath, y_offset_end_value=$y_offset_end_value"
    else
        echo "Training failed with config: csv_filepath=$csv_filepath, y_offset_end_value=$y_offset_end_value" | tee -a training_errors.log
        echo "See training_errors.log for details."
    fi
done
