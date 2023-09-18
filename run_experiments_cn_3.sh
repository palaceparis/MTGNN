#!/bin/bash

# Define an array of seed values to test
seeds=(42 56 89 159 200)

# Define the best hyperparameters
batch_size=32
conv_channels=128
dropout=0.341271958223661
end_channels=8
gcn_depth=4
layers=4
learning_rate=0.00915273292405694
node_dim=54
residual_channels=2
skip_channels=8
subgraph_size=16
weight_decay=0.00011935901028092858

# Define the csv file path and y offset end value
csv_filepath="data/emissions.csv"
y_offset_end_value=4

# Loop over each seed value and run the Python script
for seed in "${seeds[@]}"; do
    python train_multi_step.py \
        --seed $seed \
        --batch_size $batch_size \
        --conv_channels $conv_channels \
        --dropout $dropout \
        --end_channels $end_channels \
        --gcn_depth $gcn_depth \
        --layers $layers \
        --learning_rate $learning_rate \
        --node_dim $node_dim \
        --residual_channels $residual_channels \
        --skip_channels $skip_channels \
        --subgraph_size $subgraph_size \
        --weight_decay $weight_decay \
        --csv_filepath $csv_filepath \
        --y_offset_end_value $y_offset_end_value
done
