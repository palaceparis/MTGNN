#!/bin/bash

# Define an array of seed values to test
seeds=(42 56 89 159 200)

# Define the best hyperparameters
batch_size=32
conv_channels=64
dropout=0.19764406259666278
end_channels=8
gcn_depth=5
layers=7
learning_rate=0.0068541286792381155
node_dim=26
residual_channels=16
skip_channels=32
subgraph_size=15
weight_decay=0.00024126065599331553

# Define the csv file path and y offset end value
csv_filepath="data/us_emi.csv"
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
