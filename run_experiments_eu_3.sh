#!/bin/bash

# Define an array of seed values to test
seeds=(42 56 89 159 200)

# Define the best hyperparameters
batch_size=64
conv_channels=8
dropout=0.4699211392772475
end_channels=8
gcn_depth=4
layers=4
learning_rate=0.006813387761851225
node_dim=21
residual_channels=16
skip_channels=8
subgraph_size=22
weight_decay=7.490814196537802e-05

# Define the csv file path and y offset end value
csv_filepath="data/eu_emi.csv"
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
