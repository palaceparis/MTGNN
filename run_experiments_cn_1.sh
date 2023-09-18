#!/bin/bash

# Define an array of seed values to test
seeds=(42 56 89 159 200)

# Define the best hyperparameters
batch_size=64
conv_channels=16
dropout=0.49772964695405997
end_channels=4
gcn_depth=3
layers=3
learning_rate=0.007777554997048703
node_dim=21
residual_channels=64
skip_channels=4
subgraph_size=22
weight_decay=4.588263268580559e-05

# Define the csv file path and y offset end value
csv_filepath="data/emissions.csv"
y_offset_end_value=2

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
