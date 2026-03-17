#!/bin/bash

partial_ratio_list=(0.1 0.2 0.3 0.4)
dataset="wiki"

for n in "${partial_ratio_list[@]}"; do
    echo "====================================="
    echo "Running experiment with:"
    echo "Partial Ratio: $n"
    echo "====================================="

    python train.py \
        --dataset "$dataset" \
        --partial_ratio "$n" \
        --lr 1e-4
done

echo "All experiments completed!"