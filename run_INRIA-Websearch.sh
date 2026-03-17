#!/bin/bash

partial_ratio_list=(0.01 0.02 0.03 0.04)
dataset="INRIA-Websearch"

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