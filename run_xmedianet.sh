#!/bin/bash

partial_length_list=(2 3 4 5)
dataset="xmedianet"

for n in "${partial_length_list[@]}"; do
    echo "====================================="
    echo "Running experiment with:"
    echo "Partial Length: $n"
    echo "====================================="

    python train.py \
        --dataset "$dataset" \
        --partial_length "$n" \
        --lr 1e-4
done

echo "All experiments completed!"