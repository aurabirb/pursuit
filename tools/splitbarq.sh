#!/bin/bash
for i in {1..9}; do
    mkdir -p datasets/barq$i;
    find barq_images -maxdepth 1 -name "$i*" -exec "mv {} datasets/barq$i/" \; ;
done

for i in {1..9}; do
    echo $i
done | xargs -P 4 -I {} pursuit -ds barq{} ingest barq --data-dir datasets/barq{}
