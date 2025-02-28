#!/bin/bash
model_dir="models/"
model=$1
gold=$2
model_name=$(echo "$model_dir$model.model")
vocab_name=$(echo "$model_dir$model.vocab")
echo $model_name $vocab_name

python run.py evaluate \
       --fdata $gold \
       --model $model_name \
       --vocab $vocab_name
