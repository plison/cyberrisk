#!/bin/bash
model_dir="models/"
model=$1
gold=$2
pred=$3
model_name=$(echo "$model_dir$model.model")
vocab_name=$(echo "$model_dir$model.vocab")
echo $model_name $vocab_name

python run.py predict \
       --fdata $gold \
       --fpred $pred \
       --model $model_name \
       --vocab $vocab_name
