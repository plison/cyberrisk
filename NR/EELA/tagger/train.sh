#!/bin/bash
model_dir="models/"
model=$1
train=$2
dev=$3
model_name=$(echo "$model_dir$model.model")
vocab_name=$(echo "$model_dir$model.vocab")
echo $model_name $vocab_name

python run.py train --ftrain $train --fdev $dev --model $model_name \
       --vocab $vocab_name \
       --n_lstm_nodes 400 \
       --n_lstm_layers 3 \
       --device 0

