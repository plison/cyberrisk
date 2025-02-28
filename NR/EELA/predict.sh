#!/bin/bash
python convert.py $1 $2
python tagger/run.py predict --fdata $2 --fpred $3 --model $5 --vocab $6
python add_text.py $3 $1 $4
