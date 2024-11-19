#!/bin/bash

# Check if logs folder exists, create it if not
if [ ! -d "logs" ]; then
    echo "logs folder not found. Creating 'logs' folder..."
    mkdir logs
fi

# Check if models folder exists, create it if not
if [ ! -d "models" ]; then
    echo "models folder not found. Creating 'models' folder..."
    mkdir models
fi

echo "Training and testing ThaiRNN ..."
python3 main.py --model=thai_rnn

echo "Training and testing NamRNN ..."
python3 main.py --model=nam_rnn

