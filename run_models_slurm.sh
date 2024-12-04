#!/bin/bash

# Navigate to the working directory
cd '/scratch/opt/tle/FluidDynamicsSimulationChallenge'

# Activate the virtual environment
if [ -f /scratch/opt/tle/FluidDynamicsSimulationChallenge/myenv/bin/activate ]; then
    source /scratch/opt/tle/FluidDynamicsSimulationChallenge/myenv/bin/activate
else
    echo "Virtual environment not found. Exiting."
    exit 1
fi

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

# Train and test ThaiRNN
echo "Training and testing ThaiRNN ..."
python3 /scratch/opt/tle/FluidDynamicsSimulationChallenge/main.py --model=thai_rnn

# Train and test NamRNN
echo "Training and testing NamRNN ..."
python3 /scratch/opt/tle/FluidDynamicsSimulationChallenge/main.py --model=nam_rnn

