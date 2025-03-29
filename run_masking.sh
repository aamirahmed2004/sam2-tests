#!/bin/bash

# Define the paths
PATH1="/home/user/SoccerNet/jersey-number-recognition/replicating_results/jersey-number-pipeline/data/SoccerNet/test/images"
PATH2="/home/user/SoccerNet/jersey-number-recognition/replicating_results/jersey-number-pipeline/data/SoccerNet/challenge/images"

# Run the first command
echo "Running mask_dataset.py on test images..."
python mask_dataset.py --input_path "$PATH1"

if [ $? -ne 0 ]; then
    echo "Error: Failed to run mask_dataset.py on test images."
    exit 1
fi

# Run the second command
echo "Running mask_dataset.py on challenge images..."
python mask_dataset.py --input_path "$PATH2"

if [ $? -ne 0 ]; then
    echo "Error: Failed to run mask_dataset.py on challenge images."
    exit 1
fi

echo "Both commands executed successfully!"
