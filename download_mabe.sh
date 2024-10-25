#!/bin/bash

# Script to download files from https://data.caltech.edu/records/8kdn3-95j37

# Base URL
base_url="https://data.caltech.edu/records/8kdn3-95j37/files"

mkdir -p data/mabe

# Download each file
wget "${base_url}/mouse_triplet_train.npy" -O "data/mabe/mouse_triplet_train.npy"
wget "${base_url}/mouse_triplet_test.npy" -O "data/mabe/mouse_triplet_test.npy"
wget "${base_url}/mouse_triplets_test_labels.npy" -O "data/mabe/mouse_triplets_test_labels.npy"
wget "${base_url}/fly_group_train.npy" -O "data/mabe/fly_group_train.npy"
wget "${base_url}/fly_group_test.npy" -O "data/mabe/fly_group_test.npy"
wget "${base_url}/fly_groups_test_labels.npy" -O "data/mabe/fly_groups_test_labels.npy"

# Script to download the robot dataset
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1q0a6etvda3XJ498lkDdfpbkz6zGqYXAs' -O data/isaac/robot_dataset.npy

echo "Download complete."
