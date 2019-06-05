#!/bin/bash

# Download datasets
TRAIN_IMG_FILE_NAME="train-images-idx3-ubyte.gz"
TRAIN_LABELS_FILE_NAME="train-labels-idx1-ubyte.gz"
TEST_IMG_FILE_NAME="t10k-images-idx3-ubyte.gz"


mkdir data
cd data

echo "Downloading training images..."
while true; do
  wget -O "$TRAIN_IMG_FILE_NAME" -c "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz" && break
done
gunzip "$TRAIN_IMG_FILE_NAME"

echo "Downloading training labels..."
while true; do
  wget -O "$TRAIN_LABELS_FILE_NAME" -c "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz" && break
done
gunzip "$TRAIN_LABELS_FILE_NAME"

echo "Downloading test images..."
while true; do
  wget -O "$TEST_IMG_FILE_NAME" -c "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz" && break
done
gunzip "$TEST_IMG_FILE_NAME"

cd ..
