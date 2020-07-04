#!/bin/bash

# FireFlame dataset
# Author: DeepQuest AI
# Website: https://github.com/DeepQuestAI/Fire-Smoke-Dataset
# Observations: It a custom compilation of fire, smoke and neutral images.
# Some works used this like:
# "An artificial intelligence based crowdsensing solution for on-demand accident scene monitoring"
# "Deep Convolutional Neural Network for Fire Detection"
# Some images are misslabeled, we correct some labels, and complement the fire labeled images 
# with smoke, making multi-label images

printf  "Starting download of dataset...\n"
#wget "https://github.com/DeepQuestAI/Fire-Smoke-Dataset/releases/download/v1/FIRE-SMOKE-DATASET.zip" ""

printf "Unzipping\n"
unzip -qq "FIRE-SMOKE-DATASET.zip"
# move to folder
mv ./FIRE-SMOKE-DATASET/* ./
rm -r FIRE-SMOKE-DATASET

rm "FIRE-SMOKE-DATASET.zip"

printf "Some images needs to be transformed!\n"
printf "Train/Neutral/image_330.jpg "
mogrify -format jpg "Train/Neutral/image_330.jpg"
printf "OK\n"

printf "Done!\n"
