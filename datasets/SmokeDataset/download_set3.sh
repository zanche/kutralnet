#!/bin/bash

# SmokeSet3 dataset
# "Deep Domain Adaptation Based Video Smoke Detection using Synthetic Smoke Images"
# Author: Gao Xu
# Website: http://smoke.ustc.edu.cn/
# Observations: Contains real and synthetic smoke images extracted from video-source.
# This project only use the subset Real Smoke

printf "Starting download of SmokeSet3 dataset...\n"
wget "http://smoke.ustc.edu.cn/train_real_smoke.zip"

printf "Unzipping\n"
unzip -qq "train_real_smoke.zip" -d ./SmokeSet3
# locate inside the folder
mv ./SmokeSet3/train_real_smoke ./SmokeSet3/images
# del source
rm "train_real_smoke.zip"

printf "Done!\n"
