#!/bin/bash

# SmokeSet1 dataset
# "Adversarial Adaptation From Synthesis to Reality in Fast Detector For Smoke Detection"
# Author: Gao Xu
# Website: http://smoke.ustc.edu.cn/
# Observations: Contains real smoke images extracted from video-source.
# This project only use few images from the dataset due a slight difference between them.

printf "Starting download of SmokeSet1 dataset...\n"
wget "http://smoke.ustc.edu.cn/real_images.rar"

printf "Unzipping\n"
unrar x "real_images.rar"
# locate inside the folder
mkdir SmokeSet1
mv ./REALImages ./SmokeSet1/images
# del source
rm "real_images.rar"

printf "Done!\n"
