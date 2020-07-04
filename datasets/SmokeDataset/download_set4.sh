#!/bin/bash

# SmokeSet4 dataset
# "A Deep Normalization and Convolutional Neural Network for Image Smoke Detection"
# Author: Feiniu Yuan
# Website: http://staff.ustc.edu.cn/%7Eyfn/vsd.html
# Observations: Contains 4 subset, many of the images are almost the same due to video-source extraction.
# This project only use the subset 4, leaving images greater than 48x48 pixels

printf "Starting download of SmokeSet4 dataset...\n"
wget "http://staff.ustc.edu.cn/%7Eyfn/set_smoke2254_non8363.zip"

printf "Unzipping\n"
unzip -qq "set_smoke2254_non8363.zip" -d ./SmokeSet4

# del unused images
cd ./SmokeSet4/non 
ls | grep -P ".*(48_).*" | xargs -d"\n" rm
cd ../..

# del source
rm "set_smoke2254_non8363.zip"

printf "Done!\n"
