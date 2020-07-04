#!/bin/bash

printf "Starting download of each SmokeSet...\n"
# SmokeSet1 dataset
./download_set1.sh
# SmokeSet2 dataset
./download_set2.sh
# SmokeSet3 dataset
./download_set3.sh
# SmokeSet4 dataset
./download_set4.sh

printf "Done!\n"