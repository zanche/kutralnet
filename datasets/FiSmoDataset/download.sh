#!/bin/bash

# FiSmo dataset
# Author: M Cazzolato et al.
# Website: https://github.com/mtcazzolato/dsw2017
# Observations: Developed in "FiSmo: A Compilation of Datasets from Emergency Situations for Fire and Smoke Analysis"
# It a custom compilation of fire, smoke and neutral images.
# Some images are misslabeled, we corrected some labels.
printf  "Starting download of dataset...\n"
../../utils/gdown.pl "https://drive.google.com/file/d/1Cq9KGYzmQ2IlFnkWyji-03DSJWZY36jS/view" "./FiSmo-Images.zip"

printf "Unzipping\n"
unzip -qq "FiSmo-Images.zip"
mv FiSmo-Images/* ./
rm "FiSmo-Images.zip"

printf "Done!"
