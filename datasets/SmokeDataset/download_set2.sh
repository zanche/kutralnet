#!/bin/bash

# SmokeSet2 dataset
# Author: Center for Wildfire Research
# Website: http://wildfire.fesb.hr/index.php?option=com_content&view=article&id=49&Itemid=54
# Observations: Contains few smoke images for segmentation algorithms

printf "Starting download of SmokeSet2 dataset...\n"
printf "Non-segmented part\n"
../../utils/gdown.pl "https://drive.google.com/file/d/1GWmYge7orseqtRYpUDa_WEJcuWviV6x4/view" "./smoke.zip"
printf  "Segmented part\n"
../../utils/gdown.pl "https://drive.google.com/file/d/1GTBn8E7rt6peKUFcq5QFRHKtIFYflHeE/view" "./smoke_segmented.zip"

printf "Unzipping\n"
unzip -qq "smoke.zip" -d ./SmokeSet2
unzip -qq "smoke_segmented.zip" -d ./SmokeSet2

# del source and unused images
rm "smoke.zip"
rm "smoke_segmented.zip"
rm -r SmokeSet2/gt_* # del rm ground truth

printf "Done!\n"
