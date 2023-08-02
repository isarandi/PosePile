#!/usr/bin/env bash
#  @inproceedings{bhatnagar22behave,
#  title = {BEHAVE: Dataset and Method for Tracking Human Object Interactions},
#  author={Bhatnagar, Bharat Lal and Xie, Xianghui and Petrov, Ilya and Sminchisescu, Cristian and Theobalt, Christian and Pons-Moll, Gerard},
#  booktitle = {{IEEE} Conference on Computer Vision and Pattern Recognition (CVPR)},
#  year = {2022},
#  }
# https://virtualhumans.mpi-inf.mpg.de/behave/
set -euo pipefail
source functions.sh
check_data_root

mkdircd "$DATA_ROOT/behave"

for i in {01..07}; do
  wget "https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/Date${i}.zip"
done

wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/calibs.zip
wget https://datasets.d2.mpi-inf.mpg.de/cvpr22behave/split.json

unzip "Date*.zip" -d sequences
rm Date*.zip

unzip calibs.zip
rm calibs.zip

python -m humcentr_cli.detect_people --image-root="$DATA_ROOT/behave" --out-path="$DATA_ROOT/behave/yolov4_detections.pkl" --file-pattern='**/*.color.jpg'

python -m posepile.ds.behave.main
