#!/usr/bin/env bash
# @inproceedings{singleshotmultiperson2018,
#title = {Single-Shot Multi-Person 3D Pose Estimation From Monocular RGB},
#author = {Mehta, Dushyant and Sotnychenko, Oleksandr and Mueller, Franziska and Xu, Weipeng and Sridhar, Srinath and Pons-Moll, Gerard and Theobalt, Christian},
#booktitle = {3D Vision (3DV), 2018 Sixth International Conference on},
#year = {2018},
#}
# http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson
set -euo pipefail
source functions.sh
check_data_root

mkdircd "$DATA_ROOT/mupots"

wget http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/content/mupots-3d-eval.zip
unzip mupots-3d-eval.zip
mv mupots-3d-eval/* ./
rmdir mupots-3d-eval
rm mupots-3d-eval.zip

wget http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/MultiPersonTestSet.zip
unzip MultiPersonTestSet.zip
mv MultiPersonTestSet/* ./
rmdir MultiPersonTestSet
rm MultiPersonTestSet.zip

python -m posepile.ds.mupots.calibrate_intrinsics

# Originally detection was run with the https://github.com/isarandi/darknet repo as follows:
# darknet/run_yolo.sh --image-root "$DATA_ROOT/mupots" --out-path "$DATA_ROOT/mupots/yolov3_detections.pkl"
# The newer version is:
python -m humcentr_cli.detect_people --image-root="$DATA_ROOT/mupots" --out-path="$DATA_ROOT/mupots/yolov4_detections.pkl"