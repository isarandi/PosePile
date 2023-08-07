#!/usr/bin/env bash
#@inproceedings{vonMarcard2018,
#title = {Recovering Accurate 3D Human Pose in The Wild Using IMUs and a Moving Camera},
#author = {von Marcard, Timo and Henschel, Roberto and Black, Michael and Rosenhahn, Bodo and Pons-Moll, Gerard},
#booktitle = {European Conference on Computer Vision (ECCV)},
#year = {2018},
#}
# https://virtualhumans.mpi-inf.mpg.de/3DPW/

set -euo pipefail
source posepile/functions.sh
check_data_root

mkdircd "$DATA_ROOT/3dpw"

wget https://virtualhumans.mpi-inf.mpg.de/3DPW/imageFiles.zip
unzip imageFiles.zip
rm imageFiles.zip

wget https://virtualhumans.mpi-inf.mpg.de/3DPW/sequenceFiles.zip
unzip sequenceFiles.zip
rm sequenceFiles.zip
rm -rf __MACOSX

python -m posepile.ds.tdpw.main
