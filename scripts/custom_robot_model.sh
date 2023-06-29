#!/bin/bash

PYBULLET_DATA_PATH=$(python -c 'import pybullet_data; print(pybullet_data.__path__[0])')

cd $PYBULLET_DATA_PATH
gdown https://drive.google.com/uc?id=1JLa-NBbyTHhLm_-dkt06bdjxqvx6a_Y7
gdown https://drive.google.com/uc?id=1H5aNJ-z7YuBkjlFJODgk3szdHv6DAGY-
unzip -o kniova.zip
unzip -o franka_panda.zip
