#!/bin/bash

mkdir -p data && cd data
mkdir -p gnn_dyn_model/2023-01-28-10-42-05-114323 && cd gnn_dyn_model/2023-01-28-10-42-05-114323
gdown https://drive.google.com/uc?id=1uWPOHmpvGYSdun4WqP06zD6YBqZEYHbf
cd ../..
mkdir -p res_rgr_model/2023-01-30-16-17-30-292500 && cd res_rgr_model/2023-01-30-16-17-30-292500
gdown https://drive.google.com/uc?id=1DBBV978BdelcPo8ff3FalzJ4ACyijzQ0
