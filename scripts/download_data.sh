#!/bin/bash

mkdir -p data && cd data

# gdown https://drive.google.com/uc?id=1STkAaNGKnn6UvERbLULTftntfTq5kUXM
# unzip -o res_rgr_data_small.zip -d res_rgr_data_small
# rm res_rgr_data_small.zip

# gdown https://drive.google.com/uc?id=1WhvaLFkjBTd9Xzna02aZtSJzzNA9-ma1
# unzip -o res_rgr_data.zip -d res_rgr_data
# rm res_rgr_data.zip

gdown https://drive.google.com/uc?id=1xNbyZoi3gqJtRAc7Xu-M8dGbHsjpGvBs
unzip -o gnn_dyn_data.zip -d gnn_dyn_data
rm gnn_dyn_data.zip
