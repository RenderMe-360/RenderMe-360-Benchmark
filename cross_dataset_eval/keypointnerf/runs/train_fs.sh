DATA=./facescape/multi_view_data # path to facescape
num_gpus=8
python train.py --config ./configs/facescape.json --data_root ${DATA} --num_gpus ${num_gpus}

