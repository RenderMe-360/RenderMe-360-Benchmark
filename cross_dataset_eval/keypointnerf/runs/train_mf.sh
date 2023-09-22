DATA=./multiface  # path to multiface
num_gpus=8
python train.py --config ./configs/multiface.json --data_root ${DATA} --num_gpus ${num_gpus}

