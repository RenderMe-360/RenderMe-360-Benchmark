DATA=./facescape/multi_view_data # path to facescape
python render_facescape.py \
    --config ./configs/facescape.json \
    --data_root ${DATA} \
    --model_ckpt ./logs_facescape/facescape_setting2/ckpts/last.ckpt
