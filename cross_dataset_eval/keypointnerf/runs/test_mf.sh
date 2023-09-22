DATA=./multiface  # path to multiface

python render_multiface.py \
    --config ./configs/multiface.json \
    --data_root ${DATA} \
    --model_ckpt ./logs_multiface/multiface_setting2/ckpts/last.ckpt

