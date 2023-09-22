import os
import sys
import json


def get_start_from_json(path):
    start = 0
    with open(path, 'r') as f:
        dic = json.load(f)
    meta = dic['frames'] 
    img_ids = sorted([meta[i]['img_id'] for i in range(len(meta))])
    start = img_ids[0]
    return start


name = sys.argv[1] # AA_189

for i in [1]:
    dir = "%s_s%d_0"%(name,i)
    val_json_path = 'dataset/%s/transforms_val.json'%(dir)
    start_frame = get_start_from_json(val_json_path)
    start_time = float(start_frame/25)

    print('test validation frames')
    cmd = "python -u NeRFs/TorsoNeRF/run_nerf.py \
        --config dataset/%s/TorsoNeRFTest_config_s123456.txt \
        --aud_file=dataset/%s/aud.npy \
        --test_save_folder pretrain_test_s%daud_val \
        --test_size=-1 \
        --aud_start=%d "%(dir, dir, i, start_frame)
    print(cmd)
    print()
    os.system(cmd)

    print('concat audio')
    cmd = "ffmpeg -ss %f \
            -i dataset/%s/aud.wav \
            -q:a 0 \
            dataset/%s/aud_val.wav -y"%(start_time, dir, dir)
    os.system(cmd)
    
    cmd = "ffmpeg -i dataset/%s/logs/%s_com/pretrain_test_s%daud_val/result.avi \
            -i dataset/%s/aud_val.wav \
            -shortest \
            -q:v 0 \
            dataset/%s/logs/%s_com/pretrain_test_s%daud_val/result_aud.avi -y "%(dir, dir, i, dir, dir, dir, i)
    os.system(cmd)