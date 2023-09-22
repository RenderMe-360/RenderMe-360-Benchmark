import os
from tqdm import tqdm, trange

hairstyle_list = [i.strip() for i in open('hairstyle_plus.txt', 'r').readlines()]
haircolor_list = [i.strip() for i in open('haircolor_plus.txt', 'r').readlines()]

path = 'xxx/pti/inversion_results/checkpoints_converted/' # path of converted checkpoints
names = sorted(os.listdir(path))

for name in tqdm(names):
    for i in tqdm(haircolor_list + hairstyle_list):
        cmd = "python -u editing/styleclip/edit_pti.py \
        --exp_dir xxx/pti/editing_results \
        --stylegan_weights xxx/pti/inversion_results/checkpoints_converted/%s \
        --latents_path xxx/pti/inversion_results/embeddings/barcelona/PTI/%s/0.pt \
        --image_name %s \
        --neutral_text 'a face' \
        --target_tex 'a face with %s' "%(name, name[19:-5], name[19:-5], i)
        # print(cmd)
        os.system(cmd)
