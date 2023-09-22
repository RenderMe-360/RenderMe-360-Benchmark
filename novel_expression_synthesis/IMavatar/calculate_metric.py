import argparse
import os
import math
from metrics import run
import numpy as np
from glob import glob 

name_dic = {
    "testset1": ['0100','0259','0026','0099','0094','0278','0297','0295','0290','0189'],
    "testset2": ['0041','0168','0253','0250','0175'],
    "testset3": ['0116','0156','0262','0232','0195','0048'],
}

if __name__ == '__main__':

    mean_l=[]
    mean_perceptual=[]
    mean_ssim=[]
    mean_psnr=[]
    mean_keypoint=[]
    f = open('bad.txt', 'w')


    # NOTE:speech training(Protocal 2)
    #   exp part
    mean_l=[]
    mean_perceptual=[]
    mean_ssim=[]
    mean_psnr=[]
    mean_keypoint=[]
    for testset in ['testset1', 'testset2', 'testset3']: 
        for name in name_dic[testset]:
            for i in [3,4,5,7,10,11]:
                expression = f'e_{i}'
                output_dir = sorted(glob(f'./IMavatar/data/experiments/{name}/IMavatar/s_5+s_6/eval/{expression}/epoch_*'))[-1]
                gt_dir = f'./IMavatar/data/datasets/{testset}/{name}/{name}/{expression}'
                if not os.path.exists(output_dir):
                    continue
                l, perceptual, ssim, psnr, keypoint = run(output_dir, gt_dir, True)
                if math.isnan(psnr):
                    f.write(f'{name} e_{i} {psnr} \n')
                    continue
                mean_l.append(l)
                mean_keypoint.append(keypoint)
                mean_perceptual.append(perceptual)
                mean_ssim.append(ssim)
                mean_psnr.append(psnr)
            print(f"expression {name}: {np.mean(mean_keypoint):03f}, l1: {np.mean(mean_l):03f}, psnr: {np.mean(mean_psnr):03f}, ssim: {np.mean(mean_ssim):03f}, lpips: {np.mean(mean_perceptual):03f}, ")
            f.write(f"expression {name}: {np.mean(mean_keypoint):03f}, l1: {np.mean(mean_l):03f}, psnr: {np.mean(mean_psnr):03f}, ssim: {np.mean(mean_ssim):03f}, lpips: {np.mean(mean_perceptual):03f}, \n")
    print(f"expression normal: {np.mean(mean_keypoint):03f}, l1: {np.mean(mean_l):03f}, psnr: {np.mean(mean_psnr):03f}, ssim: {np.mean(mean_ssim):03f}, lpips: {np.mean(mean_perceptual):03f}, ")
    f.write(f"expression normal: {np.mean(mean_keypoint):03f}, l1: {np.mean(mean_l):03f}, psnr: {np.mean(mean_psnr):03f}, ssim: {np.mean(mean_ssim):03f}, lpips: {np.mean(mean_perceptual):03f}, \n")

    mean_l=[]
    mean_perceptual=[]
    mean_ssim=[]
    mean_psnr=[]
    mean_keypoint=[]
    for testset in ['testset1', 'testset2', 'testset3']: 
        for name in name_dic[testset]:
            for i in [1,2,6,8,9]:
                expression = f'e_{i}'
                output_dir = sorted(glob(f'./IMavatar/data/experiments/{name}/IMavatar/s_5+s_6/eval/{expression}/epoch_*'))[-1]
                gt_dir = f'./IMavatar/data/datasets/{testset}/{name}/{name}/{expression}'
                if not os.path.exists(output_dir):
                    continue
                l, perceptual, ssim, psnr, keypoint = run(output_dir, gt_dir, True)
                if math.isnan(psnr):
                    f.write(f'{name} e_{i} {psnr} \n')
                    continue
                mean_l.append(l)
                mean_keypoint.append(keypoint)
                mean_perceptual.append(perceptual)
                mean_ssim.append(ssim)
                mean_psnr.append(psnr)
            print(f"expression {name}: {np.mean(mean_keypoint):03f}, l1: {np.mean(mean_l):03f}, psnr: {np.mean(mean_psnr):03f}, ssim: {np.mean(mean_ssim):03f}, lpips: {np.mean(mean_perceptual):03f}, ")
            f.write(f"expression {name}: {np.mean(mean_keypoint):03f}, l1: {np.mean(mean_l):03f}, psnr: {np.mean(mean_psnr):03f}, ssim: {np.mean(mean_ssim):03f}, lpips: {np.mean(mean_perceptual):03f}, \n")
    print(f"expression hard: {np.mean(mean_keypoint):03f}, l1: {np.mean(mean_l):03f}, psnr: {np.mean(mean_psnr):03f}, ssim: {np.mean(mean_ssim):03f}, lpips: {np.mean(mean_perceptual):03f}, ")
    f.write(f"expression hard: {np.mean(mean_keypoint):03f}, l1: {np.mean(mean_l):03f}, psnr: {np.mean(mean_psnr):03f}, ssim: {np.mean(mean_ssim):03f}, lpips: {np.mean(mean_perceptual):03f}, \n")

    #   speech part
    mean_l=[]
    mean_perceptual=[]
    mean_ssim=[]
    mean_psnr=[]
    mean_keypoint=[]
    for testset in ['testset1', 'testset2', 'testset3']:
        for name in name_dic[testset]:
            speech = f's_4'
            print(name)
            output_dir = sorted(glob(f'./IMavatar/data/experiments/{name}/IMavatar/s_5+s_6/eval/{speech}/epoch_*'))[-1]
            gt_dir = f'./IMavatar/data/datasets/{testset}/{name}/{name}/{speech}'
            if not os.path.exists(output_dir):
                continue
            l, perceptual, ssim, psnr, keypoint = run(output_dir, gt_dir, True)
            if math.isnan(psnr):
                f.write(f'{name} s_4 {psnr} \n')
                continue
            mean_l.append(l)
            mean_keypoint.append(keypoint)
            mean_perceptual.append(perceptual)
            mean_ssim.append(ssim)
            mean_psnr.append(psnr)
            print(f"speech {name} {output_dir.split('/')[-1]}: {np.mean(mean_keypoint):03f}, l1: {np.mean(mean_l):03f}, psnr: {np.mean(mean_psnr):03f}, ssim: {np.mean(mean_ssim):03f}, lpips: {np.mean(mean_perceptual):03f}, ")
            f.write(f"speech {name} {output_dir.split('/')[-1]}: {np.mean(mean_keypoint):03f}, l1: {np.mean(mean_l):03f}, psnr: {np.mean(mean_psnr):03f}, ssim: {np.mean(mean_ssim):03f}, lpips: {np.mean(mean_perceptual):03f}, \n")
    print(f"speech : {np.mean(mean_keypoint):03f}, l1: {np.mean(mean_l):03f}, psnr: {np.mean(mean_psnr):03f}, ssim: {np.mean(mean_ssim):03f}, lpips: {np.mean(mean_perceptual):03f}, ")
    f.write(f"speech : {np.mean(mean_keypoint):03f}, l1: {np.mean(mean_l):03f}, psnr: {np.mean(mean_psnr):03f}, ssim: {np.mean(mean_ssim):03f}, lpips: {np.mean(mean_perceptual):03f}, \n")

    '''
    # NOTE: expression training (Protocal 1)
    mean_l=[]
    mean_perceptual=[]
    mean_ssim=[]
    mean_psnr=[]
    mean_keypoint=[]
    for testset in ['testset1', 'testset2', 'testset3']: # ['testset1', 'testset2', 'testset3']
        for name in name_dic[testset]:
            for i in [6, 10, 11]:
                expression = f'e_{i}'
                output_dir = f'./IMavatar/data/experiments/{name}/IMavatar/e_0+e_1+e_2+e_3+e_4+e_5/eval/{expression}/epoch_100'
                gt_dir = f'./IMavatar/data/datasets/{testset}/{name}/{name}/{expression}'
                if not os.path.exists(output_dir):
                    f.write(f'{output_dir}\n')
                    continue
                l, perceptual, ssim, psnr, keypoint = run(output_dir, gt_dir, True)
                if math.isnan(psnr):
                    f.write(f'{name} e_{i} {psnr} \n')
                    continue
                mean_l.append(l)
                mean_keypoint.append(keypoint)
                mean_perceptual.append(perceptual)
                mean_ssim.append(ssim)
                mean_psnr.append(psnr)
            print(f" {name} expression: {np.mean(mean_keypoint):03f}, l1: {np.mean(mean_l):03f}, psnr: {np.mean(mean_psnr):03f}, ssim: {np.mean(mean_ssim):03f}, lpips: {np.mean(mean_perceptual):03f}, ")
            f.write(f" {name} expression: {np.mean(mean_keypoint):03f}, l1: {np.mean(mean_l):03f}, psnr: {np.mean(mean_psnr):03f}, ssim: {np.mean(mean_ssim):03f}, lpips: {np.mean(mean_perceptual):03f}, \n")
    print(f"expression: {np.mean(mean_keypoint):03f}, l1: {np.mean(mean_l):03f}, psnr: {np.mean(mean_psnr):03f}, ssim: {np.mean(mean_ssim):03f}, lpips: {np.mean(mean_perceptual):03f}, ")
    f.write(f"expression: {np.mean(mean_keypoint):03f}, l1: {np.mean(mean_l):03f}, psnr: {np.mean(mean_psnr):03f}, ssim: {np.mean(mean_ssim):03f}, lpips: {np.mean(mean_perceptual):03f}, ")
    mean_l=[]
    mean_perceptual=[]
    mean_ssim=[]
    mean_psnr=[]
    mean_keypoint=[]
    for testset in ['testset1', 'testset2', 'testset3']:
        for name in name_dic[testset]:
            for i in [7, 8, 9]:
                expression = f'e_{i}'
                output_dir = f'./IMavatar/data/experiments/{name}/IMavatar/e_0+e_1+e_2+e_3+e_4+e_5/eval/{expression}/epoch_100'
                gt_dir = f'./IMavatar/data/datasets/{testset}/{name}/{name}/{expression}'
                if not os.path.exists(output_dir):
                    continue
                l, perceptual, ssim, psnr, keypoint = run(output_dir, gt_dir, True)
                if math.isnan(psnr):
                    f.write(f'{name} e_{i} {psnr} \n')
                    continue
                mean_l.append(l)
                mean_keypoint.append(keypoint)
                mean_perceptual.append(perceptual)
                mean_ssim.append(ssim)
                mean_psnr.append(psnr)
            print(f" {name} expression: {np.mean(mean_keypoint):03f}, l1: {np.mean(mean_l):03f}, psnr: {np.mean(mean_psnr):03f}, ssim: {np.mean(mean_ssim):03f}, lpips: {np.mean(mean_perceptual):03f}, ")
            f.write(f" {name} expression: {np.mean(mean_keypoint):03f}, l1: {np.mean(mean_l):03f}, psnr: {np.mean(mean_psnr):03f}, ssim: {np.mean(mean_ssim):03f}, lpips: {np.mean(mean_perceptual):03f}, \n")
    
    print(f"expression: {np.mean(mean_keypoint):03f}, l1: {np.mean(mean_l):03f}, psnr: {np.mean(mean_psnr):03f}, ssim: {np.mean(mean_ssim):03f}, lpips: {np.mean(mean_perceptual):03f}, ")
    f.write(f"expression: {np.mean(mean_keypoint):03f}, l1: {np.mean(mean_l):03f}, psnr: {np.mean(mean_psnr):03f}, ssim: {np.mean(mean_ssim):03f}, lpips: {np.mean(mean_perceptual):03f}, ")
    '''