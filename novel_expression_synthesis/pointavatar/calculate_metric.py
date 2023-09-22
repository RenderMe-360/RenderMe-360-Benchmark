import os
import numpy as np
from glob import glob

name_dic = {
    "testset1": ['0259','0100','0099','0094','0278','0295','0290','0189','0026','0297'],
    "testset2": ['0041','0168','0250','0175','0253'],
    "testset3": ['0116','0156','0262','0232','0195','0048'],
}

if __name__ == "__main__":

    # NOTE: expression training (protocal 1)
    data_path = './data/experiments/{person}/pointavatar/e_0+e_1+e_2+e_3+e_4+e_5/eval/{expression}/epoch_*'
    subject = name_dic['testset1'] + name_dic['testset2'] + name_dic['testset3']
    mae_l, perceptual_l, ssim_l, psnr_l, l1_l = [], [], [], [], []
    for person in subject:
        mae_p, perceptual_p, ssim_p, psnr_p, l1_p = [], [], [], [], []
        for i in [7,8,9]: # [7,8,9],[6,10,11]
            exp = f'e_{i}'
            data = data_path.format(person=person, expression=exp)
            try:
                path = sorted(glob(data))[-1]
            except:
                continue
            if not os.path.exists(os.path.join(path, 'results_no_cloth_rgb_erode_dilate.npz')):
                # print(path)
                continue
            metrics = np.load(os.path.join(path, 'results_no_cloth_rgb_erode_dilate.npz')) # results_no_cloth_rgb_erode_dilate
            mae_p.append(np.mean(metrics["mae_l"]))
            perceptual_p.append(np.mean(metrics["perceptual_l"]))
            ssim_p.append(np.mean(metrics["ssim_l"]))
            psnr_p.append(np.mean(metrics["psnr_l"]))
            l1_p.append(np.mean(metrics["l1_l"]))
            # print(f'{person} {exp}: PSNR: {np.mean(metrics["psnr_l"])}, SSIM: {np.mean(metrics["ssim_l"])}, LPIPS: {np.mean(metrics["perceptual_l"])}, L1: {np.mean(metrics["l1_l"])}')
        mae_l.extend(mae_p)
        perceptual_l.extend(perceptual_p)
        ssim_l.extend(ssim_p)
        psnr_l.extend(psnr_p)
        l1_l.extend(l1_p)
        print(f"mean of {person} PSNR: {np.mean(psnr_p)}, SSIM: {np.mean(ssim_p)}, LPIPS: {np.mean(perceptual_p)}, L1: {np.mean(l1_p)}")

    print(f"mean of all: PSNR: {np.mean(psnr_l)}, SSIM: {np.mean(ssim_l)}, LPIPS: {np.mean(perceptual_l)}, L1: {np.mean(l1_l)}")

    '''
    # NOTE: speech training (protocal 2)
    data_path = './data/experiments/audio_audio_multiflame/{person}/pointavatar/s_5+s_6/eval/{expression}/epoch_*'
    # test speech
    subject = name_dic['testset1'] + name_dic['testset2'] + name_dic['testset3']
    mae_l, perceptual_l, ssim_l, psnr_l, l1_l = [], [], [], [], []
    for person in subject:
        speech = f's_4'
        data = data_path.format(person=person, expression=speech)
        try:
            path = sorted(glob(data))[-1]
        except:
            continue
        if not os.path.exists(os.path.join(path, 'results_no_cloth_rgb_erode_dilate.npz')):
            print(path)
            continue
        metrics = np.load(os.path.join(path, 'results_no_cloth_rgb_erode_dilate.npz')) # results_no_cloth_rgb_erode_dilate
        mae_l.append(np.mean(metrics["mae_l"]))
        perceptual_l.append(np.mean(metrics["perceptual_l"]))
        ssim_l.append(np.mean(metrics["ssim_l"]))
        psnr_l.append(np.mean(metrics["psnr_l"]))
        l1_l.append(np.mean(metrics["l1_l"]))
    
    print(f"mean of all: PSNR: {np.mean(psnr_l)}, SSIM: {np.mean(ssim_l)}, LPIPS: {np.mean(perceptual_l)}, L1: {np.mean(l1_l)}")
    # test expression
    mae_l, perceptual_l, ssim_l, psnr_l, l1_l = [], [], [], [], []
    for person in subject:
        mae_p, perceptual_p, ssim_p, psnr_p, l1_p = [], [], [], [], []
        for i in [1,2,6,8,9]: # [3,4,5,7,10,11], [1,2,6,8,9]
            exp = f'e_{i}'
            data = data_path.format(person=person, expression=exp)
            try:
                path = sorted(glob(data))[-1]
            except:
                continue
            if not os.path.exists(os.path.join(path, 'results_no_cloth_rgb_erode_dilate.npz')):
                # print(path)
                continue
            metrics = np.load(os.path.join(path, 'results_no_cloth_rgb_erode_dilate.npz')) # results_no_cloth_rgb_erode_dilate
            mae_p.append(np.mean(metrics["mae_l"]))
            perceptual_p.append(np.mean(metrics["perceptual_l"]))
            ssim_p.append(np.mean(metrics["ssim_l"]))
            psnr_p.append(np.mean(metrics["psnr_l"]))
            l1_p.append(np.mean(metrics["l1_l"]))
            # print(f'{person} {exp}: PSNR: {np.mean(metrics["psnr_l"])}, SSIM: {np.mean(metrics["ssim_l"])}, LPIPS: {np.mean(metrics["perceptual_l"])}, L1: {np.mean(metrics["l1_l"])}')
        mae_l.extend(mae_p)
        perceptual_l.extend(perceptual_p)
        ssim_l.extend(ssim_p)
        psnr_l.extend(psnr_p)
        l1_l.extend(l1_p)
        print(f"mean of {person} PSNR: {np.mean(psnr_p)}, SSIM: {np.mean(ssim_p)}, LPIPS: {np.mean(perceptual_p)}, L1: {np.mean(l1_p)}")

    print(f"mean of all: PSNR: {np.mean(psnr_l)}, SSIM: {np.mean(ssim_l)}, LPIPS: {np.mean(perceptual_l)}, L1: {np.mean(l1_l)}")
    '''
