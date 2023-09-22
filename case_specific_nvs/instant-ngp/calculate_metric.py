import os, sys, json
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir,'..'))
from tqdm import tqdm
import numpy as np
import glob



if __name__ == '__main__':
    base_dir = "./data_NVS_instant-ngp"
    paths = sorted(glob.glob(os.path.join(base_dir, f"*/*/*/*/metrics.npy")))

    metrics = {}
    for path in tqdm(paths):
        # eg:./benchmarks/instant-ngp/logs_2048_train56_test4/set3/0156/0156_e_10/frame00118/metrics.npy
        s = path.split('/')
        sets, sub = s[-5], s[-4]
        if sets not in metrics.keys():
            metrics[sets] = {}
        if sub not in metrics.keys():
            metrics[sets][sub] = {
                'psnr': 0,
                'ssim': 0,
                'lpips': 0,
                'count': 0
            }
        data = np.load(path, allow_pickle=True).item()
        metrics[sets][sub]['psnr'] += data['psnr']
        metrics[sets][sub]['ssim'] += data['ssim']
        metrics[sets][sub]['lpips'] += data['lpips']
        metrics[sets][sub]['count'] += 1
    
    psnr_all = [];ssim_all = []; lpips_all = []
    for sets in metrics.keys():
        psnr_set = [];ssim_set = []; lpips_set = []
        for sub in metrics[sets].keys():
            psnr = metrics[sets][sub]['psnr'] / metrics[sets][sub]['count']
            ssim = metrics[sets][sub]['ssim'] / metrics[sets][sub]['count']
            lpips = metrics[sets][sub]['lpips'] / metrics[sets][sub]['count']
            print(f"{sub}: PSNR: {psnr:.2f}, SSIM: {ssim:.3f}, LPIPS: {lpips:.2f}")
            psnr_set.append(psnr)
            ssim_set.append(ssim)
            lpips_set.append(lpips)
        set_psnr = np.array(psnr_set).mean()
        set_ssim = np.array(ssim_set).mean()
        set_lpips = np.array(lpips_set).mean()
        print(f"{sets}: PSNR: {set_psnr:.2f}, SSIM: {set_ssim:.3f}, LPIPS: {set_lpips:.2f}")
        psnr_all.extend(psnr_set)
        ssim_all.extend(ssim_set)
        lpips_all.extend(lpips_set)
    all_psnr = np.array(psnr_all).mean()
    all_ssim = np.array(ssim_all).mean()
    all_lpips = np.array(lpips_all).mean()
    print(f"ALL: PSNR: {all_psnr:.2f}, SSIM: {all_ssim:.3f}, LPIPS: {all_lpips:.2f}")

