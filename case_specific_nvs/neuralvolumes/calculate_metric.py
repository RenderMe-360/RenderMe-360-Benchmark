import os
import math
import numpy as np
from tqdm import tqdm
import glob

subjects1=['0100','0259','0026','0099','0094','0278','0297','0295','0290','0189']

subjects2=['0041','0168','0253','0250','0175']

subjects3=['0116','0156','0262','0232','0195','0048']


if __name__ == "__main__":
    data_dir = "logs_exp/{person}"
    sum_lpips = []
    sum_ssim = []
    sum_psnr = []
    for subjects in [subjects1, subjects2, subjects3]:
        lpips = []
        ssim = []
        psnr = []
        for person in tqdm(subjects):
            print(f"{person} ******************")
            data_path = data_dir.format(person=person)
            paths = sorted(glob.glob(f"{data_path}/metrics_oneframe_*.npy"))
            for path in paths:
                if not os.path.exists(path):
                    print(f"{path} not exist")
                    continue
                data = np.load(path, allow_pickle=True).item()
                # print(data['psnr'])
                if math.isnan(data['psnr']):
                    continue
                lpips.append(data['lpips'])
                ssim.append(data['ssim'])
                psnr.append(data['psnr'])
        sum_lpips.extend(lpips)
        sum_ssim.extend(ssim)
        sum_psnr.extend(psnr)

        print(f"lpips : {np.mean(lpips):.2f}")
        print(f"ssim : {np.mean(ssim):.3f}")
        print(f"psnr : {np.mean(psnr):.2f}")

    print(f"sum_psnr : {np.mean(sum_psnr):.3f}")
    print(f"sum_ssim : {np.mean(sum_ssim):.3f}")
    print(f"sum_lpips : {np.mean(sum_lpips):.3f}")

