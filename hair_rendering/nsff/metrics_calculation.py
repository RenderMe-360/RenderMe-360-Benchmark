
import os, sys, json
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir,'..'))
import imageio
from tqdm import tqdm
import numpy as np
import cv2
from scipy.spatial import KDTree
import lpips
import torch

hair_short = ['0295_h1_2bk','0290_h1_2b','0297_h1_2bk','0259_h1_2b']
hair_long = ['0295_h1_3bk','0290_h1_3bn', '0094_h1_3bk','0278_h1_6bk','0297_h1_3bk','0189_h1_3bk','0259_h1_3bk']
hair_curls = ['0094_h1_4bn','0278_h1_4bn','0297_h1_7bk','0295_h1_7b','0259_h1_7bn']

lpips_net = lpips.LPIPS(net='alex')
def psnr(x, gt):
    """
    x: np.uint8, HxWxC, 0 - 255
    gt: np.uint8, HxWxC, 0 - 255
    """
    x = (x / 255.).astype(np.float32)
    gt = (gt / 255.).astype(np.float32)
    mse = ((x - gt) ** 2).mean()
    psnr = 10. * np.log10(1. / mse)

    # return mse, psnr
    return psnr

def ssim_channel(x, gt):

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    x = x.astype(np.float32)
    gt = gt.astype(np.float32)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(x, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(gt, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(x ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(gt ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(x * gt, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def ssim(x, gt):
    '''calculate SSIM
    the same outputs as MATLAB's
    x: np.uint8, HxWxC, 0 - 255
    gt: np.uint8, HxWxC, 0 - 255
    '''
    if not x.shape == gt.shape:
        raise ValueError('Input images must have the same dimensions.')
    if x.ndim == 2:
        return ssim_channel(x, gt)
    elif x.ndim == 3:
        if x.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim_channel(x, gt))
            return np.array(ssims).mean()
        elif x.shape[2] == 1:
            return ssim_channel(np.squeeze(x), np.squeeze(gt))
    else:
        raise ValueError('input image dimension mismatch.')

def lpips(x, gt, net=lpips_net):
    x = torch.from_numpy(x).float() / 255. * 2 - 1.
    gt = torch.from_numpy(gt).float() / 255. * 2 - 1.
    x = x.permute([2, 0, 1]).unsqueeze(0)
    gt = gt.permute([2, 0, 1]).unsqueeze(0)
    with torch.no_grad():
        loss = net.forward(x, gt)
    return loss.item()


if __name__ == "__main__":
    part = "singleview"
    result_dir = f"./experiments/{part}"
    data_dir = f"./data_hairrendering_nsff_nrnerf/dataset"
    out_dir = f"./experiments/json_out/{part}"
    os.makedirs(out_dir, exist_ok=True)
    sum_lpips = []
    sum_ssim = []
    sum_psnr = []
    for subjects in tqdm([hair_short, hair_long, hair_curls]):
        v_lpips = []
        v_ssim = []
        v_psnr = []
        for subject in subjects:
            dir_img_pred = os.path.join(result_dir, subject, 'render-lockcam-slowmo')
            preds_gt = sorted([f for f in os.listdir(dir_img_pred) if f.endswith('.jpg')])
            dir_img_gt = os.path.join(data_dir, subject, 'mv_images/25')
            imgs_gt = sorted([f for f in os.listdir(dir_img_gt) if f.endswith('.png')])

            meta = {'avg': '', 'frames': {}}
            lpips_list, ssim_list, psnr_list = [], [], []
            num_imgs = len(imgs_gt)
            for i in range(num_imgs):
                if i % 3 == 0: 
                    continue
                img = imageio.v2.imread(os.path.join(dir_img_pred, preds_gt[i]))
                gt = imageio.v2.imread(os.path.join(dir_img_gt, imgs_gt[i]))
                lpips_val = lpips(img, gt)
                ssim_val = ssim(img, gt)
                psnr_val = psnr(img, gt)
                meta['frames'][imgs_gt[i]] = {
                    'psnr': str(psnr_val),
                    'ssim': str(ssim_val),
                    'lpips': str(lpips_val)
                }
                lpips_list.append(lpips_val)
                ssim_list.append(ssim_val)
                psnr_list.append(psnr_val)
                v_lpips.append(lpips_val)
                v_ssim.append(ssim_val)
                v_psnr.append(psnr_val)
            avg_psnr = sum(psnr_list) / len(psnr_list)
            avg_ssim = sum(ssim_list) / len(ssim_list)
            avg_lpips = sum(lpips_list) / len(lpips_list)
            meta['avg'] = f'{avg_psnr:.2f}/{avg_ssim:.3f}/{avg_lpips:.2f}'
            print(subject, meta['avg'])
            out_file = os.path.join(out_dir, f'{subject}.json')

            with open(out_file, 'w') as f:
                json.dump(meta, indent=4, fp=f)
        sum_lpips.extend(v_lpips)
        sum_ssim.extend(v_ssim)
        sum_psnr.extend(v_psnr)
        print(f"psnr : {np.mean(np.array(v_psnr)):.3f}")
        print(f"ssim : {np.mean(np.array(v_ssim)):.3f}")
        print(f"lpips : {np.mean(np.array(v_lpips)):.3f}")
    print(f"sum_psnr : {np.mean(np.array(sum_psnr)):.3f}")
    print(f"sum_ssim : {np.mean(np.array(sum_ssim)):.3f}")
    print(f"sum_lpips : {np.mean(np.array(sum_lpips)):.3f}")
