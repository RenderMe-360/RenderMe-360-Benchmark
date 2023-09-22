import argparse
import json
import math
import os
import sys
import os.path as osp

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from skimage.io import imread
# global
from tqdm import tqdm
import face_alignment
perc_loss_net = None
sifid_net = None

import lpips

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

def keypoint(fa_2d, pred, gt_2dkey):
    pred_2dkey = fa_2d.get_landmarks(pred)[0]
    key_2derror = np.mean(np.sqrt(np.sum((pred_2dkey - gt_2dkey) ** 2, 2)))
    return key_2derror

def img_mse(pred, gt, mask=None, error_type='mse', return_all=False, use_mask=False):
    """
    MSE and variants

    Input:
        pred        :  bsize x 3 x h x w
        gt          :  bsize x 3 x h x w
        error_type  :  'mse' | 'rmse' | 'mae' | 'L21'
    MSE/RMSE/MAE between predicted and ground-truth images.
    Returns one value per-batch element

    pred, gt: bsize x 3 x h x w
    """
    assert pred.dim() == 4
    bsize = pred.size(0)

    if error_type == 'mae':
        all_errors = (pred-gt).abs()
    elif error_type == 'L21':
        all_errors = torch.norm(pred-gt, dim=1)
    elif error_type == "L1":
        all_errors = torch.norm(pred - gt, dim=1, p=1)
    else:
        all_errors = (pred-gt).square()

    if mask is not None and use_mask:
        assert mask.size(1) == 1

        nc = pred.size(1)
        nnz = torch.sum(mask.reshape(bsize, -1), 1) * nc
        all_errors = mask.expand(-1, nc, -1, -1) * all_errors
        errors = all_errors.reshape(bsize, -1).sum(1) / nnz
    else:
        errors = all_errors.reshape(bsize, -1).mean(1)

    if error_type == 'rmse':
        errors = errors.sqrt()

    if return_all:
        return errors, all_errors
    else:
        return errors

def img_psnr(pred, gt, mask=None, rmse=None):
    # https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
    if torch.max(pred) > 128:   max_val = 255.
    else:                       max_val = 1.

    if rmse is None:
        rmse = img_mse(pred, gt, mask, error_type='rmse')

    EPS = 1e-8
    return 20 * torch.log10(max_val / (rmse+EPS))

def run(output_dir, gt_dir, load_npz=False):
    path_result_npz = os.path.join(output_dir, "results_rgb.npz")
    if load_npz and os.path.exists(path_result_npz):
        path_result_npz = os.path.join(output_dir, "results_rgb.npz")
        results = np.load(path_result_npz)
        mse_l = results['mse_l']
        rmse_l = results['rmse_l']
        mae_l = results['mae_l']
        perceptual_l = results['perceptual_l']
        psnr_l = results['psnr_l']
        ssim_l = results['ssim_l']
        keypoint_l = results['keypoint_l']
        # print(psnr_l)
    else:
        subfolders = ['']
        res = 512
        pred_file_name = 'rgb'
        use_mask = True
        only_face_interior = False

        def _load_img(imgpath):
            image = imread(imgpath).astype(np.float32)
            if image.shape[-2] != res:
                image = cv2.resize(image, (res, res))
            image = image / 255.
            if image.ndim >= 3:
                image = image[:, :, :3]
            # 256, 256, 3
            return image

        def _to_tensor(image):
            if image.ndim == 3:
                image = image.transpose(2, 0, 1)
            image = torch.as_tensor(image).unsqueeze(0)
            # 1, 3, 256, 256
            return image

        mse_l = np.zeros(0)
        rmse_l = np.zeros(0)
        mae_l = np.zeros(0)
        perceptual_l = np.zeros(0)
        ssim_l = np.zeros(0)
        psnr_l = np.zeros(0)
        l1_l = np.zeros(0)
        keypoint_l = np.zeros(0)

        # Keep track of where the images come from
        result_subfolders = list()
        result_filenames = list()

        for subfolder_i in range(len(subfolders)):
            subfolder = subfolders[subfolder_i]
            instance_dir = os.path.join(gt_dir, subfolder)
            assert os.path.exists(instance_dir), "Data directory is empty {}".format(instance_dir)
            cam_file = '{0}/flame_params.json'.format(instance_dir)
            with open(cam_file, 'r') as f:
                camera_dict = json.load(f)

            frames = camera_dict['frames']

            fa_2d = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda')

            expressions = {os.path.basename(frame['file_path']): frame["expression"] for frame in camera_dict["frames"]}
            files = os.listdir(os.path.join(output_dir, subfolder, pred_file_name))

            files_nopng = [f[:-4] for f in files]

            files_nopng = [str(int(f)) for f in files_nopng]
            assert len(set(files_nopng).intersection(set(expressions.keys()))) == len(files)
            for i in tqdm(range(len(files))): # subsample here
                filename = files[i]
                filename_nopad = str(int(files[i][:-4])) + ".png"

                pred_path = os.path.join(output_dir, subfolder, pred_file_name, filename)
                pred_for_key = imread(pred_path)

                if pred_for_key.shape[-2] != res:
                    pred_for_key = cv2.resize(pred_for_key, (res, res))

                gt_path = osp.join(os.path.join(gt_dir, subfolder, "image", filename_nopad))
                mask_path = osp.join(os.path.join(gt_dir, subfolder, "mask", filename_nopad))

                pred = _load_img(pred_path)
                gt = _load_img(gt_path)
                mask = _load_img(mask_path)

                w, h, d = gt.shape
                gt = gt.reshape(-1, d)
                # gt[np.sum(gt, 1) == 0., :] = 1 # if background is black, change to white
                gt = gt.reshape(w, h, d)


                gt_2d_key = ((np.array(frames[int(filename[:-4]) - 1]['flame_keypoints'])[None, :, :] + 1.0) * res / 2).astype(int)
                gt_2d_key = gt_2d_key[:, :68, :]
                try:
                    key_error = keypoint(fa_2d, pred_for_key, gt_2d_key)
                except:
                    continue

                if only_face_interior:
                    lmks = gt_2d_key[0]  # 68, 2

                    hull = cv2.convexHull(lmks)
                    hull = hull.squeeze().astype(np.int32)

                    mask = np.zeros(pred_for_key.shape, dtype=np.uint8)
                    mask = cv2.fillPoly(mask, pts=[hull], color=(1, 1, 1))

                pred = _to_tensor(pred)
                gt = _to_tensor(gt)
                mask = _to_tensor(mask)
                mask = mask[:, [0], :, :]
                # Our prediction has white background, so do the same for GT
                gt_masked = gt.clone()
                gt_masked = gt_masked * mask + 1.0 * (1 - mask)
                gt = gt_masked

                l1, error_mask = img_mse(pred, gt, mask=mask, error_type='l1', use_mask=False, return_all=True)

                # if not osp.exists(osp.join(output_dir, subfolder, "err_l1")):
                #    os.mkdir(osp.join(output_dir, subfolder, "err_l1"))
                # cv2.imwrite(osp.join(output_dir, subfolder, "err_l1", filename), 255 * error_mask[0].permute(1,2,0).cpu().numpy())
                mse = img_mse(pred, gt, mask=mask, error_type='mse', use_mask=use_mask)
                rmse = img_mse(pred, gt, mask=mask, error_type='rmse', use_mask=use_mask)
                mae = img_mse(pred, gt, mask=mask, error_type='mae', use_mask=use_mask)

                assert mask.size(1) == 1
                pred = pred[0].permute(1, 2, 0).data.numpy()
                gt = gt[0].permute(1, 2, 0).data.numpy()
                pred = np.clip(255 * pred, 0, 255).astype(np.uint8)
                gt = np.clip(255 * gt, 0, 255).astype(np.uint8)
                # cv2.imwrite('gt.jpg', gt)
                # cv2.imwrite('pred.jpg', pred)
                # exit(0)
                perc_error = lpips(pred, gt)
                ssim_val = ssim(pred, gt)
                psnr_val = psnr(pred, gt)

                mse_l = np.append(mse_l, mse)
                rmse_l = np.append(rmse_l, rmse)
                mae_l = np.append(mae_l, mae)
                perceptual_l = np.append(perceptual_l, perc_error)
                ssim_l = np.append(ssim_l, ssim_val)
                psnr_l = np.append(psnr_l, psnr_val)
                l1_l = np.append(l1_l, l1)
                keypoint_l = np.append(keypoint_l, key_error)


                result_subfolders.append(subfolder)
                result_filenames.append(filename_nopad)

        result = {
            "subfolders": result_subfolders,
            "filenames": result_filenames,
            "mse_l": mse_l.copy(),
            "rmse_l": rmse_l.copy(),
            "mae_l": mae_l.copy(),
            "perceptual_l": perceptual_l.copy(),
            "ssim_l": ssim_l.copy(),
            "psnr_l": psnr_l.copy(),
            "l1_l": l1_l.copy(),
            "keypoint_l": keypoint_l.copy()
        }
        path_result_npz = os.path.join(output_dir, "results_{}.npz".format(pred_file_name))
        path_result_csv = os.path.join(output_dir, "results_{}.csv".format(pred_file_name))
        np.savez(path_result_npz, **result)
        pd.DataFrame.from_dict(result).to_csv(path_result_csv)
        # print("Written result to ", path_result_npz)
    mean_l, mean_perceptual, mean_ssim, mean_psnr, mean_keypoint = np.mean(mae_l), np.mean(perceptual_l), np.mean(ssim_l), np.mean(psnr_l), np.mean(keypoint_l)
    # print("{}\t{}\t{}\t{}\t{}".format(mean_l, mean_perceptual, mean_ssim, mean_psnr, mean_keypoint))
    return mean_l, mean_perceptual, mean_ssim, mean_psnr, mean_keypoint


if __name__ == '__main__':
    # /disks/data2/buehlmar/data/projects_outputs/others/surFace_expression_deadline/zhengyuf
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--output_dir', type=str, help='.')
    parser.add_argument('--gt_dir', type=str, help='.')
    parser.add_argument('--load_npz', default=False, action="store_true", help='If set, load from npz')

    args = parser.parse_args()

    output_dir = args.output_dir
    gt_dir = args.gt_dir
    run(output_dir, gt_dir, args.load_npz)
