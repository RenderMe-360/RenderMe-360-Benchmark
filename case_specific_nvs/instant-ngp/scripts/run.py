#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
import commentjson as json

import numpy as np

import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import *
from scenes import scenes_nerf, scenes_image, scenes_sdf, scenes_volume, setup_colored_sdf

from tqdm import tqdm

import pyngp as ngp # noqa
import lpips
import torch

lpips_net = lpips.LPIPS(net='alex')

def cal_lpips(x, gt, net=lpips_net):
    x = torch.from_numpy(x).float() * 2 - 1.
    gt = torch.from_numpy(gt).float() * 2 - 1.
    x = x.permute([2, 0, 1]).unsqueeze(0)
    gt = gt.permute([2, 0, 1]).unsqueeze(0)
    with torch.no_grad():
        loss = net.forward(x, gt)
    return loss.item()


def parse_args():
	parser = argparse.ArgumentParser(description="Run neural graphics primitives testbed with additional configuration & output options")

	parser.add_argument('--datadir', type=str, default='/mnt/lustre/share_data/chengwei/ngp', help='directory for data')
	parser.add_argument('--subject', type=str, default="huajiangtao3", help='subject to train')
	# parser.add_argument('--frame', type=str, default="0091.jpg", help='frame to run')
	parser.add_argument("--test_views", type=str, nargs='+', default=[], help="List of views to inference new mask")

	parser.add_argument("--resume", action="store_true", default=False, help="Load this snapshot before training.")
	parser.add_argument("--save_ckpt", action="store_true", default=False, help="Save this snapshot after training.")
	
	parser.add_argument("--save_mesh", action="store_true", default=False, help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.")
	parser.add_argument('--render', action="store_true", default=False, help="Output render path to generate free view point video")

	parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")
	parser.add_argument("--sharpen", default=0, help="Set amount of sharpening applied to NeRF training images.")

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_args()

	mode = ngp.TestbedMode.Nerf
	configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")
	list_frame = [f for f in os.listdir(os.path.join(args.datadir, args.subject)) if 'frame' in f]
	arg_frame = list_frame[0]
	framedir = os.path.basename(arg_frame).split('.')[0]
	basedir = os.path.join(args.datadir, args.subject, framedir)

	base_network = os.path.join(configs_dir, "base.json")
	network = base_network
	if not os.path.isabs(network):
		network = os.path.join(configs_dir, network)

	testbed = ngp.Testbed(mode)
	testbed.nerf.sharpen = float(args.sharpen)

	scene = basedir
	testbed.load_training_data(scene)

	ckpt = os.path.join(basedir, 'ckpt.msgpack')
	if args.resume and os.path.exists(ckpt):
		print("Loading snapshot ", ckpt)
		testbed.load_snapshot(ckpt)
	else:
		testbed.reload_network_from_file(network)

	testbed.shall_train = True

	testbed.nerf.render_with_camera_distortion = True

	network_stem = os.path.splitext(os.path.basename(network))[0]

	old_training_step = 0
	n_steps = args.n_steps
	if n_steps < 0:
		n_steps = 100000

	tqdm_last_update = 0
	if n_steps > 0:
		with tqdm(desc="Training", total=n_steps, unit="step") as t:
			while testbed.frame():
				if testbed.want_repl():
					repl(testbed)
				# What will happen when training is done?
				if testbed.training_step >= n_steps:
					break

				# Update progress bar
				if testbed.training_step < old_training_step or old_training_step == 0:
					old_training_step = 0
					t.reset()

				now = time.monotonic()
				if now - tqdm_last_update > 0.1:
					t.update(testbed.training_step - old_training_step)
					t.set_postfix(loss=testbed.loss)
					old_training_step = testbed.training_step
					tqdm_last_update = now

	if args.save_ckpt:
		print("Saving snapshot ", ckpt)
		testbed.save_snapshot(ckpt, False)

	if len(args.test_views) > 0 and os.path.exists(os.path.join(basedir, 'transforms.json')):
		os.makedirs(f'{basedir}/output', exist_ok=True)
		with open(os.path.join(basedir, 'transforms_all.txt')) as f:
			test_transforms = json.load(f)
		data_dir=basedir
		totmse = 0
		totpsnr = 0
		totssim = 0
		totlpips = 0
		totcount = 0
		minpsnr = 1000
		maxpsnr = 0

		# Evaluate metrics on background
		if 'white_transparent' in test_transforms.keys():
			if test_transforms['white_transparent']:
				testbed.background_color = [1.0, 1.0, 1.0, 1.0]
		if 'black_transparent' in test_transforms.keys():
			if test_transforms['black_transparent']:
				testbed.background_color = [0.0, 0.0, 0.0, 0.0]	
		# testbed.background_color = [1.0, 1.0, 1.0, 1.0]
		# Prior nerf papers don't typically do multi-sample anti aliasing.
		# So snap all pixels to the pixel centers.
		testbed.snap_to_pixel_centers = True
		spp = 8

		testbed.nerf.rendering_min_transmittance = 1e-4

		testbed.fov_axis = 1
		
		testbed.shall_train = False

		test_frames = [(i, frame) for i, frame in enumerate(test_transforms["frames"]) if os.path.basename(frame["file_path"]).split('.')[0] in args.test_views]
		print(len(test_frames))
		with tqdm(list(enumerate(test_frames)), unit="images", desc=f"Rendering test frame") as t:
			for i, (fid, frame) in t:
				p = frame["file_path"]
				ref_fname = os.path.join(data_dir, p)
				print(ref_fname)
				# testbed.set_camera_to_training_view(fid)
				transform_matrix = np.array(frame["transform_matrix"], dtype=np.float32)[:3, :4]
				focal_length = np.array([frame["fl_x"], frame["fl_y"]], dtype=np.float32)
				distortion = np.array([frame["k1"], frame["k2"], frame["p1"], frame["p2"]], dtype=np.float32)
				resolution = np.array([frame["w"], frame["h"]], dtype=np.int32)
				principal_point = np.array([frame["cx"], frame["cy"]], dtype=np.float32) / resolution.astype(np.float32)
				testbed.set_camera_to_view(transform_matrix, focal_length, resolution, distortion, principal_point)

				ref_image = read_image(ref_fname)
				testbed.set_nerf_camera_matrix(np.matrix(frame["transform_matrix"])[:-1,:])
				image = testbed.render(ref_image.shape[1], ref_image.shape[0], spp, True)

				if image.shape[1] == 2448:
					image_save = image[:,200:2248]
				else:
					image_save = image
				write_image(f"{basedir}/output/rgba_{i:03d}.png", image_save)
				write_image(f"{basedir}/output/ref_{i:03d}.png", ref_image)
				imageio.imwrite(f"{basedir}/output/mask_{i:03d}.png", (np.clip(image[:,:,-1], 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
				diffimg = np.absolute(image[:,:,:3] - ref_image)
				write_image(f"{basedir}/output/diff_{i:03d}.png", diffimg)
				
				A = np.clip(linear_to_srgb(image[...,:3]), 0.0, 1.0)
				R = np.clip(linear_to_srgb(ref_image[...,:3]), 0.0, 1.0)
				mse = float(compute_error("MSE", A, R))
				ssim = float(compute_error("SSIM", A, R))
				# import pdb; pdb.set_trace()
				res_lpips = cal_lpips(image[...,:3], ref_image)
				totssim += ssim
				totmse += mse
				totlpips += res_lpips
				psnr = mse2psnr(mse)
				totpsnr += psnr
				minpsnr = psnr if psnr<minpsnr else minpsnr
				maxpsnr = psnr if psnr>maxpsnr else maxpsnr
				totcount = totcount+1
				t.set_postfix(psnr = totpsnr/(totcount or 1))
				frame_ind = p.split('/')[-1].split('.')[0] # image/20.jpg
				print(f"single: {args.subject} {frame_ind}, PSNR={psnr}, SSIM={ssim}, LPIPS={res_lpips}") 

		psnr_avgmse = mse2psnr(totmse/(totcount or 1))
		psnr = totpsnr/(totcount or 1)
		ssim = totssim/(totcount or 1)
		res_lpips = totlpips/(totcount or 1)
		print(f"*****result sum: {args.subject}, PSNR={psnr}, [min={minpsnr} max={maxpsnr}], SSIM={ssim}, LPIPS={res_lpips}")
		metrics_dict = {"lpips": res_lpips, "ssim": ssim, "psnr": psnr}
		np.save(os.path.join(basedir, "metrics.npy"), metrics_dict)

	if args.render and os.path.exists(os.path.join(basedir, 'render_transforms.txt')):
		os.makedirs(os.path.join(basedir, 'render_path'), exist_ok=True)
		# save render_transform with file format of txt to aviod wrong loading by TestBed
		with open(os.path.join(basedir, 'render_transforms.txt')) as f:
			render_transforms = json.load(f)

		# Evaluate metrics on background
		if 'white_transparent' in render_transforms.keys():
			if render_transforms['white_transparent']:
				testbed.background_color = [1.0, 1.0, 1.0, 1.0]
		if 'black_transparent' in render_transforms.keys():
			if render_transforms['black_transparent']:
				testbed.background_color = [0.0, 0.0, 0.0, 0.0]	
		testbed.background_color = [1.0, 1.0, 1.0, 1.0] # NOTE:渲染白背景的图像
		# Prior nerf papers don't typically do multi-sample anti aliasing.
		# So snap all pixels to the pixel centers.
		testbed.snap_to_pixel_centers = True
		spp = 8

		testbed.nerf.rendering_min_transmittance = 1e-4

		testbed.fov_axis = 1
		if 'camera_angle_x' in render_transforms.keys():
			testbed.fov = render_transforms["camera_angle_y"] * 180 / np.pi
		elif 'fl_y' in render_transforms.keys():
			testbed.fov = np.arctan2(render_transforms['h']/2, render_transforms['fl_y']) * 2 * 180 / np.pi
		else:
			exit()
		testbed.shall_train = False

		with tqdm(list(enumerate(render_transforms["frames"])), unit="images", desc=f"Rendering test frame") as t:
			for i, frame in t:
				testbed.set_nerf_camera_matrix(np.matrix(frame["transform_matrix"])[:-1,:])
				image = testbed.render(int(render_transforms['w']), int(render_transforms['h']), spp, True)
				write_image(f"{basedir}/render_path/rgba_{i:03d}.png", image)
				write_image(f"{basedir}/render_path/rgb_{i:03d}.png", (image[:,:,:3]*(image[:,:,3:]>0.5).astype(np.float64)).astype(image.dtype))

	if args.save_mesh:
		res = 256
		mesh_dir = os.path.join(basedir, 'mesh.obj')
		print(f"Generating mesh via marching cubes and saving to {mesh_dir}. Resolution=[{res},{res},{res}]")
		testbed.compute_and_save_marching_cubes_mesh(mesh_dir, [res, res, res])


