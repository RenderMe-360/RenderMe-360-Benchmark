# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Render object from training camera viewpoint or novel viewpoints."""
import argparse
import importlib
import importlib.util
import os
import sys
import time
from metrics import psnr, masked_psnr, lpips, ssim, masked_psnr_new
sys.dont_write_bytecode = True
import numpy as np
import torch.utils.data
import imageio, cv2
torch.backends.cudnn.benchmark = True # gotta go fast!

def import_module(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Render')
    parser.add_argument('experconfig', type=str, help='experiment config file')
    parser.add_argument('--profile', type=str, default="Render", help='config profile')
    parser.add_argument('--datadir', type=str, default="./data/", help='directory for data')
    parser.add_argument('--annotdir', type=str, default="./data/PARAMS", help='directory for annotation')
    parser.add_argument('--outdir', type=str, default="logs", help='directory for annotation')
    parser.add_argument('--subject', type=str, default="emo", help='subject to train')
    parser.add_argument('--devices', type=int, nargs='+', default=[0], help='devices')
    parser.add_argument('--move_cam', type=int, default=0, help='move cam')
    parser.add_argument('--cam', type=int, default=None, help='cam')
    parser.add_argument('--ws_factor', type=float, nargs='+', default=[1.0], help='world scale = 2.0 / ws_factor')
    parser.add_argument('--mode', type=str, default="exp", help='sequence or expression or speech')
    parser.add_argument('--bg', action='store_true', help='estimate background')

    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg, type=eval)
    args = parser.parse_args()

    outpath = os.path.join(args.outdir, args.subject)
    print(" ".join(sys.argv))
    print("Output path:", outpath)

    # load config
    experconfig = import_module(args.experconfig, "config_renderme360")
    # print({k: v for k, v in vars(args).items()})
    profile = getattr(experconfig, args.profile)(subject=args.subject, showtarget=False, showdiff=False, viewtemplate=False)

    # load datasets
    dataset = profile.get_dataset(args.datadir, args.annotdir, args.subject, ws_factor=args.ws_factor[0], move_cam=args.move_cam, mode=args.mode, cam = args.cam)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # data writer
    writer = profile.get_writer(args.outdir)

    # build autoencoder
    # ae = profile.get_autoencoder(dataset)
    try:
        ae = torch.load("{}/aeparams.pt".format(outpath))
        ae = torch.nn.DataParallel(ae, device_ids=args.devices).to("cuda").eval()

    except:
        ae = profile.get_autoencoder(dataset)
        ae = ae.to("cuda").eval()
        state_dict = ae.state_dict()
        trained_state_dict = torch.load("{}/aeparams.pt".format(outpath))
        trained_state_dict = {k: v for k, v in trained_state_dict.items() if k in state_dict}
        state_dict.update(trained_state_dict)
        ae.load_state_dict(state_dict, strict=False)
    
    # eval
    iternum = 0
    itemnum = 0
    starttime = time.time()
    rgbs2write = []

    with torch.no_grad():
        P = []
        L = []
        S = []
        for i,data in enumerate(dataloader):
            # st()
            b = next(iter(data.values())).size(0)
            _dict = {k: x.to("cuda") for k, x in data.items()}
            
            #   output = ae(iternum, [], **({k: x.to("cuda") for k, x in data.items()}.pop('mask_face').pop('mask_hair')), **profile.get_ae_args())
            output = ae(iternum, [], **_dict, **profile.get_ae_args())
            encoding = output['encoding']

            #   np.save(f'nv_{args.subject}_{i:05d}.npy', 'rb', encoding.detach().cpu().numpy())
            os.makedirs(f'{outpath}/encoding', exist_ok=True)
            os.makedirs(f'{outpath}/gt_img', exist_ok=True)
            np.save(os.path.join(outpath, 'encoding', f'nv_{args.subject}_{i:05d}.npy'), encoding.detach().cpu().numpy())
            
            for batch_idx in range(len(output['irgbrec'])):
                # print(f"irgbrec shape: {output['irgbrec'].shape}")
                pred =  output["irgbrec"][batch_idx].data.to("cpu").numpy().transpose((1, 2, 0))[..., ::-1].astype(np.uint8).copy()
                pred = np.clip(pred, 0, 255)
                pred = pred + (1. - output['ialpharec'].data.to("cpu").numpy()[:, 0, :, :, None]) * np.array([255, 255, 255])[None, None, None, :]
                pred = np.clip(pred, 0, 255)


                rgbs2write.append(pred[..., ::-1])
                gt = data["image"][batch_idx].data.to("cpu").numpy().transpose((1, 2, 0))[..., ::-1].copy()
                cv2.imwrite(os.path.join(outpath,'gt_img', f'test_gt_{args.cam}_{i:05d}_mc{args.move_cam}.jpg'), gt)
                pred = pred[0]
                os.makedirs(os.path.join(outpath, str(args.cam)), exist_ok=True)
                cv2.imwrite(os.path.join(outpath, str(args.cam), f'{i:05d}.jpg'), pred)
                if not (i >= 90 and i <= 120):
                    continue
                P.append(psnr(pred, gt))
                L.append(lpips(pred, gt))
                S.append(ssim(pred, gt))
            writer.batch(iternum, itemnum + torch.arange(b), **data, **output)

            endtime = time.time()
            ips = 1. / (endtime - starttime)
            print("{:4} / {:4} ({:.4f} iter/sec)".format(itemnum, len(dataset), ips), end="\n")
            starttime = time.time()

            iternum += 1
            itemnum += b

    # cleanup
    writer.finalize(args.cam)
    metrics_dict = {"lpips": np.mean(L), "ssim": np.mean(S), "psnr": np.mean(P)}
    np.save(os.path.join(outpath, f"metrics_oneframe_{args.cam}.npy"), metrics_dict)
    outstring = (f'psnr:\t {np.mean(P)} \n'
                 f'lpips:\t {np.mean(L)} \n'
                 f'ssim:\t {np.mean(S)} \n')
                #  f'masked psnr:\t {np.mean(MP)} \n')
                #  f'hair psnr:\t {np.mean(MP_H)} \n'
                #  f'face psnr:\t {np.mean(MP_F)} \n')
    with open(f'{outpath}/eval_{args.experconfig[:-3]}_{args.subject}_{args.mode}_mc{args.move_cam}_c{args.cam}', 'w') as f:
        f.write(outstring)
    # imageio.mimwrite(os.path.join(outpath, f'nv_{args.subject}_demo.mp4'), rgbs2write, quality=8, fps=30)