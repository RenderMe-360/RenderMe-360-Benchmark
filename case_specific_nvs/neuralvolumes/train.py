# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""Train an autoencoder."""
import argparse
import gc
import importlib
import importlib.util
import os
import sys
import time
# sys.dont_write_bytecode = True

import numpy as np

import torch
import torch.utils.data
from config_renderme360 import Train, Progress
from tqdm import tqdm, trange
torch.backends.cudnn.benchmark = True # gotta go fast!
from pdb import set_trace as st


def import_module(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description='Train an autoencoder')
    parser.add_argument('experconfig', type=str, help='experiment config file')
    parser.add_argument('--profile', type=str, default="Train", help='config profile')
    parser.add_argument('--datadir', type=str, default="./data/", help='directory for data')
    parser.add_argument('--annotdir', type=str, default="./PARAMS", help='directory for annotation')
    parser.add_argument('--outdir', type=str, default="logs", help='directory for annotation')
    parser.add_argument('--subject', type=str, default="emo", help='subject to train')
    parser.add_argument('--mode', type=str, default="exp", help='sequence or expression or speech')
    parser.add_argument('--devices', type=int, nargs='+', default=[0], help='devices')
    parser.add_argument('--resume', action='store_true', help='resume training')
    # NOTE: try to find the best world scale
    parser.add_argument('--ws_factor', type=float, nargs='+', default=[0.6], help='world scale = 2.0 / ws_factor')
    # NOTE: estimatebg
    parser.add_argument('--bg', action='store_true', help='estimate background')

    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg, type=eval)
    args = parser.parse_args()
    print(args.devices)
    outpath = os.path.join(args.outdir, args.subject)
    os.makedirs(outpath, exist_ok=True)

    print("Python", sys.version)
    print("PyTorch", torch.__version__)
    print(" ".join(sys.argv))
    print("Output path:", outpath)
    print("Resume", args.resume)
    # load config
    starttime = time.time()
    experconfig = import_module(args.experconfig, "config_renderme360")
    profile = getattr(experconfig, args.profile)(**{k: v for k, v in vars(args).items() if k not in parsed})
    progressprof = experconfig.Progress()
    print("Config loaded ({:.2f} s)".format(time.time() - starttime))
    # build dataset & testing dataset
    starttime = time.time()
    # NOTE: try to find the best world scale
    testdataset = progressprof.get_dataset(args.datadir, args.annotdir, args.subject, ws_factor=args.ws_factor[0], mode=args.mode)
    dataloader = torch.utils.data.DataLoader(testdataset, batch_size=progressprof.batchsize, shuffle=False, drop_last=True, num_workers=16)
    for testbatch in dataloader:
        break
    # NOTE: try to find the best world scale
    dataset = profile.get_dataset(args.datadir, args.annotdir, args.subject, ws_factor=args.ws_factor[0], mode=args.mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=profile.batchsize, shuffle=True, drop_last=True, num_workers=16)
    print("batchsize: ", profile.batchsize)
    print("Dataset instantiated ({:.2f} s)".format(time.time() - starttime))

    # data writer
    starttime = time.time()
    writer = progressprof.get_writer()
    print("Writer instantiated ({:.2f} s)".format(time.time() - starttime))

    # build autoencoder
    starttime = time.time()
    ae = profile.get_autoencoder(dataset).to("cuda")
    # ae = torch.nn.DataParallel(ae, device_ids=args.devices).to("cuda")
    if args.resume:
        try:
            ae.load_state_dict(torch.load("{}/aeparams.pt".format(outpath)), strict=False)
        except:
            ae = torch.load("{}/aeparams.pt".format(outpath))
    print("Autoencoder instantiated ({:.2f} s)".format(time.time() - starttime))

    # build optimizer
    starttime = time.time()
    aeoptim = profile.get_optimizer(ae)
    lossweights = profile.get_loss_weights()
    print("Optimizer instantiated ({:.2f} s)".format(time.time() - starttime))

    # train
    starttime = time.time()
    evalpoints = np.geomspace(1., profile.maxiter, 100).astype(np.int32)
    # iternum = log.iternum
    iternum = 0
    prevloss = np.inf

    for epoch in trange(1000):
        ae.train()
        for i, data in enumerate(tqdm(dataloader)):
            # forward
            _dict = {k: x.to("cuda") for k, x in data.items()}
            # _dict.pop('mask_hair')
            # _dict.pop('mask_face')

            '''
            to see the contents in 'data'.
            
            print(len(data["frameid"]))
            print(data.keys())
            exit(2)
            '''

            output = ae(iternum, lossweights.keys(), **_dict)

            # encoding = output['encoding']
            
            # np.save(f'animation_code_train/{args.subject}/nv_{args.subject}_{i:05d}.npy', encoding.detach().cpu().numpy())

            # compute final loss
            loss = sum([
                lossweights[k] * (torch.sum(v[0]) / torch.sum(v[1]) if isinstance(v, tuple) else torch.mean(v))
                for k, v in output["losses"].items()])

            # print current information
            if iternum % 25 == 0:
                print("[{}] Iteration {}: loss = {:.3f}, ".format(args.subject, iternum, float(loss.item())) +
                        ", ".join(["{} = {:.3f}".format(k,
                            float(torch.sum(v[0]) / torch.sum(v[1]) if isinstance(v, tuple) else torch.mean(v)))
                            for k, v in output["losses"].items()]))
            # update parameters
            aeoptim.zero_grad()
            loss.backward()
            aeoptim.step()

            # compute evaluation output
            if iternum in evalpoints:
                ae.eval()
                with torch.no_grad():
                    _dict = {k: x.to("cuda") for k, x in testbatch.items()}
                    # _dict.pop('mask_hair')
                    # _dict.pop('mask_face')
                    testoutput = ae(iternum, [], **_dict, **progressprof.get_ae_args())
                b = data["campos"].size(0)
                testoutput["logdir"] = outpath
                writer.batch(iternum, iternum * profile.batchsize + torch.arange(b), **testbatch, **testoutput)
                ae.train()

            # check for loss explosion
            if loss.item() > 20 * prevloss or not np.isfinite(loss.item()):
                tqdm.write("Unstable loss function; resetting")
                ae.load_state_dict(torch.load("{}/aeparams.pt".format(outpath)), strict=False)
                aeoptim = profile.get_optimizer(ae)
            prevloss = loss.item()
            iternum += 1

        # save intermediate results
        if epoch % 2 == 0:
            # from pdb import set_trace as st
            torch.save(ae.state_dict(), "{}/aeparams.pt".format(outpath))
            # torch.save(ae.state_dict(), "{}/{:06d}.pt".format(outpath, epoch))
            # torch.save(ae, "{}/aeparams.pt".format(outpath))
            #   torch.save(ae, "{}/{:06d}.pt".format(outpath, epoch))
        if iternum >= profile.maxiter:
            break

    # cleanup
    writer.finalize()
