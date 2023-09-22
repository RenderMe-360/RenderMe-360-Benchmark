# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch

from src import config

from src.model_multiface import KeypointNeRFLightningModule
from src.multiface_dataset import MultifaceDataset

if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    torch.autograd.set_detect_anomaly(True)
    
    # load configuration
    parser = config.create_parser()
    args = parser.parse_args(None)
    cfg = config.load_cfg(args.config)
    cfg['dataset']['data_root'] = args.data_root
    if args.out_dir is not None:
        cfg['out_dir'] = args.out_dir

    cfg['expname'] = cfg.get('expname', 'default')

    # create data loader
    # cfg['dataset']['pid'] = args.pid
    cfg['dataset']['test_split'] = args.renface_split
    test_dataset = MultifaceDataset.from_config(cfg['dataset'], 'val', cfg)
    data_loader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, pin_memory=True,
        num_workers=cfg['training'].get('val_num_workers', 0),
        collate_fn=KeypointNeRFLightningModule.collate_fn)

    # create model
    renderer = KeypointNeRFLightningModule.from_config(cfg, cfg.get('method', None)).cuda()
    renderer.load_ckpt(args.model_ckpt)
    renderer.test_multiface(data_loader, label=args.renface_split.replace('/', ':'))