# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Example configuration file."""
import os
from glob import glob
import data.multiview_renderme360 as datamodel

import torch
import torch.nn as nn

import numpy as np

holdoutcams = []
holdoutseg = []

# default setting (setting 2): 38 train views, 22 test views
test_view = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 8, 12, 36, 44, 52, 54, 56, 58, 51]
'''
# the following 2 settings are for camera split ablation experiments
# setting 1: 56 train views, 4 test views
test_view = [1,13,25,40]

# setting 2: 26 train views, 34 test views
test_view = [1, 3, 4, 5, 7, 9, 11, 13, 15, 16, 17, 19, 21, 23, 25, 27, 28, 29, 31, 33, 35, 37, 39, 40, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59]
'''


def get_dataset(
        camerafilter=lambda x: type(x) is int and x not in holdoutcams,
        segmentfilter=lambda x: x not in holdoutseg,
        keyfilter=[],
        maxframes=-1,
        subsampletype=None,
        downsample=4,
        **kwargs):
    """
    Parameters
    -----
    camerafilter : Callable[[str], bool]
        Function to determine cameras to use in dataset (camerafilter(cam) ->
        True to use cam, False to not use cam).
    segmentfilter : Callable[[str], bool]
        Function to determine segments to use in dataset (segmentfilter(seg) ->
        True to use seg, False to not use seg).
    keyfilter : list
        List of data to load from dataset. See Dataset class (e.g.,
        data.multiviewvideo) for a list of valid keys.
    maxframes : int
        Maximum number of frames to load.
    subsampletype : Optional[str]
        Type of subsampling to perform on camera images. Note the PyTorch
        module does the actual subsampling, the dataset class returns an array
        of pixel coordinates.
    downsample : int
        Downsampling factor of input images.
    """
    return datamodel.Dataset(
        views=60,
        krtpath=kwargs['krtpath'],
        geomdir=kwargs['geomdir'],
        imagepath=kwargs['imagepath'],
        bgpath=kwargs['bgpath'],
        flamepath=kwargs['flamepath'],
        averagepath=kwargs['averagepath'],
        maskpath=kwargs['maskpath'],
        texpath=kwargs['texpath'],
        returnbg=False,
        avgtexsize=256,
        baseposepath="",
        camerafilter=camerafilter,
        segmentfilter=segmentfilter,
        framelist=kwargs['framelist'],
        keyfilter=["camera", "modelmatrix", "modelmatrixinv", "pixelcoords", "image", "avgtex", "verts"] + keyfilter,
        maxframes=maxframes,
        subsampletype=subsampletype,
        subsamplesize=384,
        downsample=downsample,
        blacklevel=[3.8, 2.5, 4.0],
        maskbright=True,
        maskbrightbg=True)

def get_renderoptions():
    """Return dict of rendering options"""
    return dict(
            algo=0, # raymarcher can support multiple types of rendering, 0 is default
            chlast=True, # whether voxel grid has channels (RGBA) in the last dimension
            dt=1.0,
            # viewslab=True, # prim rgb show
            # viewaxes=True, # only a blue axis
            # colorprims=True # add prim a special color
            ) # stepsize

def get_autoencoder(dataset, renderoptions):
    """Return an autoencoder instance"""
    import torch
    import torch.nn as nn
    import models.volumetric as aemodel
    import models.encoders.geotex as encoderlib
    import models.decoders.mvp as decoderlib
    import models.raymarchers.mvpraymarcher as raymarcherlib
    import models.colorcals.colorcal as colorcalib
    import models.bg.lap as bglib
    from utils import utils

    allcameras = dataset.get_allcameras()
    ncams = len(allcameras)
    width, height = next(iter(dataset.get_krt().values()))["size"]

    # per-camera color calibration
    colorcal = colorcalib.Colorcal(dataset.get_allcameras())

    # mesh topology
    objpath = "./configs/head_template_mesh.obj"
    v, vt, vi, vti = utils.load_obj(objpath)
    vt = np.array(vt, dtype=np.float32)
    vi = np.array(vi, dtype=np.int32)
    vti = np.array(vti, dtype=np.int32)
    idxim, tidxim, barim = utils.gentritex(v, vt, vi, vti, 1024)
    idxim = torch.tensor(idxim).long()
    tidxim = torch.tensor(tidxim).long()
    barim = torch.tensor(barim)

    vertmean = torch.from_numpy(dataset.vertmean)
    vertstd = dataset.vertstd

    encoder = encoderlib.Encoder(texsize=256, vertsize=15069)
    print("encoder:", encoder)

    volradius = 512.
    decoder = decoderlib.Decoder(
        vt, vertmean, vertstd,
        idxim, tidxim, barim,
        volradius=volradius,
        dectype="slab2d",
        nprims=256,
        primsize=(32, 32, 32),
        #nprims=4096,
        #primsize=(16, 16, 8),
        motiontype="deconv",
        sharedrgba=False,
        elr=True,
        postrainstart=100,
        renderoptions=renderoptions)

    print("decoder:", decoder)

    raymarcher = raymarcherlib.Raymarcher(volradius=volradius)
    print("raymarcher:", raymarcher)

    bgmodel = bglib.BGModel(width, height, allcameras, trainstart=0, startlevel=2, buftop=True)
    # initialize bg
    for i, cam in enumerate(dataset.cameras):
        if cam in dataset.bg:
            bgmodel.lap.pyr[-1].image[0, :, allcameras.index(cam), :,  :].data[:] = (
                    torch.from_numpy(dataset.bg[cam][0]).to("cuda"))
    print("bgmodel:", bgmodel)

    ae = aemodel.Autoencoder(
        dataset,
        encoder,
        decoder,
        raymarcher,
        colorcal,
        volradius,
        bgmodel,
        encoderinputs=["verts", "avgtex"],
        topology={"vt": vt, "vi": vi, "vti": vti},
        imagemean=200.,
        imagestd=25.)

    print("encoder params:", sum(p.numel() for p in ae.encoder.parameters() if p.requires_grad))
    print("decoder params:", sum(p.numel() for p in ae.decoder.parameters() if p.requires_grad))
    print("colorcal params:", sum(p.numel() for p in ae.colorcal.parameters() if p.requires_grad))
    print("bgmodel params:", sum(p.numel() for p in ae.bgmodel.parameters() if p.requires_grad))
    print("total params:", sum(p.numel() for p in ae.parameters() if p.requires_grad))

    return ae

# profiles
class Train():
    """Profile for training models."""
    batchsize=6
    def __init__(self, maxiter=5000000, **kwargs):
        self.maxiter = maxiter
        self.otherargs = kwargs
        self.train_view = [i for i in range(60) if i not in test_view]
        print(f'train_view: {self.train_view}')
    def get_autoencoder(self, dataset):
        """Returns a PyTorch Module that accepts inputs and produces a dict
        of output values. One of those output values should be 'losses', another
        dict with each of the separate losses. See models.volumetric for an example."""
        return get_autoencoder(dataset, **self.get_ae_args())
    def get_outputlist(self):
        """A dict that is passed to the autoencoder telling it what values
        to compute (e.g., irgbrec for the rgb image reconstruction)."""
        return []
    def get_ae_args(self):
        """Any non-data arguments passed to the autoencoder's forward method."""
        return dict(renderoptions={**get_renderoptions(), **self.otherargs})
    def get_dataset(self, **kwargs):
        """A Dataset class that returns data for the autoencoder"""
        camerafilter = lambda x: type(x) is int and x in self.train_view
        return get_dataset(camerafilter=camerafilter,subsampletype="stratified", **kwargs)
    def get_optimizer(self, ae):
        """The optimizer used to train the autoencoder parameters."""
        import itertools
        import torch.optim
        lr = 0.002
        aeparams = itertools.chain(
            [{"params": x} for k, x in ae.encoder.named_parameters()],
            [{"params": x} for k, x in ae.decoder.named_parameters()],
            [{"params": x} for x in ae.bgmodel.parameters()],
            )
        return torch.optim.Adam(aeparams, lr=lr, betas=(0.9, 0.999))
    def get_loss_weights(self):
        return {"irgbmse": 1.0, "vertmse": 0.1, "kldiv": 0.001, "primvolsum": 0.01}

class ProgressWriter():
    def batch(self, iternum, itemnum, **kwargs):
        import numpy as np
        from PIL import Image
        rows = []
        row = []
        for i in range(kwargs["image"].size(0)):
            row.append(
                np.concatenate((
                    kwargs["irgbrec"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2],
                    kwargs["image"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2]), axis=1))
            if len(row) == 4:
                rows.append(np.concatenate(row, axis=1))
                row = []
        if len(row) > 0:
            rows.append(np.concatenate([row[i] if i < len(row) else row[0]*0. for i in range(maxcols)], axis=1))
        imgout = np.concatenate(rows, axis=0)
        outpath = os.path.join(kwargs['outpath'], 'images')
        os.makedirs(outpath, exist_ok=True)
        Image.fromarray(np.clip(imgout, 0, 255).astype(np.uint8)).save(
                os.path.join(outpath, "prog_{:06}.jpg".format(iternum)))
    def finalize(self):
        pass

class Progress():
    """Profile for writing out progress images during training."""
    batchsize=4
    def get_outputlist(self): return ["irgbrec"]
    def get_ae_args(self): return dict(renderoptions=get_renderoptions())
    def get_dataset(self, **kwargs): return get_dataset(**kwargs)
    def get_writer(self, **kwargs): return ProgressWriter()

class Eval():
    """Profile for evaluating models."""
    def __init__(self, outfilename=None, outfilesuffix=None,
            cam=None, camdist=768., camperiod=512, camrevs=0.25,
            segments=["S23_When_she_awoke_she_was_the_ship."],
            maxframes=-1,
            keyfilter=[],
            **kwargs):
        self.outfilename = outfilename
        self.outfilesuffix = outfilesuffix
        self.cam = cam
        self.camdist = camdist
        self.camperiod = camperiod
        self.camrevs = camrevs
        self.segments = segments
        self.segmentfilter = lambda x: True if segments == "all" else x in segments
        self.maxframes = maxframes
        self.keyfilter = keyfilter
        self.otherargs = kwargs
    def get_autoencoder(self, dataset): return get_autoencoder(dataset, **self.get_ae_args())
    def get_outputlist(self): return [k[0] if isinstance(k, tuple) else k for k in self.keyfilter]
    def get_ae_args(self): return dict(renderoptions={**get_renderoptions(),
        **self.otherargs})
    def get_dataset(self, **kwargs):
        import data.utils
        import data.camrender as cameralib
        if self.cam == "all":
            camerafilter = lambda x: type(x) is int
        elif self.cam == "holdout":
            camerafilter = lambda x: x in holdoutcams
        elif self.cam == "testview":
            camerafilter = lambda x: x in test_view
        else:
            camerafilter = lambda x: x == self.cam
        print(f"test view: {test_view}")
        dataset = get_dataset(camerafilter=camerafilter,
                segmentfilter=self.segmentfilter,
                keyfilter=[k[0] if isinstance(k, tuple) else k for k in self.keyfilter],
                maxframes=self.maxframes,
                **{k: v for k, v in self.otherargs.items() if k in get_dataset.__code__.co_varnames},
                **kwargs,
                )
        if self.cam is None:
            camdataset = cameralib.Dataset(len(dataset),krtpath=kwargs['krtpath'], averagepath=kwargs['averagepath'])
            return data.utils.ColCatDataset(camdataset, dataset)
        else:
            return dataset
    def get_writer(self, **kwargs):
        import utils.videowriter as writerlib
        if self.outfilename is None:
            outfilename = (
                    "render_{}_{}".format("-".join([x[:4].replace('_', '') for x in self.segments]), self.cam) +
                    (self.outfilesuffix if self.outfilesuffix is not None else "") +
                    ".mp4")
        else:
            outfilename = self.outfilename
        return writerlib.Writer(
            os.path.join(kwargs['outpath'], outfilename),
            keyfilter=self.keyfilter,
            colcorrect=[1.35, 1.16, 1.5],
            bgcolor=[255., 255., 255.],
            **{k: v for k, v in self.otherargs.items() if k in ["cmap", "cmapscale", "colorbar"]})
