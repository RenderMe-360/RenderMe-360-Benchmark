# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import data.renderme360 as datamodel
from pdb import set_trace as st
def get_dataset(datadir, annotdir, subject, ws_factor, subsampletype=None, move_cam=0, mode='sequence',cam=None):
    # default setting (setting 2): 38 train views, 22 test views
    train_views = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 53,
                   55, 57, 59, 4, 16, 28, 40, 0, 20, 24, 32, 48]
    test_views = [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 8, 12, 36, 44, 52, 54, 56, 58,
                  51]
    '''
    # the following 2 settings are for camera split ablation experiments
    # setting 1: 56 train views, 4 test views
    train_views = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, \
                        32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    test_views = [1,13,25,40]
    
    # setting 2: 26 train views, 34 test views
    train_views = [0, 2, 6, 8, 10, 12, 14, 18, 20, 22, 24, 26, 30, 32, 34, 36, 38, 42, 44, 46, 48, 50, 52, 54, 56, 58]
    test_views = [1, 3, 4, 5, 7, 9, 11, 13, 15, 16, 17, 19, 21, 23, 25, 27, 28, 29, 31, 33, 35, 37, 39, 40, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59]
    '''

    return datamodel.Dataset(
        datadir,
        annotdir,
        subject,
        keyfilter=["bg", "fixedcamimage", "camera", "image", "pixelcoords"],
        ws_factor=ws_factor,
        fixedcammean=100.,
        fixedcamstd=25.,
        imagemean=100.,
        imagestd=25.,
        subsampletype=subsampletype,
        subsamplesize=128,
        move_cam=move_cam,
        mode=mode,
        cam=cam,
        train_views = train_views,
        test_views = test_views)

def get_autoencoder(dataset):
    import models.neurvol_ghr as aemodel
    import models.encoders.mvconv as encoderlib
    import models.decoders.voxel1 as decoderlib
    import models.volsamplers.warpvoxel as volsamplerlib
    import models.colorcals.colorcal1 as colorcalib
    return aemodel.Autoencoder(
        dataset,
        encoderlib.Encoder(dataset.ninput),
        decoderlib.Decoder(globalwarp=True),
        volsamplerlib.VolSampler(),
        colorcalib.Colorcal([str(key) for key in dataset.get_allcameras()]),
        4. / 256,
        estimatebg=True)

### profiles
# A profile is instantiated by the training or evaluation scripts
# and controls how the dataset and autoencoder is created
class Train():
    batchsize=8
    maxiter=500000
    def get_autoencoder(self, dataset): return get_autoencoder(dataset)
    def get_dataset(self, datadir, annotdir, subject, ws_factor, mode='sequence'): 
        return get_dataset(datadir, annotdir, subject, ws_factor=ws_factor, subsampletype="random2", mode=mode)
    def get_optimizer(self, ae):
        import itertools
        import torch.optim
        lr = 0.0001
        aeparams = itertools.chain(
            [{"params": x} for x in ae.encoder.parameters()],
            [{"params": x} for x in ae.decoder.parameters()],
            [{"params": x} for x in ae.colorcal.parameters()])
        return torch.optim.Adam(aeparams, lr=lr, betas=(0.9, 0.999))
    def get_loss_weights(self):
        return {"irgbmse": 1.0, "kldiv": 0.001, "alphapr": 0.01, "tvl1": 0.01}

class ProgressWriter():
    def batch(self, iternum, itemnum, **kwargs):
        import numpy as np
        from PIL import Image
        rows = []
        row = []
        batchsize = kwargs["image"].size(0)
        if batchsize > 1:
            batchsize = int(np.sqrt(batchsize)) ** 2
            for i in range(batchsize):
                row.append(
                    np.concatenate((
                            kwargs["irgbrec"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2],
                            kwargs["image"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2]), axis=1)
                        )
                if len(row) == int(np.sqrt(batchsize)):
                    rows.append(np.concatenate(row, axis=1))
                    row = []
            imgout = np.concatenate(rows, axis=0)
        else:
            imgout = np.concatenate((
                        kwargs["irgbrec"][0].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2],
                        kwargs["image"][0].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2]), axis=1)
        
        # outpath = os.path.dirname(__file__)
        outpath = kwargs["logdir"]
        os.makedirs(os.path.join(outpath, "images"), exist_ok=True)
        Image.fromarray(np.clip(imgout, 0, 255).astype(np.uint8)).save(os.path.join(outpath, "images", "prog_{:06}.jpg".format(iternum)))

    def finalize(self):
        pass

class Progress():
    """Write out diagnostic images during training."""
    batchsize=4
    def get_ae_args(self): return dict(outputlist=["irgbrec"])
    def get_dataset(self, datadir, annotdir, subject, ws_factor, mode='sequence'): 
        return get_dataset(datadir, annotdir, subject, ws_factor=ws_factor, mode=mode)
    def get_writer(self): return ProgressWriter()

class Render():
    """Render model with training camera or from novel viewpoints.
    
    e.g., python render.py {configpath} Render --maxframes 128"""
    def __init__(self, subject, showtarget=False, showdiff=False, viewtemplate=False):
        self.subject = subject
        self.showtarget = showtarget
        self.viewtemplate = viewtemplate
        self.showdiff = showdiff
    def get_autoencoder(self, dataset): return get_autoencoder(dataset)
    def get_ae_args(self): return dict(outputlist=["irgbrec", "irgbsqerr", 'ialpharec'], viewtemplate=self.viewtemplate)
    def get_dataset(self, datadir, annotdir, subject, ws_factor, move_cam=0, mode='sequence', cam=None):
        return get_dataset(datadir, annotdir, subject, ws_factor, move_cam=move_cam, mode=mode, cam=cam)
    def get_writer(self, outdir=None):
        import eval.writers.videowriter as writerlib
        return writerlib.Writer(
            os.path.join(outdir if outdir is not None else 'logs', self.subject,
                "render{}.mp4".format(
                    "_template" if self.viewtemplate else "")),
            showtarget=self.showtarget,
            showdiff=self.showdiff,
            bgcolor=[255., 255., 255.])
