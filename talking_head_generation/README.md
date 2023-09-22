# Talking Head generation

This task refers to the setting of audio-based talking head generation using NeRFs.

## Subsets

We use two subsets of Renderme360: five identities speaking English and five identities speaking Chinese Mandarin. You can find these two subsets [here](https://opendatalab.com/RenderMe-360/download).

## AD-NeRF & SSP-NeRF

We pre-processed, trained, and tested [AD-NeRF](https://github.com/YudongGuo/AD-NeRF) and [SSP-NeRF](https://github.com/alvinliu0/SSP-NeRF) following their official implementation. Thanks to [Xi&#39;an](https://alvinliu0.github.io/) for providing the training code of [SSP-NeRF](https://github.com/alvinliu0/SSP-NeRF). We also provided our pre-trained checkpoints of AD-NeRF [here](https://opendatalab.com/RenderMe-360/download).

## Reimplementation of AD-NeRF

### Testing

1. Git clone the original repository of [AD-NeRF](https://github.com/YudongGuo/AD-NeRF) and create the virtual environment following the instruction.
2. Download the [pre-trained checkpoints ](https://drive.google.com/drive/folders/1hWVgOexnuH_WfjDnAi6GXq_CsDnA9e5J?usp=sharing) and dataset `benchamrk_data/data_talkinghead.zip` and put them to `AD-NeRF/dataset/`.
3. Use the provided files in `./AD-NeRF/NeRFs` to replace the original files.
4. Run `python test_s123456.py $id`  to render the video.

### Training

1. For loading multiple video frames for the same identity, we provide files in  `./AD-NeRF/NeRFs/` and training scripts in `./AD-NeRF/`.

### Data Processing

1. We follow the data processing pipeline of AD-NeRF basically. We replace the landmark detection and face parsing parts following our own annotation pipeline. Please refer to `./data_util/process_data_renderme360.py` for details

## Citation

```
@inproceedings{guo2021adnerf,
  title={AD-NeRF: Audio Driven Neural Radiance Fields for Talking Head Synthesis},
  author={Yudong Guo and Keyu Chen and Sen Liang and Yongjin Liu and Hujun Bao and Juyong Zhang},
  booktitle={IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

```
@article{liu2022semantic,
  title = {Semantic-Aware Implicit Neural Audio-Driven Video Portrait Generation},
  author = {Liu, Xian and Xu, Yinghao and Wu, Qianyi and Zhou, Hang and Wu, Wayne and Zhou, Bolei},
  journal={arXiv preprint arXiv:2201.07786},
  year = {2022}
}
```
