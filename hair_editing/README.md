# Hair Editing

This task refers to the setting of editing the hair attributes, given the source image and target text prompt.

## Subsets & Pre-processing

We use two subsets of Renderme360: normal cases, and hard cases with deformable accessories. You can find these two subsets in `./hari_editing/data/data_e0`. We follow the [face alignment pipeline](https://github.com/danielroich/PTI/blob/main/utils/align_data.py) of [PTI](https://github.com/danielroich/PTI) to do the pre-precessing. The aligned faces can be found in `./hari_editing/data/aligned_e0`.

## Inversion & Text-aware Editing

For the face image inversion part, we use the [e4e](https://github.com/omertov/encoder4editing), [ReStyle_e4e](https://github.com/yuval-alaluf/restyle-encoder), [PTI](https://github.com/danielroich/PTI) and [HyperStyle](https://github.com/yuval-alaluf/hyperstyle) following their official implementation. For the text-aware editing part, we use [StyleCLIP](https://github.com/orpatashnik/StyleCLIP.git) and [HairCLIP](https://github.com/wty-ustc/HairCLIP) following their official implementation. The hair color and hairstyles are defined in `haircolor_plus.txt` and `hairstyle_plus.txt`

## Reimplementation

We provide the testing pipelines of two combinations: HairCLIP(e4e) and StyleCLIP(PTI), that correspondingly have the highest CLIP-score and ID-score in our experiments.

### StyleCLIP(PTI)

We follow the official implementation of [PTI](https://github.com/danielroich/PTI) inversion and StyleCLIP editing from [here](https://github.com/yuval-alaluf/hyperstyle/blob/main/editing/styleclip/edit.py). As the bridge of inversion and editing, we borrow the scripts from [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) for converting the format of tuned checkpoints. We also provide the editing scripts in `./hyperstyle/editing/styleclip/editing_pti.py`. The overall pipeline is listed as follows:

1. Get the lantents and corresponding tuned generator weights following the instruction of [PTI](https://github.com/danielroich/PTI).
2. Convert the tuned generator weights to the desired format by using `export_weights.py` in [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch).
3. Put the provided `./hyperstyle/editing/styleclip/editing_pti.py` into the corresponding position of the [repo](https://github.com/yuval-alaluf/hyperstyle/blob/main/editing/styleclip/edit.py).
4. Modify the paths in the provided script `./hyperstyle/process_edit_pti_styleclip.py` and run it to generate the edited images.

### HairCLIP(e4e)

We follow the the official implementation of [HairCLIP](https://github.com/wty-ustc/HairCLIP).

## Citation

```
@article{tov2021designing,
  title={Designing an Encoder for StyleGAN Image Manipulation},
  author={Tov, Omer and Alaluf, Yuval and Nitzan, Yotam and Patashnik, Or and Cohen-Or, Daniel},
  journal={arXiv preprint arXiv:2102.02766},
  year={2021}
}
```

```
@InProceedings{alaluf2021restyle,
      author = {Alaluf, Yuval and Patashnik, Or and Cohen-Or, Daniel},
      title = {ReStyle: A Residual-Based StyleGAN Encoder via Iterative Refinement}, 
      month = {October},
      booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},  
      year = {2021}
}
```

```
@article{roich2021pivotal,
  title={Pivotal Tuning for Latent-based Editing of Real Images},
  author={Roich, Daniel and Mokady, Ron and Bermano, Amit H and Cohen-Or, Daniel},
  publisher = {Association for Computing Machinery},
  journal={ACM Trans. Graph.},
  year={2021}
}
```

```
@misc{alaluf2021hyperstyle,
      title={HyperStyle: StyleGAN Inversion with HyperNetworks for Real Image Editing}, 
      author={Yuval Alaluf and Omer Tov and Ron Mokady and Rinon Gal and Amit H. Bermano},
      year={2021},
      eprint={2111.15666},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```
@InProceedings{Patashnik_2021_ICCV,
    author    = {Patashnik, Or and Wu, Zongze and Shechtman, Eli and Cohen-Or, Daniel and Lischinski, Dani},
    title     = {StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {2085-2094}
}
```

```
@article{wei2022hairclip,
  title={Hairclip: Design your hair by text and reference image},
  author={Wei, Tianyi and Chen, Dongdong and Zhou, Wenbo and Liao, Jing and Tan, Zhentao and Yuan, Lu and Zhang, Weiming and Yu, Nenghai},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```
