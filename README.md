# RenderMe-360-Benchmark

This is the Benchmark PyTorch implementation of the paper *"[RenderMe-360: A Large Digital Asset Library and Benchmarks Towards High-fidelity Head Avatars](https://arxiv.org/abs/2305.13353)"*

> 
>
> **Abstract:** *Synthesizing high-fidelity head avatars is a central problem for many applications on AR, VR, and Metaverse. While head avatar synthesis algorithms have advanced rapidly, the best ones still face great obstacles in real-world scenarios. One of the vital causes is the inadequate datasets  -- 1) current public datasets can only support researchers to explore high-fidelity head avatars in one or two task directions, such as viewpoint, head pose, hairstyle, or facial expression; 2) these datasets usually contain digital head assets with limited data volume, and narrow distribution over different attributes, such as expressions, ages, and accessories. In this paper, we present *RenderMe-360*, a comprehensive 4D human head dataset to drive advance in head avatar algorithms across different scenarios. RenderMe-360 contains massive data assets, with 250+ million complete head frames and over 800k video sequences from 500 different identities captured by synchronized HD multi-view cameras at 30 fps. It is a large-scale digital library for head avatars with three key attributes: 1) High Fidelity: all subjects are captured by 60 synchronized, high-resolution 2K cameras to collect their portrait data in 360 degrees. 2) High Diversity: The collected subjects vary from different ages, eras, ethnicity, and cultures, providing abundant materials with distinctive styles in appearance and geometry. Moreover, each subject is asked to perform various dynamic motions, such as expressions and head rotations, which further extend the richness of assets. 3) Rich Annotations: the dataset provides annotations with different granularities: cameras' parameters, background matting, scan, 2D as well as 3D facial landmarks, FLAME fitting labeled by semi-auto annotation, and text description. Based on the dataset, we build a comprehensive benchmark for head avatar research, with 16 state-of-the-art methods performed on five main tasks: novel view synthesis, novel expression synthesis, hair rendering, hair editing, and talking head generation. Our experiments uncover the strengths and weaknesses of state-of-the-art methods, showing that extra efforts are needed for them to perform in such diverse scenarios. RenderMe-360 opens the door for future exploration in modern head avatars. All of the data, code, and models will be publicly available at https://renderme-360.github.io/.* <br>


|         Benchmark         |                                      Method                                      | Reimplementation                                                                                                           |
| :------------------------: | :------------------------------------------------------------------------------: | -------------------------------------------------------------------------------------------------------------------------- |
|     Case-specific NVS     |                instant-ngp <br> NeuS <br> MVP <br> NV                | [case_specific_nvs/README.md](case_specific_nvs/README.md)                                                                    |
|     Generalizable NVS     |                 IBRNet <br> KeypointNerf <br> VisionNerf                 | [generalizable_nvs/README.md](generalizable_nvs/README.md) <br> [cross_dataset_eval/README.md](cross_dataset_eval/README.md) |
| Novel Expression Synthesis |                 NerFace <br> IM Avatar <br> Point-Avatar                 | [novel_expression_synthesis/README.md](novel_expression_synthesis/README.md)                                                  |
|       Hair Rendering       | instant-ngp <br> NeuS <br> MVP <br> NV <br> NSFF <br> NRNerf | [hair_rendering/README.md](hair_rendering/README.md)                                                                          |
|       Hair Editting       |                          StyleCLIP <br> HairCLIP                        | [hair_editing/README.md](hair_editing/README.md)                                                                              |
|  Talking Head Generation  |                            AD-Nerf <br> SSP-Nerf                            | [talking_head/README.md](talking_head/README.md)                                                                              |

We provide the train/test data for the whole benchmark in a unified path. Please click [here](https://opendatalab.org.cn/RenderMe-360/download) to download our benchmark data.