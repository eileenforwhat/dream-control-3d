# Dream-Control-3D

A fork of [stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion) that adds ControlNet guidance to
enable conditioning of 3d generation on user scribble. Includes zero1to3 support with ControlNet.

# Final report
Download [final_report.pdf](https://github.com/eileenforwhat/dream-control-3d/blob/main/final_report.pdf) to read a summary of the project methodology and results. 

# Install

```bash
git clone https://github.com/eileenforwhat/dream-control-3d.git
cd dream-control-3d

bash setup.sh
```
Please see [https://github.com/ashawkey/stable-dreamfusion#install](https://github.com/ashawkey/stable-dreamfusion#install)
for more details and troubleshooting.

To use `controlnet-zero1to3` guidance, you need to download the pretrained checkpoint of [Zero-1-to-3](https://github.com/cvlab-columbia/zero123) under `./pretrained/`.
We use `105000.ckpt` by default, and it is hard-coded in `guidance/zero123_utils.py`.
```bash
cd pretrained
wget https://huggingface.co/cvlab/zero123-weights/resolve/main/105000.ckpt
```

# Usage

Please see [https://github.com/ashawkey/stable-dreamfusion#usage](https://github.com/ashawkey/stable-dreamfusion#usage)
for usage of StableDiffusion guidance. We will only include ControlNet guidance below.

```bash
# ControlNet guidance

# ControlNet + Zero1to3 guidance

# ControlNet + view_emphasis
```

For advanced tips and other developing stuff, check [Advanced Tips](./assets/advanced.md).

# Acknowledgement

This work is based on an increasing list of amazing research works and open-source projects, thanks a lot to all the authors for sharing!

* [DreamFusion: Text-to-3D using 2D Diffusion](https://dreamfusion3d.github.io/).
    ```
    @article{poole2022dreamfusion,
        author = {Poole, Ben and Jain, Ajay and Barron, Jonathan T. and Mildenhall, Ben},
        title = {DreamFusion: Text-to-3D using 2D Diffusion},
        journal = {arXiv},
        year = {2022},
    }
    ```

* [Magic3D: High-Resolution Text-to-3D Content Creation](https://research.nvidia.com/labs/dir/magic3d/):
   ```
   @inproceedings{lin2023magic3d,
      title={Magic3D: High-Resolution Text-to-3D Content Creation},
      author={Lin, Chen-Hsuan and Gao, Jun and Tang, Luming and Takikawa, Towaki and Zeng, Xiaohui and Huang, Xun and Kreis, Karsten and Fidler, Sanja and Liu, Ming-Yu and Lin, Tsung-Yi},
      booktitle={IEEE Conference on Computer Vision and Pattern Recognition ({CVPR})},
      year={2023}
    }
   ```

* [Zero-1-to-3: Zero-shot One Image to 3D Object](https://github.com/cvlab-columbia/zero123)
    ```
    @misc{liu2023zero1to3,
        title={Zero-1-to-3: Zero-shot One Image to 3D Object}, 
        author={Ruoshi Liu and Rundi Wu and Basile Van Hoorick and Pavel Tokmakov and Sergey Zakharov and Carl Vondrick},
        year={2023},
        eprint={2303.11328},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }
    ```

* [RealFusion: 360° Reconstruction of Any Object from a Single Image](https://github.com/lukemelas/realfusion)
    ```
    @inproceedings{melaskyriazi2023realfusion,
        author = {Melas-Kyriazi, Luke and Rupprecht, Christian and Laina, Iro and Vedaldi, Andrea},
        title = {RealFusion: 360 Reconstruction of Any Object from a Single Image},
        booktitle={CVPR}
        year = {2023},
        url = {https://arxiv.org/abs/2302.10663},
    }
    ```

* [Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation](https://fantasia3d.github.io/)
    ```
    @article{chen2023fantasia3d,
        title={Fantasia3D: Disentangling Geometry and Appearance for High-quality Text-to-3D Content Creation},
        author={Rui Chen and Yongwei Chen and Ningxin Jiao and Kui Jia},
        journal={arXiv preprint arXiv:2303.13873},
        year={2023}
    }
    ```

* [Stable Diffusion](https://github.com/CompVis/stable-diffusion) and the [diffusers](https://github.com/huggingface/diffusers) library.

    ```
    @misc{rombach2021highresolution,
        title={High-Resolution Image Synthesis with Latent Diffusion Models},
        author={Robin Rombach and Andreas Blattmann and Dominik Lorenz and Patrick Esser and Björn Ommer},
        year={2021},
        eprint={2112.10752},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }

    @misc{von-platen-etal-2022-diffusers,
        author = {Patrick von Platen and Suraj Patil and Anton Lozhkov and Pedro Cuenca and Nathan Lambert and Kashif Rasul and Mishig Davaadorj and Thomas Wolf},
        title = {Diffusers: State-of-the-art diffusion models},
        year = {2022},
        publisher = {GitHub},
        journal = {GitHub repository},
        howpublished = {\url{https://github.com/huggingface/diffusers}}
    }
    ```
