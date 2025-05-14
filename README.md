

# Diffusion-FFPE

[BIBM2024] [Leveraging Pre-trained Models for FF-to-FFPE Histopathological Image Translation](https://ieeexplore.ieee.org/abstract/document/10822745/)



## Introduction

The two primary types of Hematoxylin and Eosin (H&E) slides in histopathology are Formalin-Fixed Paraffin-Embedded (FFPE) and Fresh Frozen (FF). FFPE slides offer high quality histopathological images but require a labor-intensive acquisition process. In contrast, FF slides can be prepared quickly, but the image quality is relatively poor. Our task is to translate FF images into FFPE style, thereby improving the image quality for diagnostic purposes. In this paper, we propose Diffusion-FFPE, a method for FF-to-FFPE histopathological image translation using a pre-trained diffusion model. Specifically, we utilize a one-step diffusion model as the generator, which we fine-tune using LoRA adapters within an adversarial learning framework. To enable the model to effectively capture both global structural patterns and local details, we introduce a multi-scale feature fusion module that leverages two VAE encoders to extract features at different image resolutions, performing feature fusion before inputting them into the UNet. Additionally, a pre-trained vision-language model for histopathology serves as the backbone for the discriminator, enhancing model performance. Our FF-to-FFPE translation experiments on the TCGA-NSCLC dataset demonstrate that the proposed approach outperforms existing methods.



## Architecture

![](./assets/Overview.png)



## Prerequisites

- Python >= 3.8

- NVIDIA GPU + CUDA CuDNN

- Torch 2.0.1 + Torchvision 0.15.2

  

## Getting started

- Clone this repository:

```bash
git clone git@github.com:QilaiZhang/Diffusion-FFPE.git
cd Diffusion-FFPE
pip install -r requirements.txt
```

- Install [CONCH](https://huggingface.co/MahmoodLab/CONCH) and place it in ./checkpoints
- Prepare FF and FFPE datasets following [AI-FFPE](https://github.com/DeepMIALab/AI-FFPE), and split the datasets into training, validation, and test sets.



## Training

- Train the Diffusion-FFPE model:

```bash
python train.py --train_source_folder [TRAIN_FF_FOLDER] --train_target_folder [TRAIN_FFPE_FOLDER] --valid_source_folder [VALID_FF_FOLDER] --valid_target_folder [VALID_FFPE_FOLDER]
```

- Resume training from latest checkpoints:

```bash
python train.py --train_source_folder [TRAIN_FF_FOLDER] --train_target_folder [TRAIN_FFPE_FOLDER] --valid_source_folder [VALID_FF_FOLDER] --valid_target_folder [VALID_FFPE_FOLDER] --ckpt_path [CHECKPOINTS_FOLDER] --resume
```



## Inference

- Download pre-trained [checkpoints](https://cloud.tsinghua.edu.cn/d/2892cd01a94e49519068/) and place it in ./checkpoints.
- Generate FFPE images from FF test dataset:

```bash
python inference.py --img_path [TEST_FF_FOLDER] --pretrained_path ./checkpoints/model.pkl
```



## Evaluation

- Compute statistics for FFPE test datasets:

```bash
python eval.py --data_path [TEST_FFPE_FOLDER] --ref_path [TEST_STATISTICS_PATH] --save-stats
```

- Compute FID and KID:

```bash
python eval.py --data_path [GENERATE_FFPE_FOLDER] --ref_path [TEST_STATISTICS_PATH] --fid --kid
```



## Visualization

![](./assets/visual.png)



## References

If our work is useful for your research, please consider citing:

```
@inproceedings{zhang2024leveraging,
  title={Leveraging Pre-trained Models for FF-to-FFPE Histopathological Image Translation},
  author={Zhang, Qilai and Li, Jiawen and Liao, Peiran and Hu, Jiali and Guan, Tian and Han, Anjia and He, Yonghong},
  booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={3993--3996},
  year={2024},
  organization={IEEE}
}
```



## Acknowledgments

Our code is developed based on [img2img-turbo](https://github.com/GaParmar/img2img-turbo), [vision-aided-gan](https://github.com/nupurkmr9/vision-aided-gan), [CONCH](https://github.com/mahmoodlab/CONCH) and [clean-fid](https://github.com/GaParmar/clean-fid). Thanks for their awesome work.