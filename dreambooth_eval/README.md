# DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation

> Official *Stable Diffusion* Implementation

![teaser](docs/teaser_static.jpg)

### [project page](https://dreambooth.github.io/) | [arxiv](https://arxiv.org/abs/2208.12242)

This is the official Stable Diffusion implementation of the Google paper DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. This repository contains code to DreamBooth finetune Stable Diffusion models for subject or concept-driven generation. It also includes evaluation scripts in order to compute the quantitative metrics included in the paper (DINO, CLIP-Image, CLIP-Text). This implementation is built on the great [diffusers](https://github.com/huggingface/diffusers) library.

## Setup

This code was tested with Python 3.10.4, [Pytorch](https://pytorch.org/) 1.12.1 using pre-trained models through [huggingface / diffusers](https://github.com/huggingface/diffusers#readme).
Specifically, we implemented our method over [Stable Diffusion](https://huggingface.co/CompVis/stable-diffusion-v1-4).
Additional required packages are listed in the requirements file.

## Dataset

First download the dataset and move it to the following path: data/train_dataset/dataset

## How To DreamBooth Finetune

Use the dreambooth.py script to finetune a Stable Diffusion model on a subject or a concept. An example of how to run it on the *backpack* object in our dataset can be found below:
```
accelerate launch dreambooth.py   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" --use_auth_token   --train_data_dir=data/train_data/dataset/backpack    --unique_token="monadikos" --class_name="backpack"   --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=1   --max_train_steps=400   --learning_rate=5.0e-06 --scale_lr   --lr_scheduler="constant"   --lr_warmup_steps=0   --output_dir="data/models/dataset_output/dreambooth/backpack"
```
If you would like to generate your own unique token, you can do so by running the command below and copy-pasting your generated token:
```
python generate_unique_id.py --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4"
```

In order to use the prior preservation loss, we need to generate a folder of images of the same class as the subject.  In order to generate regularization images for our dataset, run the following:
```
python generate_regularization_images.py
```
You can modify the script to generate regularization images for the classes you're interested in.

Now, to run DreamBooth on the *backpack* object with the prior preservation loss run:
```
accelerate launch dreambooth.py   --pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" --use_auth_token   --train_data_dir=data/train_data/dataset/backpack  --reg_data_dir=data/dataset_output/reg_images/backpack    --prior_preservation_loss    --unique_token="monadikos" --class_name="backpack"   --resolution=512   --train_batch_size=1   --gradient_accumulation_steps=1   --max_train_steps=800   --learning_rate=5.0e-06 --scale_lr   --lr_scheduler="constant"   --lr_warmup_steps=0   --output_dir="data/models/dataset_output/dreambooth_reg/backpack"
```

If you would like to train models for all the subjects in our paper please run scripts/run/dataset_models.sh and scripts/run/dataset_models_reg.sh

Aditionally, if you want to finetune the text encoder as well, add the `--train_text_encoder` flag, and the `--text_encoder_learning_rate=1.0e-06` flag. Usually the text encoder is trained with a lower learning rate than the U-Net, but this hyperparameter can be tuned.

## How To Generate DreamBooth Images

To generate images, simply modify and run the inferency.py script:
```
python inference.py
```
You can modify the model path, the class name, the unique identifier and the prompts. For a script that generates results for our full dataset and full list of prompts please run inference_dataset.py

## How To Evaluate Results Quantitatively

First, in order to generate the folder structure for evaluation, run all scripts in the scripts/data_prep folder in order (from 1 to 5). These scripts delete black images (NSFW classification makes Stable Diffusion output black images), deletes images above the 4 count for each subject/prompt pair, normalizes folder names, filters prompts as explained in our paper (live subjects have one set of prompts, and objects have others) and finally generates the evaluation folder structure. Note that script 4-filter_prompts.py has to be copied and ran inside of the output folder.

In order to run subject fidelity evaluations using the DINO and CLIP-Image (CLIP-I) metrics, run:
```
python evaluate_all_dataset.py --data_path TARGET_PATH --mode dino --dist cosine
python evaluate_all_dataset --data_path TARGET_PATH --mode clip --dist cosine
```
To evaluate prompt fidelity using the CLIP-Text metric (CLIP-T) run:
```
python evaluate_all_dataset_clip_text.py --data_path TARGET_PATH --dist cosine
```

In order to evaluate output diversity (DIV) please run:
```
python evaluate_diversity_intra.py --data_path TARGET_PATH --data_path_real REAL_DATA_PATH --filter_subjects
```

## Academic Citation

If you use this work please cite:
```
@article{ruiz2022dreambooth,
  title={Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation},
  author={Ruiz, Nataniel and Li, Yuanzhen and Jampani, Varun and Pritch, Yael and Rubinstein, Michael and Aberman, Kfir},
  journal={arXiv preprint arXiv:2208.12242},
  year={2022}
}
```

## Disclaimer

This is not an officially supported Google product.