"""
 Copyright 2022 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import torch
import torchvision
from torchvision import transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import argparse
import numpy as np
from collections import defaultdict
from functools import partial
import clip
import glob

import lpips
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class FolderDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.imgs = glob.glob(os.path.join(self.image_dir, '*.png')) + glob.glob(os.path.join(self.image_dir, '*.jpg'))

        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        image_name = self.imgs[index]
        image = Image.open(image_name)
        if self.transform:
            image = self.transform(image)
        return image

class_subject_dict = {
    'backpack': 'backpack', 'backpack_dog': 'backpack', 'bear_plushie': 'stuffed animal',
    'berry_bowl': 'bowl', 'can': 'can', 'candle': 'candle', 'cat': 'cat', 'cat2': 'cat',
    'clock': 'clock', 'colorful_sneaker': 'sneaker', 'dog': 'dog', 'dog2': 'dog',
    'dog3': 'dog', 'dog5': 'dog', 'dog6': 'dog', 'dog7': 'dog', 'dog8': 'dog',
    'duck_toy': 'toy', 'fancy_boot': 'boot', 'grey_sloth_plushie': 'stuffed animal',
    'monster_toy': 'toy', 'pink_sunglasses': 'sunglasses', 'poop_emoji': 'toy',
    'rc_car': 'toy', 'red_cartoon': 'cartoon', 'robot_toy': 'toy',
    'shiny_sneaker': 'sneaker', 'teapot': 'teapot', 'vase': 'vase', 'wolf_plushie': 'stuffed animal'
}

included_subjects = ['candle', 'cat', 'cat2', 'clock', 'dog', 'dog6', 'dog7', 'duck_toy',
'fancy_boot', 'grey_sloth_plushie', 'poop_emoji', 'red_cartoon', 'shiny_sneaker', 'teapot', 'vase']

def eval(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    loss_fn = lpips.LPIPS(net='alex').cuda()

    overall_mean_dist = defaultdict(list)

    all_folders = sorted(glob.glob('{}/*/'.format(args.data_path)))
    n_classes = 8

    for data_folder_path in all_folders:
        subject_name = data_folder_path.split('/')[-2]
        if args.filter_subjects and subject_name not in included_subjects:
            continue
        print('Computing for', data_folder_path)

        val_transform = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        
        dataset_val = torchvision.datasets.ImageFolder(data_folder_path, transform=val_transform)
        val_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            shuffle=False
        )

        label_list = []
        img_paths = dataset_val.imgs
        
        method_mean_dist = defaultdict(list)
        for idx, (inp, label) in enumerate(val_loader):
            inp = inp.cuda(non_blocking=True)
            label_cpu = label.clone()
            label = label.cuda(non_blocking=True)
        
            # if label == 0:
            #     continue
            method_1 = img_paths[idx][0].split('/')[-3]
            prompt_1 = img_paths[idx][0].split('/')[-2]
            img_number_1 = img_paths[idx][0].split('/')[-1]

            real_data_folder_path = os.path.join(args.data_path, '{}/{}/{}'.format(subject_name, method_1, prompt_1))
            real_dataset = FolderDataset(real_data_folder_path, transform=val_transform)
            real_loader = torch.utils.data.DataLoader(
                real_dataset,
                batch_size=1,
                num_workers=1,
                pin_memory=True,
                shuffle=False
            )
            real_img_paths = real_dataset.imgs

            for idy, (real_inp) in enumerate(real_loader):
                # print(idx, idy)
                # prompt_2 = real_img_paths[idy][0].split('/')[-2]
                img_number_2 = real_img_paths[idy][0].split('/')[-1]
                # if real_label != label_cpu:
                #     continue
                if img_number_1 == img_number_2:
                    continue
                real_inp = real_inp.cuda(non_blocking=True)
                with torch.no_grad():
                    dist = loss_fn(inp, real_inp).cpu().numpy().item()
                    overall_mean_dist[label.cpu().numpy().item()].append(dist)
                    method_mean_dist[label.cpu().numpy().item()].append(dist)

                label_list.append(label.cpu().numpy())
        
        print('Mean distances')
        for label in range(n_classes):
            if label in method_mean_dist:
                print(label, ' - dist =', sum(method_mean_dist[label]) / len(method_mean_dist[label]))

        label_npy = np.array(label_list).squeeze()

    print('Overall mean distances')
    for label in range(n_classes):
        if label in overall_mean_dist:
            print(label, ' - dist =', sum(overall_mean_dist[label]) / len(overall_mean_dist[label]))

    print('End')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Compute diversity')
    parser.add_argument('--data_path', default='data/dataset_output_final', type=str)
    parser.add_argument('--data_path_real', default='data/train_data/dataset', type=str)
    parser.add_argument('--filter_subjects', action='store_true')
    args = parser.parse_args()
    eval(args)