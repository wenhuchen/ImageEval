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
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import argparse
import numpy as np
from collections import defaultdict
from functools import partial
import clip
import glob

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

    model, val_transform = clip.load("ViT-B/32", device=device)

    model.to(device)
    model.eval()

    overall_mean_dist = defaultdict(list)

    all_folders = sorted(glob.glob('{}/*/'.format(args.data_path)))
    n_classes = 6

    for data_folder_path in all_folders:
        print('Computing for', data_folder_path)
        subject_name = data_folder_path.split('/')[-2]
        if args.filter_subjects and subject_name not in included_subjects:
            continue
        dataset_val = torchvision.datasets.ImageFolder(data_folder_path, transform=val_transform)
        val_loader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            shuffle=False
        )

        features_list = []
        label_list = []
        features_dict_list = defaultdict(list)
        img_paths = dataset_val.imgs

        def compute_cosine_distance(image_features, image_features2):
            # normalized features
            image_features = image_features / np.linalg.norm(image_features, ord=2)
            image_features2 = image_features2 / np.linalg.norm(image_features2, ord=2)
            return np.dot(image_features, image_features2)

        method_mean_dist = defaultdict(list)
        for idx, (inp, label) in enumerate(val_loader):
            inp = inp.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            if label == 0:
                continue

            # Get prompt
            img_path = img_paths[idx][0]
            subject_name = img_path.split('/')[-4]
            class_name = class_subject_dict[subject_name]
            prompt = img_path.split('/')[-2]
            prompt = prompt.replace('_', ' ')
            prompt = prompt.replace('token', class_name)

            # Get clip text
            text = clip.tokenize([prompt]).to(device) 

            with torch.no_grad():
                output = model.encode_image(inp).cpu().numpy().astype(np.float32).squeeze(0)
                text_features = model.encode_text(text).cpu().numpy().astype(np.float32).squeeze(0)

                logits_per_image, logits_per_text = model(inp, text)
                dist = compute_cosine_distance(output, text_features)
                overall_mean_dist[label.cpu().numpy().item()].append(dist)
                method_mean_dist[label.cpu().numpy().item()].append(dist)

            label_list.append(label.cpu().numpy())
        
        print('Mean distances')
        for label in range(1,n_classes):
            if label in method_mean_dist:
                print(label, ' - dist =', sum(method_mean_dist[label]) / len(method_mean_dist[label]))

        label_npy = np.array(label_list).squeeze()

    print('Overall mean distances')
    for label in range(1,n_classes):
        if label in overall_mean_dist:
            print(label, ' - dist =', sum(overall_mean_dist[label]) / len(overall_mean_dist[label]))

    print('End')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Prompt fidelity evaluation')
    parser.add_argument('--data_path', default='data/dataset_output_final', type=str)
    parser.add_argument('--dist', default='cosine', type=str)
    parser.add_argument('--filter_subjects', action='store_true')
    args = parser.parse_args()
    eval(args)