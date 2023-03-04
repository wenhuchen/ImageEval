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

included_subjects = ['candle', 'cat', 'cat2', 'clock', 'dog', 'dog6', 'dog7', 'duck_toy',
'fancy_boot', 'grey_sloth_plushie', 'poop_emoji', 'red_cartoon', 'shiny_sneaker', 'teapot', 'vase']

def eval(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == 'dino':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        val_transform = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    elif args.mode == 'clip':
        model, val_transform = clip.load("ViT-B/32", device=device)

    model.to(device)
    model.eval()

    overall_mean_dist = defaultdict(list)

    all_folders = sorted(glob.glob('{}/*/'.format(args.data_path)))

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
        )

        features_list = []
        label_list = []
        features_dict_list = defaultdict(list)

        for inp, label in val_loader:
            inp = inp.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)

            with torch.no_grad():
                if args.mode == 'dino':
                    output = model(inp)
                elif args.mode == 'clip':
                    output = model.encode_image(inp)
                features_list.append(output.cpu().numpy())
                features_dict_list[label.cpu().numpy().item()].append(output.cpu().numpy())
            
            label_list.append(label.cpu().numpy())
        
        label_npy = np.array(label_list).squeeze()
        features_npy = np.array(features_list).squeeze()

        features_dict = defaultdict(partial(np.ndarray, 0))
        for label in range(len(np.unique(label_npy))):
            features_dict[label] = np.array(features_dict_list[label]).squeeze(0)

        df = pd.DataFrame()
        df["y"] = label_npy

        if args.tsne:
            X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=7).fit_transform(features_npy)
            df["comp-1"] = X_embedded[:,0]
            df["comp-2"] = X_embedded[:,1]

        n_classes = len(df['y'].unique())

        def compute_cosine_distance(image_features, image_features2):
            # normalized features
            image_features = image_features / np.linalg.norm(image_features, ord=2)
            image_features2 = image_features2 / np.linalg.norm(image_features2, ord=2)
            return np.dot(image_features, image_features2)
        
        def compute_l2_distance(image_features, image_features2):
            return np.linalg.norm(image_features - image_features2)

        if args.dist == 'cosine':
            compute_distance = compute_cosine_distance
        elif args.dist == 'l2':
            compute_distance = compute_l2_distance

        print(features_dict[0].shape, features_npy.shape)
        # Compute all pairwise distances between real samples and generated samples
        for idx in range(features_dict[0].shape[0]):
            dist_list = []
            for idy in range(features_npy.shape[0]):
                dist_list.append(compute_distance(features_npy[idy], features_dict[0][idx]))
            df['dist_{}'.format(idx)] = dist_list

        # print('Feature Distances')
        mean_feat_dist = defaultdict(list)
        for idx in range(features_dict[0].shape[0]):
            # print('Exemplar', idx)
            for label in range(n_classes):
                dist = df[df['y'] == label]['dist_{}'.format(idx)].mean()
                mean_feat_dist[label].append(dist)
                # print(label, ' - dist =', dist)

        print('Mean distances')
        for label in range(n_classes):
            print(label, ' - dist =', sum(mean_feat_dist[label]) / len(mean_feat_dist[label]))
            overall_mean_dist[label].append(sum(mean_feat_dist[label]) / len(mean_feat_dist[label]))

        plt.show()

    print('Overall mean distances')
    for label in range(n_classes):
        print(label, ' - dist =', sum(overall_mean_dist[label]) / len(overall_mean_dist[label]))

    print('End')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Subject fidelity evaluation')
    parser.add_argument('--data_path', default='data/dataset_output_final', type=str)
    parser.add_argument('--mode', default='dino', type=str)
    parser.add_argument('--dist', default='cosine', type=str)
    parser.add_argument('--filter_subjects', action='store_true')
    args = parser.parse_args()
    eval(args)