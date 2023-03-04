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

import glob
import os
import shutil

# This script deletes all images above the limit of 4

all_folders = glob.glob('data/dataset_output/dreambooth/*/')

for folder_path in all_folders:
    print(folder_path)
    inner_folders = glob.glob(os.path.join(folder_path, '*/'))
    for inner_folder in inner_folders:
        image_paths = sorted(glob.glob(os.path.join(inner_folder, '*.png')))
        for idx, img_path in enumerate(image_paths):
            if idx > 3:
                # delete
                print('remove', img_path)
                os.remove(img_path)
            else:
                # rename
                tgt_path = os.path.join(os.path.dirname(img_path), '{}.png'.format(idx+1))
                print(img_path, tgt_path)
                os.rename(img_path, tgt_path)