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

# This script renames dreambooth subject folders to have "token"
# instead of "monadikos_classname"

all_folders = glob.glob('data/dataset_output/dreambooth/*/')

# Two step because of candle/can

class_name_list = ['monadikos_backpack', 'monadikos_stuffed_animal', 'monadikos_bowl', 'monadikos_candle', 
'monadikos_cat', 'monadikos_clock', 'monadikos_sneaker', 'monadikos_dog', 'monadikos_toy', 'monadikos_boot', 
'monadikos_sunglasses', 'monadikos_cartoon', 'monadikos_teapot', 'monadikos_vase', 'monadikos_object']

for folder_path in all_folders:
    inner_folders = glob.glob(os.path.join(folder_path, '*/'))
    for inner_folder in inner_folders:
        for class_name in class_name_list:
            if class_name in inner_folder:
                orig = inner_folder
                tgt = inner_folder.replace(class_name, 'token')
                print(orig, tgt)
                shutil.move(orig, tgt)

class_name_list = ['monadikos_can']

for folder_path in all_folders:
    inner_folders = glob.glob(os.path.join(folder_path, '*/'))
    for inner_folder in inner_folders:
        for class_name in class_name_list:
            if class_name in inner_folder:
                orig = inner_folder
                tgt = inner_folder.replace(class_name, 'token')
                print(orig, tgt)
                shutil.move(orig, tgt)