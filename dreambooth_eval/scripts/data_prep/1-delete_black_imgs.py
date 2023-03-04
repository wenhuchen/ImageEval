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

from PIL import Image
import glob
import os

# This script deletes all black images generated when we hit the NSFW filter

all_imgs = glob.glob('data/dataset_output/dreambooth/*/*/*.png')

for img_path in all_imgs:
    img = Image.open(img_path)
    if not img.getbbox():
        print(img_path, 'is blank - deleting.')
        os.remove(img_path)