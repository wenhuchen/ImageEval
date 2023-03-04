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

from torch import autocast
from diffusers import StableDiffusionPipeline
import torch
import os

class_name_list = ['backpack', 'stuffed animal', 'bowl', 'can', 'candle', 'cat', 'clock', 'sneaker',
'dog', 'toy', 'boot', 'sunglasses', 'cartoon', 'teapot', 'vase']

class_token = '<class_token>'

prompt = 'a {}'.format(class_token)

n_samples = 500
model_id = 'CompVis/stable-diffusion-v1-4'
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda") 

for class_name in class_name_list:
    prompt_new = prompt.replace(class_token, class_name)
    print(prompt_new)
    for idx in range(n_samples):
        with autocast("cuda"):
            image = pipe(prompt_new, num_inference_steps=50, guidance_scale=7.5).images[0]
        if not os.path.exists('data/dataset_output/reg_images/{}'.format(class_name)):
            os.makedirs('data/dataset_output/reg_images/{}'.format(class_name))
        image.save("data/dataset_output/reg_images/{0}/{1}.png".format(class_name, idx))