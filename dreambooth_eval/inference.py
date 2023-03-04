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

model_name = 'backpack'
model_id = 'data/models/dataset_output/dreambooth/{}'.format(model_name)
class_name = 'backpack'

unique_token = 'monadikos'
class_token = '<class_token>'

prompt_list = [
'a {0} {1} in the snow'.format(unique_token, class_token),
'a {0} {1} on top of a dirt road'.format(unique_token, class_token),
'a {0} {1} on top of a white rug'.format(unique_token, class_token),
'a white {0} {1}'.format(unique_token, class_token)
]

n_samples = 6

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda") 
for prompt in prompt_list:
    prompt_new = prompt.replace(class_token, class_name)
    output_string = prompt_new.replace(' ', '_')
    print(prompt_new)
    for idx in range(n_samples):
        with autocast("cuda"):
            image = pipe(prompt_new, num_inference_steps=50, guidance_scale=7.5).images[0]
        if not os.path.exists('data/output/{}/{}'.format(model_name, output_string)):
            os.makedirs('data/output/{}/{}'.format(model_name, output_string))
        image.save("data/output/{0}/{1}/{1}_{2}.png".format(model_name, output_string, idx))