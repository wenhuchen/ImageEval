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

model_name_list = ['backpack', 'backpack_dog', 'bear_plushie', 'berry_bowl', 'can', 'candle', 'cat', 'cat2', 'clock',
'colorful_sneaker', 'dog', 'dog2', 'dog3', 'dog5', 'dog6', 'dog7', 'dog8', 'duck_toy', 
'fancy_boot', 'grey_sloth_plushie', 'monster_toy', 'pink_sunglasses', 'poop_emoji', 
'rc_car', 'red_cartoon', 'robot_toy', 'shiny_sneaker', 'teapot', 'vase', 'wolf_plushie']

class_name_list = ['backpack', 'backpack', 'stuffed animal', 'bowl', 'can', 'candle', 'cat', 'cat', 'clock', 'sneaker',
'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'dog', 'toy', 'boot', 'stuffed animal', 'toy', 'sunglasses', 'toy', 'toy', 
'cartoon', 'toy', 'sneaker', 'teapot', 'vase', 'stuffed animal']

print('Number of models:', len(model_name_list))

unique_token = 'monadikos'
class_token = '<class_token>'

assert(len(model_name_list) == len(class_name_list))

prompt_list = [
'a {0} {1} in the jungle'.format(unique_token, class_token),
'a {0} {1} in the snow'.format(unique_token, class_token),
'a {0} {1} on the beach'.format(unique_token, class_token),
'a {0} {1} on a cobblestone street'.format(unique_token, class_token),
'a {0} {1} on top of pink fabric'.format(unique_token, class_token),
'a {0} {1} on top of a wooden floor'.format(unique_token, class_token),
'a {0} {1} with a city in the background'.format(unique_token, class_token),
'a {0} {1} with a mountain in the background'.format(unique_token, class_token),
'a {0} {1} with a blue house in the background'.format(unique_token, class_token),
'a {0} {1} on top of a purple rug in a forest'.format(unique_token, class_token),
'a {0} {1} with a wheat field in the background'.format(unique_token, class_token),
'a {0} {1} with a tree and autumn leaves in the background'.format(unique_token, class_token),
'a {0} {1} with the Eiffel Tower in the background'.format(unique_token, class_token),
'a {0} {1} floating on top of water'.format(unique_token, class_token),
'a {0} {1} floating in an ocean of milk'.format(unique_token, class_token),
'a {0} {1} on top of green grass with sunflowers around it'.format(unique_token, class_token),
'a {0} {1} on top of a mirror'.format(unique_token, class_token),
'a {0} {1} on top of the sidewalk in a crowded street'.format(unique_token, class_token),
'a {0} {1} on top of a dirt road'.format(unique_token, class_token),
'a {0} {1} on top of a white rug'.format(unique_token, class_token),
'a {0} {1} wearing a red hat'.format(unique_token, class_token),
'a {0} {1} wearing a santa hat'.format(unique_token, class_token),
'a {0} {1} wearing a rainbow scarf'.format(unique_token, class_token),
'a {0} {1} wearing a black top hat and a monocle'.format(unique_token, class_token),
'a {0} {1} in a chef outfit'.format(unique_token, class_token),
'a {0} {1} in a firefighter outfit'.format(unique_token, class_token),
'a {0} {1} in a police outfit'.format(unique_token, class_token),
'a {0} {1} wearing pink glasses'.format(unique_token, class_token),
'a {0} {1} wearing a yellow shirt'.format(unique_token, class_token),
'a {0} {1} in a purple wizard outfit'.format(unique_token, class_token),
'a red {0} {1}'.format(unique_token, class_token),
'a purple {0} {1}'.format(unique_token, class_token),
'a shiny {0} {1}'.format(unique_token, class_token),
'a wet {0} {1}'.format(unique_token, class_token),
'a cube shaped {0} {1}'.format(unique_token, class_token)
]

n_samples = 6

for class_name, model_name in zip(class_name_list, model_name_list):
    print(class_name, model_name)
    model_id = "data/models/cvpr_models/dreambooth/{}".format(model_name)
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda") 
    for prompt in prompt_list:
        prompt_new = prompt.replace(class_token, class_name)
        output_string = prompt_new.replace(' ', '_')
        print(prompt_new)
        for idx in range(n_samples):
            with autocast("cuda"):
                image = pipe(prompt_new, num_inference_steps=50, guidance_scale=7.5).images[0]
            if not os.path.exists('data/cvpr_output/dreambooth/{}/{}'.format(model_name, output_string)):
                os.makedirs('data/cvpr_output/dreambooth/{}/{}'.format(model_name, output_string))
            image.save("data/cvpr_output/dreambooth/{0}/{1}/{1}_{2}.png".format(model_name, output_string, idx))