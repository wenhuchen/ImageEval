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

import os

names = ['backpack', 'backpack_dog', 'bear_plushie', 'berry_bowl', 'can', 'candle', 'cat', 'cat2', 'clock',
'colorful_sneaker', 'dog', 'dog2', 'dog3', 'dog5', 'dog6', 'dog7', 'dog8', 'duck_toy', 
'fancy_boot', 'grey_sloth_plushie', 'monster_toy', 'pink_sunglasses', 'poop_emoji', 
'rc_car', 'red_cartoon', 'robot_toy', 'shiny_sneaker', 'teapot', 'vase', 'wolf_plushie']

eval_folder = 'data/dataset_output_final'
if not os.path.exists(eval_folder):
    os.makedirs(eval_folder)

for name in names:
    os.system('mkdir {0}'.format(name))
    os.system('cp -r data/train_data/dataset/{0} {1}/{0}/0'.format(name, eval_folder))
    os.system('cp -r data/dataset_output/dreambooth_reg/{0} {1}/{0}/1'.format(name, eval_folder))
    os.system('cp -r data/dataset_output/dreambooth/{0} {1}/{0}/2'.format(name, eval_folder))

    # Optional - if you've generated Textual Inversion outputs
    # os.system('cp -r data/dataset_output/ti_output/{0} {0}/3'.format(name))