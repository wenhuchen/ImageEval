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

# Copy this script inside of data/dataset_output/dreambooth or data/dataset_output/dreambooth_reg and run it inside of the folder.

# iterate through all subjects
# for each subject, for all methods in side of subject, 
# delete the folders with prompts that do not correspond to subject

import glob
import os
import shutil

all_subjects = sorted(glob.glob('*/'))

live_subjects = ['cat', 'cat2', 'dog', 'dog2', 'dog3', 'dog5', 'dog6',
'dog7', 'dog8', 'red_cartoon']

object_subjects = ['backpack', 'backpack_dog', 'bear_plushie', 'berry_bowl', 
'can', 'candle', 'clock', 'colorful_sneaker', 'duck_toy', 'fancy_boot', 
'grey_sloth_plushie', 'monster_toy', 'pink_sunglasses', 'poop_emoji', 'rc_car', 
'robot_toy', 'shiny_sneaker', 'teapot', 'vase', 'wolf_plushie']

object_prompts = \
[
'a_token_floating_in_an_ocean_of_milk',
'a_token_floating_on_top_of_water',
'a_token_in_the_jungle',
'a_token_in_the_snow',
'a_token_on_a_cobblestone_street',
'a_token_on_the_beach',
'a_token_on_top_of_a_dirt_road',
'a_token_on_top_of_a_mirror',
'a_token_on_top_of_a_purple_rug_in_a_forest',
'a_token_on_top_of_a_white_rug',
'a_token_on_top_of_a_wooden_floor',
'a_token_on_top_of_green_grass_with_sunflowers_around_it',
'a_token_on_top_of_pink_fabric',
'a_token_on_top_of_the_sidewalk_in_a_crowded_street',
'a_token_with_a_blue_house_in_the_background',
'a_token_with_a_city_in_the_background',
'a_token_with_a_mountain_in_the_background',
'a_token_with_a_tree_and_autumn_leaves_in_the_background',
'a_token_with_a_wheat_field_in_the_background',
'a_token_with_the_Eiffel_Tower_in_the_background',
'a_cube_shaped_token',
'a_purple_token',
'a_red_token',
'a_shiny_token',
'a_wet_token'
]

live_prompts = \
[
'a_token_in_the_jungle',
'a_token_in_the_snow',
'a_token_on_a_cobblestone_street',
'a_token_on_the_beach',
'a_token_on_top_of_a_purple_rug_in_a_forest',
'a_token_on_top_of_a_wooden_floor',
'a_token_on_top_of_pink_fabric',
'a_token_with_a_blue_house_in_the_background',
'a_token_with_a_city_in_the_background',
'a_token_with_a_mountain_in_the_background',
'a_cube_shaped_token',
'a_purple_token',
'a_red_token',
'a_shiny_token',
'a_wet_token',
'a_token_wearing_a_rainbow_scarf',
'a_token_wearing_a_red_hat',
'a_token_wearing_a_santa_hat',
'a_token_wearing_a_yellow_shirt',
'a_token_wearing_pink_glasses',
'a_token_wearing_a_black_top_hat_and_a_monocle',
"a_token_in_a_chef’s_outfit",
'a_token_in_a_chef_outfit',
'a_token_in_a_firefighter_outfit',
'a_token_in_a_police_outfit',
'a_token_in_a_purple_wizard_outfit'
]



for subject in all_subjects:
    # print(subject)
    all_methods = sorted(glob.glob(os.path.join(subject, '*/')))
    subject_id = subject.replace('/', '')
    for method in all_methods:
        # print(method)
        all_prompts = sorted(glob.glob(os.path.join(method, '*/')))
        for prompt in all_prompts:
            prompt_id = prompt.split('/')[-2].replace('/', '')
            if subject_id in object_subjects and prompt_id not in object_prompts:
                print(subject_id, 'is object subject and', prompt_id, 'is not object prompt.')
                shutil.rmtree(prompt)
            elif subject_id in live_subjects and prompt_id not in live_prompts:
                print(subject_id, 'is live subject and', prompt_id, 'is not subject prompt.')
                shutil.rmtree(prompt)