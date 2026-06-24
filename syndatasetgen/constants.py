# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 09:42:12 2026

@author: mhf
"""

IMAGE_SUFFIX = '.png'
CSV_SUFFIX = '.csv'
LABEL_SUFFIX = '__labels.json'
IMAGES_FOLDER_NAME = 'label_frames'
DOTA_ANNOTATION_FOLDER_NAME = 'annfiles'
LABELS_FOLDER_NAME = 'seg_labels_json'
OCCLUSION_FOLDER_NAME = 'occlusion_frames'
AREAS_FOLDER_NAME = 'visible_fish_areas'
UNMATCHED_FOLDER_NAME = 'unmatched_seg_labels_json'
SPLITS_FOLDER_NAME = 'splits'
IMAGE_SIZE = (512, 512)
MAX_ATTEMPTS = 10

CLS_BELTS = ['belt_A', 'belt_B', 'belt_C', 'belt_D', 
             'belt_E', 'belt_F', 'belt_G']

# All fish classes which could be used in the synthetic images (whole, manually annotated examples in dataset). Used
# for classification
CLS_FISH_CLASSES = ['fish_mackerel', 'fish_redgurnard', 'fish_catfish', 'fish_haddock', 'fish_ling',
                    'fish_lemonsole', 'fish_monk', 'fish_dogfish', 'fish_commondab', 'fish_megrim',
                    'fish_doversole', 'fish_herring', 'fish_horsemackerel', 'fish_argentines',
                    'fish_skate_ray', 'fish_longroughdab', 'fish_plaice', 'fish_greygurnard',
                    'fish_whiting', 'fish_saithe', 'fish_norwaypout', 'fish_bib', 'fish_boar_fish', 'fish_brill',
                    'fish_seabass', 'fish_cod', 'fish_hake', 'fish_john_dory', 'fish_commondragonet']
