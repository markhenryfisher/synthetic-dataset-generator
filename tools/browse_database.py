# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 12:12:25 2026

@author: mhf

Spyder command line...
runfile('S:/mhf/GitLab/synthetic-dataset-generator/tools/browse_database.py', \
                 wdir='S:/mhf/GitLab/synthetic-dataset-generator')

"""
import configparser
import numpy as np
import cv2
import time
from syndatasetgen.synth_utils.repo_mgt import FishRepo
from syndatasetgen.image_labelling_tool import labelling_tool
from syndatasetgen.debug_utils import visualize as vis 


def display_database(params):
    splits_folder = params['single_whole_fish_root']

    # instantiate FishRepo    
    fish_repo = FishRepo(splits_folder, 
                         fname='inventory.csv')
    
    for n in range(fish_repo.num_fish):
        this_entry = fish_repo.next_item()
        lbl_fname = this_entry['Fname']
        img_fname = FishRepo.label_to_image(lbl_fname)
        single_limg = labelling_tool.PersistentLabelledImage(
                          img_fname,
                          lbl_fname)
        label_img, _ = single_limg.render_label_instances(
            [this_entry['Species']], 
            multichannel_mask=False)
        contours, _ = cv2.findContours(label_img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # there may be more than one contour
        # temp=[]
        # for c in contours:
        #     temp.append(c)
            
        # c = np.vstack(temp)
        
        isClosed = True
        color = (255, 0, 0) # Red color in RGB
        thickness = 2
        temp = (single_limg.pixels * 255).astype(np.uint8)
        
        # Draw all contours        
        temp_image = cv2.polylines(temp, contours, isClosed, color, thickness)

        # display annotated fish
        vis.display_img(temp_image, '{}_{}'.format(this_entry['Species'], n))        

        time.sleep(2)    

        
        


if __name__ == '__main__':
    config=configparser.ConfigParser()
    config.read('./configs/config.ini')    
    params = config['DEFAULT']
    display_database(params)