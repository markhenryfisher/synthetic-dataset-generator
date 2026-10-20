# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:39:27 2026

@author: mhf

Spyder command line...
runfile('S:/mhf/GitHub/synthetic-dataset-generator/tools/summarise_dataset.py', \
                 wdir='S:/mhf/GitHub/synthetic-dataset-generator')
"""
import configparser
import glob
import os
import pandas as pd
from syndatasetgen.constants import LABEL_SUFFIX, CLS_BELTS, \
            CLS_FISH_CLASSES



def analyse_data(exp_params):
    
    def update_tally(cls_list, filename, tally):
        for item in cls_list:
            if item in filename:
                ok = True
                tally[item] += 1
                break
            else:
                ok = False
            
        if not ok:
            raise RuntimeError('Error: {}'.format(filename))
            
        return item, tally

    
    output_folder = exp_params['output_folder']
    os.makedirs(output_folder, exist_ok=True)
    # params    
    single_whole_fish_labels = exp_params['single_whole_fish_labels']
    fish_labels = glob.glob(os.path.join(single_whole_fish_labels, '*' + LABEL_SUFFIX))

    belt_tally = {belt : 0 for belt in CLS_BELTS}
    fish_tally = {fish : 0 for fish in CLS_FISH_CLASSES}
  
    for fish_label_filename in fish_labels:
        filepath, filename = os.path.split(fish_label_filename)
        belt, belt_tally =  update_tally(CLS_BELTS, filename, belt_tally) 
        fish, fish_tally =  update_tally(CLS_FISH_CLASSES, filename, fish_tally) 
        print('{}:{}'.format(belt, fish))

    print('\nNo. whole fish: {}'.format(len(fish_labels)))
    
    # Create DataFrame
    belt_df = pd.DataFrame(belt_tally, index=['Count'])
    fish_df = pd.DataFrame(fish_tally, index=['Count'])
    # print(belt_df)
    # print(fish_df)
    filename = os.path.join(output_folder, 'belt_tally.csv')
    print(f"\nWriting '{filename}'")
    belt_df.to_csv(filename, index=False)    
    filename = os.path.join(output_folder, 'fish_tally.csv')
    print(f"Writing '{filename}'")
    fish_df.to_csv(filename, index=False)    

    return


if __name__ == '__main__':
    config=configparser.ConfigParser()
    config.read('./configs/config.ini')
    params = config['summarise_dataset.example']
    analyse_data(params)
