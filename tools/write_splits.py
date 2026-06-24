# -*- coding: utf-8 -*-
"""
Produce train/test/val (csv) files from inventory.csv

Created on Tue Jun 16 12:54:48 2026

@author: mhf

Spyder command line...
runfile('S:/mhf/GitLab/synthetic-dataset-generator/tools/write_splits.py', \
                 wdir='S:/mhf/GitLab/synthetic-dataset-generator')
"""
import configparser
from syndatasetgen.synth_utils.repo_mgt import FishRepo 

def write_splits(exp_params):
    splits_folder = exp_params['single_whole_fish_root']
    split = eval(exp_params['split'])
    repo = FishRepo(splits_folder)
    repo.write_splits(split)
    
    return


if __name__ == '__main__':
    config=configparser.ConfigParser()
    config.read('./configs/config.ini')
    params = config['write_splits.example']
    write_splits(params)