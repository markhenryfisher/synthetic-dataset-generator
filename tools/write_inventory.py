# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:39:27 2026

@author: mhf

Spyder command line...
runfile('S:/mhf/GitLab/synthetic-dataset-generator/tools/write_inventory.py', \
                 wdir='S:/mhf/GitLab/synthetic-dataset-generator')
"""
import configparser
from syndatasetgen.synth_utils.repo_mgt import FishRepo  


def write_inventory(exp_params):
    splits_folder = exp_params['single_whole_fish_root']
    repo = FishRepo(splits_folder)
    repo.write_inventory()
    
    return


if __name__ == '__main__':
    config=configparser.ConfigParser()
    config.read('./configs/config.ini')
    params = config['DEFAULT']
    write_inventory(params)