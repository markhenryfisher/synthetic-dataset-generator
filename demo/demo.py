# -*- coding: utf-8 -*-
"""
Simple synthetic-dataset-generator demo 
Created on Mon Jun  1 08:38:32 2026

@author: mhf

Spyder command line...
runfile('S:/mhf/GitLab/synthetic-dataset-generator/demo/demo.py', \
                 wdir='S:/mhf/GitLab/synthetic-dataset-generator')

"""
import configparser

from syndatasetgen.core import CompositeDataset

if __name__ == '__main__':
    config=configparser.ConfigParser()
    config.read('./configs/config.ini')
    assert config['demo.example'], 'No DEMO section in config.ini'
    params = config['demo.example']
       
    dataset = CompositeDataset(params)    
    for belt_name in eval(params['dest_belts']):
        dataset.run_placement(belt_name)
