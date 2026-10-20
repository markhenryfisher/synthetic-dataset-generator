# -*- coding: utf-8 -*-
"""
Simple synthetic-dataset-generator demo 
Created on Mon Jun  1 08:38:32 2026

@author: mhf

Spyder command line...
runfile('S:/mhf/GitHub/synthetic-dataset-generator/demo/demo.py', \
                 wdir='S:/mhf/GitHub/synthetic-dataset-generator')

"""
import configparser

from syndatasetgen.core import CompositeDataset


if __name__ == '__main__':
    config=configparser.ConfigParser()
    config.read('./configs/config.ini')
    assert config['demo.example'], 'No DEMO section in config.ini'
    placement = config['demo.example']['placement']
    assert placement in ['random', 'iou_spec'], 'No placement found'
    
    # combine demo with placement
    config['combined'] = {}
    # Copy section1
    config["combined"].update(config["demo.example"])
    if placement == 'random':
        # Copy section2 (overwrites duplicate keys from section1)
        config["combined"].update(config["placement.random"])
    else:
        config["combined"].update(config["placement.iou_spec"])
        
    params = config['combined']       
    dataset = CompositeDataset(params)    
    for belt_name in eval(params['dest_belts']):
        dataset.run_placement(belt_name)
