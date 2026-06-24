The config is stored at "./configs/config.ini". We do not pass any command line arguments.
Instead we use the [configparser](https://docs.python.org/3/library/configparser.html) to load the config.  
Sections in the config define parameters for each program (\[ DEFAULT \] defines common parameters used by ALL programs)  

## \[DEFAULT]

\[DEFAULT] includes paths to the single-whole-fish database used by the compositor. We assume there is a symbolic
link  from "./data" to folder "single_wholefish_sample", but if this is not in place then "data_root" need to be changed.


### data_root
```python
# path to database
data_root: ./data
```

### single_whole_fish_root
```python
# intermediate path 
single_whole_fish_root: %(data_root)s/single_whole_fish
```

### single_whole_fish_labels
```python
# path to image annotation (json)
single_whole_fish_labels: %(single_whole_fish_root)s/seg_labels_json/all
```

### backgrounds
```python
# path to background images
backgrounds: %(data_root)s/backgrounds/
```
---
## \[demo.example]
\[demo.example] includes params. for "./demo/demo.py"

### source_belts
```Python
# Single whole fish database imagery is harvested from a number of vessels (belts). 
# source_belts: lists those available to the compositor. 
# Note: The sample database only includes imagery from 'belt_F'
source_belts: ['belt_F']
```
### source_species
```python
# Single whole fish database imagery includes a variety of fish species. 
# source_species: lists those available to the compositor.
# Note: The sample database only includes 'fish_haddock'
source_species: ['fish_haddock']
```

### dest_belts
```python
# Lists background canvas used by the compositor (see "single_whole_fish/backgrounds")
dest_belts: ['belt_F']
```

### canvas_size
```python
# The sample background (dest_belts) are resized to (x, y)
canvas_size: (512, 512) 
```

### area_thresh
```python
# Only composite single whole fish with area (pixels) > area_thresh. 
area_thresh: 1000
```

### op_mode
```python
# Details of all single whole fish are recorded in "inventory.csv".
# These are split into train.csv, val.csv, test.csv by "./tools/write_splits.py"
# op_mode: determines which is used by the compositor, i.e. train, val, or test.
op_mode: test
```

### show_output
```python
# Option controls if composited images are displayed (True/False).
show_output: True
```

### output_folder
```python
# Output will be written to this directory. 
# The output is collected in subdirectories determined by dest_belts, tgt_iou, op_mode. e.g.
# belt_F/all/test - whole composited dataset
# Note: File name format comprises 'P' followed by 6 digits. 
# First digit (0,1,2,3, etc.) indicates IoU
# belt_F_/iou_0/test - tgt_iou 0.0
# belt_F/iou_1/test - tgt_iou 0.1 etc.
output_folder: ./work_dirs/synth_demo
```

### supp_output
```python
The compositor will (optionally) also produce occlusion masks and figures. 
supp_output: {'fig_folder': None, 'occ_folder': None}
```
### fish_per_image
```python
# Number of individuals composited.
# Either [a] or [a, b], where a,b determines a random number of individuals. 
# Note: IoU is measured between the two most recent composited individuals.
# If fish_per_image > 2 the IoU between ALL individuals cannot be guaranteed.
fish_per_image: [2]
```
 
### iou_mode
```python
# Determines how IoU is measured. 
# It can either be measured between horizontal bounding boxes (HBB) 
# or oriented bounding boxes (OBB). 
iou_mode: ['hbbox_iou'] 
```

### tgt_iou
```python
# List of IoU bin centres
# Values must be 0.1x (x = 0, 1, 2, .., 5)
# Bins have fixed size (width = 0.1)
tgt_iou: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
```

### images_per_tgt_iou
```python
# Number of images per IoU bin
# Note: Consider the size of the database when setting this param.
images_per_tgt_iou: 1
```

### rotation_mode
```python
# random or None
rotation_mode: random
```

### rotation
```python
# A list of explicit rotations applied when rotation_mode is None
# Note: For debugging
# rotation: [0, 0]
```

---
## \[summarise_dataset.example]
```python
name: summarise_dataset
# write inventory to folder
output_folder: ./work_dirs/synth_dataset_summary
```
---
## \[write_splits.example] 
```python
name: write_splits
# train/val/test split (percentages)
split: [80, 10, 10]
```
---
## \[convert2coco.example]
```python
[convert2coco.example]
name: convert_dataset_to_coco_format
data_root: ./work_dirs/synth_demo/belt_F/all/test
#  comment out next parameter to omit conversion (and show annotation)
out_folder = ./work_dirs/synth_demo/belt_F/all/test/coco

```