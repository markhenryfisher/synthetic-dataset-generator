# Useful Tools
See "./tools"
## browse_database
Displays database imagery with annotation. Example output is shown [here](./database.md)

## summerise_dataset
Produces "belt_tally.csv" and "fish_tally.csv" files

## write_inventory
Database file "inventory.csv" is used by "demo.py". It must be updated if the single_whole_fish 
database is changed (i.e. by running "write_inventory.py").  
Note: "write_inventory.py" will not overwrite "inventory.csv". You must rename or delete it first.

## write_splits
Samples "inventory.csv" and writes "train.csv", "val.csv" and "test.csv" using params. in \[write_splits.example] 

## dataset_converters
The compositor ("demo.py") produces a dataset in a legacy UEA catchmonitor format with separate subdirectories for image frames and json annotation). 

### belt2coco_from_json
"belt2coco_from_json.py" produces [coco format](https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html).  
Note: \[convert2coco.example] config section provides input/output paths. 

### belt2everyfish_coco
"belt2everyfish_coco.py" produces [EVERYFISH_COCO v1.0 format](https://www.dropbox.com/scl/fi/vx3x4i13hx9ildc3gy4qz/EVERYFISH-Data-Structure.pdf?rlkey=w3mup6q0ql5emw9211ng7sn0b&st=ny5y4wlt&dl=0).  
Note: \[convert2everyfish_coco.example] config section provides input/output paths.
