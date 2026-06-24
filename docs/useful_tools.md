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

### belt2coco_from_json
The compositor ("demo.py") produces a dataset in the same format as the database (i.e. a legacy format with separate folders for image frames and json annotation). 
"belt2coco_from_json.py" produces [coco format](https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html).  
Note: \[convert2coco] config section provides input/output paths. If the output path is omitted then dataset images will be displayed with associated annotation. 

