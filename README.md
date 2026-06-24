<div align="left">
  <img src="resources/UEA1289.png" width="120">
</div>
<div>&nbsp;</div>
<div align="center">
<img src="resources/example_0.png" width="100"> 
<img src="resources/example_2.png" width="100"> 
<img src="resources/example_6.png" width="100"> 
<img src="resources/example_4.png" width="100"> 
<img src="resources/example_5.png" width="100"> 
</div>

# Project Title
EVERYFISH SYNTHETIC-DATASET-GENERATOR (COMPOSITOR)
## Description
A Python library for generating 'synthetic' (composited) imagery for training and testing AI models targeting fisheries REM video surveilance applications. 
The placement of individual fish in the composited image is parameterised. This allows training/validation/test datasets to be generated with specific 
intersection-over-union (IoU) relationships.  These are useful for exploring the robustness of AI models with regard to occlusion.
## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowlegement](#ackknowledgement)
## Installation
1. Clone the repository:
  git clone https://<username>@git.imr.no/everyfish/synthetic-dataset-generator 
2. Install syndatagen package:  
$ cd synthetic-dataset-generator  
$ pip install -v -e .  
'-v' means verbose, or more output,  
'-e' means installing a project in editable mode,  
thus any local modifications made to the code will take effect without reinstallation.
3. Dependencies  
You do not need a GPU. See requirements.txt for details of the development environment. 
4. Download the single whole fish (sample) database:  
<a href="https://www.dropbox.com/scl/fi/ztd86xj3k8xdw00btku8y/single_whole_fish_sample.zip?rlkey=ti5wl3xgrctr3snfjqf48ri5p&st=zn84mssc&dl=0" download>Download ZIP</a>   
Note: click the link, or paste it into the search bar in a web browser. 
Then, click the downward arrow in the upper-right corner and choose "Or continue with download only" to download a file without an account.
5. Create a symbolic link to the database: ./data -> single_whole_fish  
Note: If ./data is not linked you will need to edit the \[DEFAULT] section of "./configs/config.ini"  
6. Run the demo
$ python ./demo/demo.py

## Documentation
- [learn about config](docs/customize_config.md)
- [learn about database](docs/database.md)
- [learn about demo](docs/demo.md)
- [useful tools](docs/useful_tools.md)  

## Contributing
Contributers are welcome to clone/modify/share the package subject to the terms of the License. 

## License
This project is released under the [GNU General Puplic License V3.0](https://www.gnu.org/licenses/gpl-3.0.en.html) 

## Acknowledgement
We acknowledge funding from the European Unions Horizon Europe research and innovation programme under grant agreement number 101059892.  
Contributers:  
Dr. Geoff French <brittix1023@gmail.com>  
Dr. Ruth Veevers  
Dr. Mark Fisher <Mark.Fisher@uea.ac.uk>  
Dr. Hang Zhou <Hang.Zhou@uea.ac.uk>  
Prof. Michal Mackiewicz <M.Mackiewicz@uea.ac.uk>

## Contact
<Mark.Fisher@uea.ac.uk>
