# Traffic Engineering with Joint Link Weight and Segment Optimization

## Overview 
This Project is an extension of https://github.com/tfenz/TE_SR_WAN_simulation.git.
This Code extends the original project by 3 new algorithms LeastLoadedLinkFirst(LLLF), ... and ... . We also analyse two other objectives for these algorithms.
For further information on all the dependencies and how to set up the project go to the original. There is all the information you should need.

## Install Python & Dependencies
Create a conda environment and install all python dependencies using the provided environment.yml file:
```bash
conda env create -f environment.yml
```
The created environment is named 'wan_sr', activate with:
```bash
conda activate wan_sr
```

## Run Tests
Navigate to source code root:
```bash
cd ./src
```

### Start 
Run evaluation with:
```bash
python3 main.py
```

### Output
The results are stored in a JSON file located in **[out/](src)** after running the main.py script.

## Plot Results
Create Plots from provided raw result data 
```bash
python3 plot_results.py [optional <data-dir> containing json result data]
```
```bash
python3 plot_results.py "../out/"
```

