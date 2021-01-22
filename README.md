# Synwalk - Community Detection via Random Walk Modelling

This repository contains the evaluation framework we used in the course of our work on [Community Detection via Random Walk Modelling](http://arxiv.org/abs/2101.08623). In summary, it provides functionality for generating LFR benchmark graphs in a structured way, to run the community detection methods Synwalk, Infomap and Walktrap arbitrary graphs in edgelist format, and to evaluate the results with several metrics. Further, we also provide code for plotting the evaluations results to generate the figures in the paper mentioned above.


## Getting Started

These instructions should get you a copy of the project up and running on
your local machine in three steps:


### 1. Conda environment setup

We recommend to use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for easily recreating the Python environment. You can install the latest version like so:

```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda-latest-Linux-x86_64.sh
./Miniconda-latest-Linux-x86_64.sh
```
Once you have Miniconda installed, get a local copy of this repository via

```
git clone https://github.com/synwalk/synwalk-analysis.git
```

and create a virtual environment from the included environment description in `environment.yml` like so:
```
conda env create -f environment.yml
```
Finally, activate the conda environment via 
```
conda activate synwalk
```
and set your python path to the project root
```
export PYTHONPATH="/path/to/synwalk/"
```

### 2. Compile Synwalk
The implementation of Synwalk resides in [this repository](https://github.com/synwalk/synwalk). Follow the compilation instructions there and copy the resulting `Infomap` executable into the project root directory.

### 3. Get the benchmark data

You can get the generated LFR benchmark networks [here](https://doi.org/10.5281/zenodo.4450167). Place the data in the given directory structure in `project_root/data/lfr_benchmark/`.

For other (e.g. empirical) networks there are plenty of resources, e.g. [SNAP](https://snap.stanford.edu/data/), [KONECT](konect.cc) or the [Network Repository](http://networkrepository.com/). For the networks analyzed in [our paper](http://arxiv.org/abs/2101.08623) we implemented some cleaning methods for the raw networks, but it should be fairly straight forward to implement such cleaners yourself.

### 4. Compile the LFR network generator (Optional)

In case you want to generate LFR benchmark networks yourself, you need to compile the generator first. Simply run `make` in the `lfr_generator/` directory. The code of the LFR generator is provided by  [Santo Fortunato](https://www.santofortunato.net/) on his [homepage](https://www.santofortunato.net/resources) and we made minor changes to the interface for the integration with our framework.

### Project Layout

The following gives you a brief overview on the organization and contents of this project. Note: in general it should be clear where to change the default paths in the scripts and notebooks, but if you don't want to waste any time just use the default project structure.
```
	├── LICENSE				<- Contains the project's license.
    │
    ├── README.md			<- This readme file.
    │
    ├── environment.yml		<- The file for reproducing the Python environment necessary for
    │  	  	  	 		   	   running the code in this project. Generated with `conda <env> export > environment.yml`
    │
    ├── data
    │   ├── clean			<- Cleaned data sets, e.g. remove invalid/missing data,...
    │   ├── processed		<- The final, canonical data sets for modeling after any preprocessing steps.
    │   └── raw				<- The original, immutable data dump.
    │
    ├── figures				<- Output directory for generated figures.
    |
    ├── lfr_generator		<- Contains the C++ code for generating the LFR benchmark networks.
    │
    ├── notebooks			<- Jupyter notebooks for running experiments and analysis.
    │
    ├── results				<- Clustering and metric results get stored here.
    |
    ├── src					<- Python source code of the analysis framework.
    │   ├── __init__.py		<- Makes src a Python module
    │   ├── data			<- Cleaning empirical networks, generating and loading LFR networks.
    │   ├── lfr				<- Code for generating and storing LFR benchmark results.
    │   ├── scripts			<- Scripts for running experiments.
    │   ├── utils			<- Utils for plotting and cluster analysis.
    │   └── wrappers		<- Wrappers for igraph, infomap/synwalk, the LFR generator and some metrics.
    │
    ├── workspace			<- Workspace for temporary files created by Infomap and Synwalk.
```



## Running the code

### Generating LFR networks
Follow the instructions and run the code in `notebooks/generate_lfr_graphs.ipynb`.

### Detecting communities

Follow the instructions and run the code in `notebooks/run_lfr_benchmark.ipynb` or `notebooks/run_empirical.ipynb`to detect communities in the generated LFR networks, or in any empirical network, respectively.
You can also run detection on empirical networks via the python script in `src/scripts/run_empirical.py`, which makes sense especially for large networks when the notebook server starts running out of memory.

### Analyzing and plotting the detection and evaluation results

Follow the instructions and run the code in `notebooks/run_lfr_benchmark.ipynb` for computing Adjusted Mutual Information scores for your detection results on the LFR benchmark.  You can plot these scores by using `notebooks/plot_lfr_results.ipynb` or the script in `src/scripts/plot_lfr.py`.  You can plot sample graphs using `notebooks/plot_graph.ipynb` and run a misclassification analysis for nodes using `notebooks/misclassification_analysis.ipynb`.

Follow the instructions and run the code in `notebooks/analyze_empirical.ipynb` to analyze and plot your detection results on any empirical network.

## Authors

[Christian Toth](https://github.com/chritoth) - don't hesitate to contact me in case you have questions, remarks or suggestions about the code or about Synwalk!

## Acknowledgments

We thank [Santo Fortunato](https://www.santofortunato.net/) for providing his code for generating LFR benchmark networks on his [homepage](https://www.santofortunato.net/resources).

## License

The content of this repository is licensed under the Apache License v2.0 as stated in LICENSE.md.

