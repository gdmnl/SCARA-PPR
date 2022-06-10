# SCARA-PPR
This is the original code for *SCARA: Scalable Graph Neural Networks with Feature-Oriented Optimization*

## Baselines
* GraphSAINT: [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT)
* APPNP: [APPNP](https://github.com/benedekrozemberczki/APPNP)
* PPRGo: [PPRGo](https://github.com/TUM-DAML/pprgo_pytorch)
* GBP: [GBP](https://github.com/chennnM/GBP)
* AGP: [AGP](https://github.com/wanghzccls/AGP-Approximate_Graph_Propagation)
* GAS: [GAS](https://github.com/rusty1s/pyg_autoscale)

## Data
* Citeseer & Pubmed: [GBP](https://github.com/chennnM/GBP)
* PPI: [GraphSAGE](http://snap.stanford.edu/graphsage/)
* Yelp: [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT)
* Reddit: [PPRGo](https://github.com/TUM-DAML/pprgo_pytorch)
* Products & Papers100M: [OGB](https://github.com/snap-stanford/ogb)
* Amazon: [Cluster-GCN](http://manikvarma.org/downloads/XC/XMLRepository.html)
* MAG: [PANE](https://renchi.ac.cn/datasets/)

## Usage

### Precompute
1. Requirements: CMake 3.16, C++ 11
2. Install [fast_double_parser](https://github.com/lemire/fast_double_parser)
3. CMake `cmake -B build`, then `make`
4. Run scripts `./run_reddit.sh`

### Train
1. Install dependencies: `conda create --name <envname> --file requirements.txt`
2. Run python `python run.py`
