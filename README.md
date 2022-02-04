# SCARA-PPR
This is the origianl code for SCARA: Scalable Graph Neural Networks with Feature-Oriented Optimization

## Baselines
* GraphSAINT: [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT)
* APPNP: [APPNP](https://github.com/benedekrozemberczki/APPNP)
* PPRGo: [PPRGo](https://github.com/TUM-DAML/pprgo_pytorch)
* GBP: [GBP](https://github.com/chennnM/GBP)
* AGP: [AGP](https://github.com/wanghzccls/AGP-Approximate_Graph_Propagation)

## Data
* Citeseer & Pubmed: [GBP](https://github.com/chennnM/GBP)
* PPI: [GraphSAGE](http://snap.stanford.edu/graphsage/)
* Yelp: [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT)
* Reddit: [PPRGo](https://github.com/TUM-DAML/pprgo_pytorch)
* Amazon & Papers100M: [OGB](https://github.com/snap-stanford/ogb)

## Usage

### Precompute
1. Install [cnpy](https://github.com/rogersce/cnpy), [fast_double_parser](https://github.com/lemire/fast_double_parser)
2. CMake `cmake -B build`, then `make`
3. Run scripts `./run_reddit.sh`

### Train
1. Run python `python run_transductive.py`
