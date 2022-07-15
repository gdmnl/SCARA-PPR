# SCARA-PPR
This is the original code for *SCARA: Scalable Graph Neural Networks with Feature-Oriented Optimization*

[GitHub](https://github.com/gdmnl/SCARA-PPR) |
[Tech Report](https://sites.google.com/view/scara-techreport)

**Reference Format:**
```
Ningyi Liao, Dingheng Mo, Siqiang Luo, Xiang Li, and Pengcheng Yin.
SCARA: Scalable Graph Neural Networks with Feature-Oriented Optimization. PVLDB, 15(11): XXX-XXX, 2022.
```

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

### Data Preparation
* Path: `data/[dataset_name]`
* `adj.txt`: adjacency table
  * First line: "`# [number of nodes]`"
* `degrees.npz`: node degrees in .npz 'arr_0'
* `feats_norm.npz`: normalized features in .npz 'arr_0' uncompressed
  * Large matrix can be split
* `labels.npz`: node label information
  * 'label': labels (number or one-hot)
  * 'idx_train/idx_val/idx_test': indices of training/validation/test nodes
* `query.txt`: indices of query nodes

### Precompute
1. Requirements: CMake 3.16, C++ 11
2. CMake `cmake -B build`, then `make`
3. Run scripts `./run_reddit.sh`

### Train
1. Install dependencies: `conda create --name <envname> --file requirements.txt`
2. Run python `python run.py`
