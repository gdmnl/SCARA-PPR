# SCARA-PPR
This is the original code for *SCARA: Scalable Graph Neural Networks with Feature-Oriented Optimization*

[GitHub](https://github.com/gdmnl/SCARA-PPR) |
[Tech Report](https://sites.google.com/view/scara-techreport)
<!-- TODO: paper & arxiv link -->

## Usage
**We provide a complete example and its log in the [demo notebook](demo.ipynb). The sample PubMed dataset is available in the [data folder](data/pubmed/).**

### Data Preparation
1. Download data (links [below](#dataset-link)) in GBP format to path `data/[dataset_name]`. Similar to the PubMed dataset example, there are three files:
  * `adj.txt`: adjacency table
    * First line: "`# [number of nodes]`"
  * `feats.npy`: features in .npy array
  * `labels.npz`: node label information
    * 'label': labels (number or one-hot)
    * 'idx_train/idx_val/idx_test': indices of training/validation/test nodes (inductive task)
2. Run command `python data_processor.py` to generate additional processed files:
  * `degrees.npz`: node degrees in .npz 'arr_0'
  * `feats_norm.npy`: normalized features in .npy array
    * Large matrix can be split
  * `query.txt`: indices of queried nodes

### Precompute
1. Environment: CMake 3.16, C++ 11. Dependencies (already included): [SFMT](https://github.com/MersenneTwister-Lab/SFMT), [libnpy](https://github.com/llohse/libnpy/)
2. CMake `cmake -B build`, then `make`
3. Run script: `./run_pubmed.sh`

### Train and Test
1. Install dependencies: `conda create --name [envname] --file requirements.txt`
2. Run experiment: `python run.py -f [seed] -c [config_file] -v [device]`

## Baseline Models
* GraphSAINT: [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT)
* APPNP: [APPNP](https://github.com/benedekrozemberczki/APPNP)
* PPRGo: [PPRGo](https://github.com/TUM-DAML/pprgo_pytorch)
* GBP: [GBP](https://github.com/chennnM/GBP)
* AGP: [AGP](https://github.com/wanghzccls/AGP-Approximate_Graph_Propagation)
* GAS: [GAS](https://github.com/rusty1s/pyg_autoscale)

## Dataset Links
* Citeseer & Pubmed: [GBP](https://github.com/chennnM/GBP)
* PPI: [GraphSAGE](http://snap.stanford.edu/graphsage/)
* Yelp: [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT)
* Reddit: [PPRGo](https://github.com/TUM-DAML/pprgo_pytorch)
* Products & Papers100M: [OGB](https://github.com/snap-stanford/ogb)
* Amazon: [Cluster-GCN](http://manikvarma.org/downloads/XC/XMLRepository.html)
* MAG: [PANE](https://renchi.ac.cn/datasets/)

## Citing

If you find this work useful, please cite our paper:
```
Ningyi Liao, Dingheng Mo, Siqiang Luo, Xiang Li, and Pengcheng Yin.
SCARA: Scalable Graph Neural Networks with Feature-Oriented Optimization.
PVLDB, 15(11): 3240-3248, 2022.
```
```
@article{liao2022scara,
  title={SCARA: Scalable Graph Neural Networks with Feature-Oriented Optimization},
  author={Liao, Ningyi and Mo, Dingheng and Luo, Siqiang and Li, Xiang and Yin, Pengcheng},
  journal={Proceedings of the VLDB Endowment},
  volume={15},
  number={11},
  pages={3240-3248},
  year={2022},
  publisher={VLDB Endowment}
}
```