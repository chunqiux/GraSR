# Dataset

You can download SCOPe v2.07 (40% ID filtered subset) from its [website](https://scop.berkeley.edu/downloads/pdbstyle/pdbstyle-sel-gs-bib-40-2.07.tgz).
We remove some structures due to the technical issues. The sid of remaining structures after removal is provided in [SCOPe_v207_filtered.txt](https://github.com/chunqiux/GraSR/blob/master/data/SCOPe_v207_filtered.txt).

Besides SCOPe v2.07, we also construct an independent dataset to evaluate our model and other state-of-the-art methods.
This dataset consists of protein structures derived from PDB, the release date of which is from Oct 1st 2017 to Oct 1st 2019. If a certain PDB file contains multiple chains, it will be split into multiple files, each of which contains only one chain. Then, CD-HIT is used to remove redundant sequences. The sequence identity of the independent set and that between the independent set and SCOPe v2.07 are both under 40% after filtering. At last, 51 protein chains are removed due to the technical issues.
The final independent test set (named [ind_PDB](https://github.com/chunqiux/GraSR/blob/master/data/ind_PDB.txt)) contains 1,914 protein structures.

You can easily download the structure files of ind_PDB from PDB website according to the PDB ID and chain ID of each structure, which can be downloaded [here](https://www.rcsb.org/). For example, 5ypzA means chain A of 5YPZ.