# GraSR
Fast Protein Structure Comparison through Effective Representation Learning with Contrastive Graph Neural Networks

Zenodo: [![DOI](https://zenodo.org/badge/374857961.svg)](https://zenodo.org/badge/latestdoi/374857961)

## Preparation
GraSR is implemented with Python3, so a Python3 (>=3.6) interpreter is required.

At first, download the source code of GraSR from GitHub:

    $ git clone https://github.com/chunqiux/GraSR.git

Then, we recommend you to use a virtual environment, such as virtualenv, to install the dependencies of 
GraSR. If virtualenv is not available in your OS, try to install it as following:

    $ pip3 install virtualenv

Then, create and activate the virtual environment as following:

    # Higher python version is permitted
    $ virtualenv gsr_env -p python3.6
    $ source gsr_env/bin/activate

Then, install the dependencies as following:

    $ pip install -r requirements.txt

When you want to quit the virtual environment, just:

    $ deactivate

## Usage
### Generate descriptors for query structures
If you only want to generate descriptors for query structures, you can run the command as following:

    $ python main.py -m saved_models -q example -o result

where '-m saved_models' means the directory path of models is './saved_models', '-q example'
 means the directory path of query structures is './example' and '-o result' means the output
 directory path is './result'.

The format of descriptor file is .pkl. you can check it in python3 as following:

    $ python3
    >>> import pickle
    >>> with open("result/query_descriptors.pkl", "rb") as qd_file:
    ...     d = pickle.load(qd_file)
    ...     print(d)

'd' is a dictionary. Its keys are filenames of query structures and corresponding values are
descriptors (Numpy.ndarray).

### Structure retrieval from a database
If you want to retrieve structural neighbors of the query structures, you can run the command as
 following:

    $ python main.py -r -m saved_models -q example -k descriptors/scope_207_id40.pkl -o result

where '-r' means retrieval mode and '-k descriptors/scope_207_id40.pkl' means the file path of 
database is './descriptors/scope_207_id40.pkl'. An example of retrieval result is shown below:

    Top-10 structural neighbors
    sid			Length-scaling cosine distance
    d1ca4a1.ent		0.33015
    d1lb6a_.ent		0.33941
    d3ivva1.ent		0.36020
    d2edma1.ent		0.38801
    d4ca1a_.ent		0.38843
    d2ed6a1.ent		0.42102
    d2g1da1.ent		0.43577
    d1qhva_.ent		0.43767
    d5gv0a1.ent		0.44182
    d4akma_.ent		0.44263

where 'sid' denotes the sid of the structural neighbors in the SCOPe and 'Length-scaling 
cosine distance' denotes the distance between the query structure and structural neighbors.

## License
Our project is under [GPLv3.0](https://github.com/chunqiux/GraSR/blob/master/LICENSE). 

The moco.py is modified from 
[MoCo/builder.py](https://github.com/facebookresearch/moco/blob/master/moco/builder.py), 
which is under CC-BY-NC 4.0 license. The details can be referred from 
[MoCo_LICENSE](https://github.com/chunqiux/GraSR/blob/master/MoCo_LICENSE).

## Online service
We also provide online retrieval service [here](http://www.csbio.sjtu.edu.cn/bioinf/GraSR/).
Our website follows a 'filter and refine' paradigm, which means it can provide more accurate result.
