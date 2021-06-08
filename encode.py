import pickle
import os
import math

from Bio.PDB import PDBParser
import torch
import torch.nn as nn
import numpy as np

from model import Encoder
from moco import MoCo

# Hyper-parameters
N_REF_POINTS = 31
DESCRIPTOR_DIM = 400
QUEUE_SIZE = 1024
MOMENTUM = 0.999
TAU = 0.07


def load_model(model_path: str) -> MoCo:
    """
    Load saved models for feature representation
    :param model_path: Path of a saved pytorch model
    :return: A modified MoCo object for reference
    """
    net = MoCo(N_REF_POINTS, Encoder, DESCRIPTOR_DIM, QUEUE_SIZE, MOMENTUM, TAU)
    state_dict = {k.replace("module.", ""): v for k, v in torch.load(model_path, map_location='cpu').items()}
    net.load_state_dict(state_dict)
    return net


def get_ca_coordinate(pdb_path: str) -> list:
    """
    Get coordinates of alpha carbon atoms in the given pdb file
    :param pdb_path: Path of a pdb file
    :return: List of Ca coordinates
    """
    s = PDBParser().get_structure('protein', pdb_path)
    atom_ids = ('CA',)
    fea = []
    for chain in s[0]:
        for res in chain:
            for ai in atom_ids:
                if ai in res:
                    fea.append(res[ai].coord)
    return fea


def get_raw_feature_tensor(pdb_path_list: list) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Get torch.tensor of raw features of proteins in one batch.
    Raw features consist of relative coordinates, sequence length list and adjacency matrices.
    All features are padded according to the length of longest protein sequence.
    :param pdb_path_list: List of PDB file path
    :return: Relative coordinates, sequence length list and adjacency matrices
    """
    xyz_list, am_list = [], []
    for pdb_path in pdb_path_list:
        ca_co = get_ca_coordinate(pdb_path)
        assert len(ca_co) > 0
        g, a = extract_raw_features(ca_co, N_REF_POINTS)
        xyz_list.append(torch.tensor(g, dtype=torch.float))
        am_list.append(a)
    xyz_pad = nn.utils.rnn.pad_sequence(xyz_list, batch_first=True)
    len_list = [xyz.shape[0] for xyz in xyz_list]
    len_tensor = torch.tensor(len_list, dtype=torch.int)
    max_len = torch.max(len_tensor)

    am_pad_list = []
    for am in am_list:
        am_len = am.shape[0]
        if am_len < max_len:
            am_pad_list.append(torch.cat(
                (torch.from_numpy(am).float(), torch.zeros((am_len, max_len - am_len))), dim=1))
        else:
            am_pad_list.append(torch.from_numpy(am).float())
    am_pad_tensor = nn.utils.rnn.pad_sequence(am_pad_list, batch_first=True)

    return xyz_pad, len_tensor, am_pad_tensor


@torch.no_grad()
def get_descriptors(model_dir_path: str, pdb_dir_path: str, out_path: str) -> dict:
    """
    Generate descriptors for PDB files.
    :param model_dir_path: Path of saved model directory (or file)
    :param pdb_dir_path: Path of PDB directory (os file)
    :param out_path: Path of output descriptors
    :return: None
    """
    import warnings
    warnings.filterwarnings("ignore")
    # load model
    if os.path.isdir(model_dir_path):
        model_name_list = os.listdir(model_dir_path)
        model_path_list = [os.path.join(model_dir_path, n) for n in model_name_list]
    else:
        model_path_list = [model_dir_path]
    net_list = [load_model(mp) for mp in model_path_list]
    # load file path
    if os.path.isdir(pdb_dir_path):
        pdb_fn_list = os.listdir(pdb_dir_path)
        pdb_path_list = [os.path.join(pdb_dir_path, n) for n in pdb_fn_list]
    else:
        pdb_path_list = [pdb_dir_path]
        pdb_fn_list = [os.path.basename(pdb_dir_path)]

    descriptor_dict = {}
    batch_size = 50
    n_pdb_files = len(pdb_fn_list)
    # compute descriptors in batch to accelerate
    for j in range(0, n_pdb_files, batch_size):
        if j + batch_size < n_pdb_files:
            end_idx = j + batch_size
        else:
            end_idx = n_pdb_files
        x, ld, am = get_raw_feature_tensor(pdb_path_list[j:end_idx])
        y_list = []
        for net in net_list:
            net.eval()
            y = net((x, x, ld, ld, am, am), True)
            y_list.append(y)
        avg_y = sum(y_list) / len(y_list)
        avg_y = avg_y.numpy()
        # sequence length is used as the last element of the descriptors
        for i, pdb_fn in enumerate(pdb_fn_list[j:end_idx]):
            descriptor_dict[pdb_fn] = np.concatenate((avg_y[i, :], ld[i].numpy().reshape(1)))
    # save descriptors
    with open(out_path, 'wb') as out_f:
        pickle.dump(descriptor_dict, out_f)
    return descriptor_dict


def extract_raw_features(xyz_list: list, n_ref_points: int) -> (np.ndarray, np.ndarray):
    """
    Extract raw features for a single protein structure
    :param xyz_list: List of Ca coordinates
    :param n_ref_points: Number of reference points
    :return: Raw node features and adjacency matrix
    """
    # get raw node features
    rxyz = get_relative_coordinate(xyz_list, n_ref_points)
    alpha_angle = cal_alphac_angle(xyz_list)
    fea = np.concatenate((rxyz, alpha_angle), axis=1)
    # get the adjacency matrix
    omega, epsilon = 4.0, 2.0
    p = np.array(xyz_list)
    vp = np.expand_dims(p, axis=1)
    dist_mat = np.sqrt(np.sum(np.square(vp - p), axis=2))
    adj_mat = np.divide(omega, np.maximum(dist_mat, epsilon))

    return fea, adj_mat


def get_relative_coordinate(xyz: list, n_ref_points: int) -> np.ndarray:
    """
    Compute relative coordinates of a protein structure
    :param xyz: List of Ca coordinates
    :param n_ref_points: Number of reference points
    :return: Array of relative coordinates
    """
    xyz = np.array(xyz)
    group_num = int(np.log2(n_ref_points + 1))
    assert 2 ** group_num - 1 == n_ref_points,\
        "The number of anchor points is {} and should be 2^k - 1, " \
        "where k is an integer, but k is {}.".format(n_ref_points, group_num)
    n_points = xyz.shape[0]
    ref_points = []
    for i in range(group_num):
        n_points_in_group = 2 ** i
        for j in range(n_points_in_group):
            beg, end = n_points * j // n_points_in_group, math.ceil(n_points * (j + 1) / n_points_in_group)
            ref_point = np.mean(xyz[beg:end, :], axis=0)
            ref_points.append(ref_point)
    coordinates = [np.linalg.norm(xyz - rp, axis=1).reshape(-1, 1) for rp in ref_points]

    return np.concatenate(coordinates, axis=1)


def cal_alphac_angle(xyz: list) -> np.ndarray:
    """
    Get angle features of a protein structure. Each angle is formed by three consecutive Ca.
    The cosine of each angle is used as its angle feature.
    :param xyz: List of Ca coordinates
    :return: Array of angle features
    """
    direction_vec = np.array(xyz)[1:, :] - np.array(xyz)[:-1, :]
    dv_1 = direction_vec[:-1, :]
    dv_2 = direction_vec[1:, :]
    dv_dot = np.sum(dv_1 * dv_2, axis=1)
    dv_norm = np.linalg.norm(dv_1, axis=1) * np.linalg.norm(dv_2, axis=1)
    # padding for two terminals
    pad_dv_norm = np.zeros((len(xyz)))
    pad_dv_norm[1:-1] = dv_dot / dv_norm

    return pad_dv_norm.reshape((-1, 1))


