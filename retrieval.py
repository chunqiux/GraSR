import os

import torch
import numpy as np


@torch.no_grad()
def retrieval(q_descriptors: dict, k_descriptors: dict) -> dict:
    """
    Retrieve structural neighbors of query proteins from the database.
    :param q_descriptors: Descriptors of query protein structures
    :param k_descriptors: Descriptors of proteins in the database
    :return: retrieval result of each query structure
    """
    q_ids, q_des_len = zip(*q_descriptors.items())
    q_des = np.stack([d[:-1] for d in q_des_len])
    q_len = [d[-1] for d in q_des_len]

    k_ids, k_des_len = zip(*k_descriptors.items())
    k_des = np.stack([d[:-1] for d in k_des_len])
    k_len = [d[-1] for d in k_des_len]
    dist_mat = len_scaling_cos_dist(torch.from_numpy(q_des), torch.from_numpy(k_des), q_len, k_len)
    order_tensor = torch.argsort(dist_mat, dim=1)

    result_dict = {}
    for i, q_id in enumerate(q_ids):
        pdb_dist_list = []
        for j in range(10):
            if dist_mat[i, order_tensor[i, j]].abs() < 1e-5:
                pdb_dist_list.append((k_ids[order_tensor[i, j]], 0))
            else:
                pdb_dist_list.append((k_ids[order_tensor[i, j]], dist_mat[i, order_tensor[i, j]]))
        result_dict[q_id] = pdb_dist_list

    return result_dict


def len_scaling_cos_dist(x1: torch.Tensor, x2: torch.Tensor, len_1: list, len_2: list) -> torch.Tensor:
    """
    :param x1: descriptors of query structures
    :param x2: descriptors of compared structures
    :param len_1: length list of query structures
    :param len_2: length list of compared structures
    :return: distance matrix
    """
    max_num = 1000
    x1, x2 = torch.tensor(x1, dtype=torch.float), torch.tensor(x2, dtype=torch.float)
    dist_mat = torch.zeros((x1.shape[0], x2.shape[0]))
    len_1_t, len_2_t = torch.tensor(len_1, dtype=torch.float), torch.tensor(len_2, dtype=torch.float)
    max_len = len_2_t.max()
    for i in range(0, x1.shape[0], max_num):
        if i + max_num < x1.shape[0]:
            dist_mat[i:i + max_num, :] = cos_dist(x1[i:i + max_num, :], x2)
        else:
            dist_mat[i:, :] = cos_dist(x1[i:, :], x2)
    len_scaling_mat = 1 + torch.clamp((- len_1_t.unsqueeze(1) + len_2_t.unsqueeze(0)) / max_len, min=0)
    return dist_mat / len_scaling_mat


def cos_dist(x1, x2):
    x1_norm = torch.norm(x1, dim=1).unsqueeze(1)
    x2_norm = torch.norm(x2, dim=1).unsqueeze(0)
    norm_mat = torch.clamp_min(torch.matmul(x1_norm, x2_norm), 1e-6)
    product_mat = x1.matmul(x2.transpose(1, 0))
    return 1 - product_mat / norm_mat


def write_retrieval_result(retrieval_result: dict, out_dir_path: str) -> None:
    """
    Write retrieval results to the output directory
    """
    ret_dir = os.path.join(out_dir_path, "retrieval_results")
    if not os.path.exists(ret_dir):
        os.mkdir(ret_dir)
    for q_id in retrieval_result.keys():
        with open("{}/{}.txt".format(ret_dir, q_id), "w") as rank_file:
            rank_file.write("Top-10 structural neighbors\n")
            rank_file.write("sid\t\t\tLength-scaling cosine distance\n")
            for i in range(10):
                rank_file.write("{}\t\t{:.5f}\n".format(retrieval_result[q_id][i][0], retrieval_result[q_id][i][1]))
