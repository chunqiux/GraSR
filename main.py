#!gf_env/bin/python

import argparse
import pickle
import os
from encode import get_descriptors
from retrieval import retrieval, write_retrieval_result


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str,
                        help='Path of the model file(s) to be loaded. If it is a directory, all model files in it will '
                             'be loaded and the average of descriptors will be output.')
    parser.add_argument('-k', '--key_descriptors', type=str, default="",
                        help='Path of descriptor files of the database (Only needed when retrieval)')
    parser.add_argument('-q', '--query_protein', type=str,
                        help='Path of query protein structure file(s) (PDB format). If it is a directory, all model '
                             'files in it will be loaded')
    parser.add_argument('-o', '--output', type=str, default="./",
                        help="Path of output directory")
    parser.add_argument('-r', "--retrieval", help='retrieval from the selected database', action='store_true')
    return parser.parse_args()


def main():
    args = parse_argument()
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    qd_out_path = os.path.join(args.output, "query_descriptors.pkl")
    q_descriptors = get_descriptors(args.model, args.query_protein, qd_out_path)
    if args.retrieval:
        with open(args.key_descriptors, "rb") as kd_file:
            k_descriptors = pickle.load(kd_file)
        ret_result = retrieval(q_descriptors, k_descriptors)
        write_retrieval_result(ret_result, args.output)


if __name__ == "__main__":
    main()
