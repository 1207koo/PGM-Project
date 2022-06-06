import argparse
import os
import random
import numpy as np

parser = argparse.ArgumentParser(description="demo args", formatter_class=argparse.RawTextHelpFormatter)

### hyperparameters
parser.add_argument("--feature-maps", type=int, default=64, help="number of feature maps")
parser.add_argument("--rotations", action="store_true", help="use of rotations self-supervision during training")
parser.add_argument("--model", type=str, default="ResNet12_nd", help="model to train")
parser.add_argument("--sampler-model", type=str, default="Sampler3_nd", help="sampler model to train")
parser.add_argument("--false-sample", type=int, default=4, help="number to sample with sampler")
parser.add_argument("--variance", type=float, default=1.0, help="variance for nondeterministic layer")
parser.add_argument("--support-aggregate", type=str, default='mean', help="aggregation method for support descriptors")
parser.add_argument("--query-aggregate", type=str, default='mean', help="aggregation method for query descriptors")

### pytorch options
parser.add_argument("--dataset-path", type=str, default="/gallery_moma/junseo.koo/dataset", help="dataset path")
parser.add_argument("--dataset", type=str, default="miniImageNet", help="dataset to use")
parser.add_argument("--load-model", type=str, default="model.pt", help="load model from file")
parser.add_argument("--load-sampler-model", type=str, default="sampler.pt", help="load sampler model from file")

### few-shot parameters
parser.add_argument("--n-shots", type=str, default="1", help="how many shots per few-shot run, can be int or list of ints. In case of episodic training, use first item of list as number of shots.")
parser.add_argument("--n-ways", type=int, default=5, help="number of few-shot ways")
parser.add_argument("--n-queries", type=int, default=5, help="number of few-shot queries")

try :
    get_ipython()
    args = parser.parse_args(args=[])
except :
    args = parser.parse_args()

if args.dataset_path[-1] != '/':
    args.dataset_path += "/"

try:
    n_shots = int(args.n_shots)
    args.n_shots = [n_shots]
except:
    args.n_shots = eval(args.n_shots)

if args.variance < 0:
    args.variance = None

args.seed = 0
args.deterministic = False
args.dropout = False
args.ncm_loss = False