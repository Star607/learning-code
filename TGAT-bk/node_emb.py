"""Unified interface to all dynamic graph model experiments"""
from data_util import load_graph, load_label_data, load_split_data
import math
import logging
import time
import sys
import random
import argparse

from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from module import TGAN
from graph import NeighborFinder

random.seed(222)
np.random.seed(222)
torch.manual_seed(222)

### Argument and global variables
parser = argparse.ArgumentParser(
    'Interface for TGAT experiments on node classification')
parser.add_argument('-d',
                    '--data',
                    type=str,
                    help='data sources to use, try wikipedia or reddit',
                    default='wikipedia')
parser.add_argument('-f', '--freeze', action='store_true')
parser.add_argument('--bs', type=int, default=200, help='batch_size')
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--n_degree',
                    type=int,
                    default=50,
                    help='number of neighbors to sample')
parser.add_argument('--n_neg', type=int, default=1)
parser.add_argument('--n_head', type=int, default=2)
parser.add_argument('--n_epoch', type=int, default=15, help='number of epochs')
parser.add_argument('--n_layer', type=int, default=2)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument(
    '--tune',
    action='store_true',
    help='parameters tunning mode, use train-test split on training data only.'
)
parser.add_argument('--drop_out',
                    type=float,
                    default=0.1,
                    help='dropout probability')
parser.add_argument('--gpu',
                    type=int,
                    default=0,
                    help='idx for the gpu to use')
parser.add_argument('--node_dim',
                    type=int,
                    default=128,
                    help='Dimentions of the node embedding')
parser.add_argument('--time_dim',
                    type=int,
                    default=128,
                    help='Dimentions of the time embedding')
parser.add_argument('--agg_method',
                    type=str,
                    choices=['attn', 'lstm', 'mean'],
                    help='local aggregation method',
                    default='attn')
parser.add_argument('--attn_mode',
                    type=str,
                    choices=['prod', 'map'],
                    default='prod')
parser.add_argument('--time',
                    type=str,
                    choices=['time', 'pos', 'empty'],
                    help='how to use time information',
                    default='time')

parser.add_argument('--new_node', action='store_true', help='model new node')
parser.add_argument('--uniform',
                    action='store_true',
                    help='take uniform sampling from temporal neighbors')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
UNIFORM = args.uniform
NEW_NODE = args.new_node
USE_TIME = args.time
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
SEQ_LEN = NUM_NEIGHBORS
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_LAYER = 2
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Load data and train val test split
if True:
    edges, n_nodes, val_time, test_time = load_graph(dataset=DATA)
    g_df = edges[["from_node_id", "to_node_id", "timestamp"]].copy()
    g_df["idx"] = np.arange(1, len(g_df) + 1)
    g_df.columns = ["u", "i", "ts", "idx"]

    if len(edges.columns) > 4:
        e_feat = edges.iloc[:, 4:].to_numpy()
        padding = np.zeros((1, e_feat.shape[1]))
        e_feat = np.concatenate((padding, e_feat))
    else:
        e_feat = np.zeros((len(g_df) + 1, NODE_DIM))

    if args.freeze:
        # e_feat = edges.iloc[:, 4:].to_numpy()
        # NODE_DIM = e_feat.shape[1]
        n_feat = np.zeros((n_nodes + 1, NODE_DIM))
    else:
        # e_feat = np.zeros((len(g_df) + 1, NODE_DIM))
        bound = np.sqrt(6 / (2 * NODE_DIM))
        n_feat = np.random.uniform(-bound, bound, (n_nodes + 1, NODE_DIM))

    src_l = g_df.u.values
    dst_l = g_df.i.values
    e_idx_l = g_df.idx.values
    ts_l = g_df.ts.values
    label_l = edges.state_label.values

    max_src_index = src_l.max()
    max_idx = max(src_l.max(), dst_l.max())

    random.seed(2020)

# set train, validation, test datasets
if True:
    valid_train_flag = (ts_l < val_time)

    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    train_ts_l = ts_l[valid_train_flag]
    train_e_idx_l = e_idx_l[valid_train_flag]
    train_label_l = label_l[valid_train_flag]

    _, val_data, test_data = load_split_data(dataset=DATA)

    val_src_l = val_data.u.values
    val_dst_l = val_data.i.values
    val_ts_l = val_data.ts.values
    val_label_l = val_data.state_label.values

    test_src_l = test_data.u.values
    test_dst_l = test_data.i.values
    test_ts_l = test_data.ts.values
    test_label_l = test_data.state_label.values

### Initialize the data structure for graph and edge sampling
adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l,
                              train_ts_l):
    adj_list[src].append((dst, eidx, ts))
    adj_list[dst].append((src, eidx, ts))
train_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM)

# full graph with all the data for the test and validation purpose
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)

### Model initialize
device = torch.device('cuda:{}'.format(GPU))
tgan = TGAN(train_ngh_finder,
            n_feat,
            e_feat,
            num_layers=NUM_LAYER,
            use_time=USE_TIME,
            agg_method=AGG_METHOD,
            attn_mode=ATTN_MODE,
            seq_len=SEQ_LEN,
            n_head=NUM_HEADS,
            drop_out=DROP_OUT,
            node_dim=NODE_DIM,
            time_dim=TIME_DIM)
# optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
# criterion = torch.nn.BCELoss()
tgan = tgan.to(device)

num_instance = len(train_src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)
logger.debug('num of training instances: {}'.format(num_instance))
logger.debug('num of batches per epoch: {}'.format(num_batch))
idx_list = np.arange(num_instance)
np.random.shuffle(idx_list)

logger.info('loading saved TGAN model')
model_path = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{DATA}.pth'
tgan.load_state_dict(torch.load(model_path))
tgan.eval()
logger.info('TGAN models loaded')
logger.info('Start node embedding generation.')

from tqdm import trange


@torch.no_grad()
def eval_emb(tgan, src_l, dst_l, ts_l, label_l, batch_size, num_layer=NODE_LAYER):
    embs = []
    tgan.eval()
    num_instance = len(src_l)
    num_batch = math.ceil(num_instance / batch_size)

    for k in trange(num_batch):
        s_idx = k * batch_size
        e_idx = min(num_instance, s_idx + batch_size)
        src_l_cut = src_l[s_idx:e_idx]
        dst_l_cut = dst_l[s_idx:e_idx]
        ts_l_cut = ts_l[s_idx:e_idx]
        src_embed = tgan.tem_conv(src_l_cut, ts_l_cut, num_layer)
        dst_embed = tgan.tem_conv(dst_l_cut, ts_l_cut, num_layer)
        embs.append(torch.cat([src_embed, dst_embed], dim=1))

    return torch.cat(embs, dim=0)


train_embs = eval_emb(tgan, train_src_l, train_dst_l, train_ts_l,
                      train_label_l, BATCH_SIZE)
train_embs = train_embs.cpu().numpy()
val_embs = eval_emb(tgan, val_src_l, val_dst_l, val_label_l, val_ts_l,
                    BATCH_SIZE)
val_embs = val_embs.cpu().numpy()
test_embs = eval_emb(tgan, test_src_l, test_dst_l, test_ts_l, test_label_l,
                     BATCH_SIZE)
test_embs = test_embs.cpu().numpy()
np.savez(f'./saved_embs/{DATA}.npz', train_embs=train_embs, val_embs=val_embs, test_embs=test_embs)

