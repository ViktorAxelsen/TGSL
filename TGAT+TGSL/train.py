import os
import math
import time
import random
import argparse
import torch
import dgl
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import average_precision_score, roc_auc_score

from TGAT import TGAN
from neighbor_finder import NeighborFinder
from utils import EarlyStopMonitor, RandEdgeSampler, set_seed, get_device, show_time
from config import *
from view_learner import ETGNN, TimeMapping
from MTL import MTL

torch.autograd.set_detect_anomaly(True)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='dataset', default='wikipedia')
parser.add_argument('--cuda', type=str, required=True, help='idx for the gpu to use')
parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'], help='local aggregation method', default='attn')
parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod', help='use dot product attention or mapping based')
parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'], help='how to use time information', default='time')
parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--tolerance', type=float, default=1e-3)
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--gtau', type=float, default=1.0)
parser.add_argument('--K', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--coe', type=float, default=0.2)
parser.add_argument('--ratio', type=float, default=0.02)
parser.add_argument('--infer_bs', type=int, default=200)
parser.add_argument('--can_nn', type=int, default=20)
parser.add_argument('--rnn_nn', type=int, default=20)
parser.add_argument('--rnn_layer', type=int, default=1)
parser.add_argument('--can_type', type=str, choices=['1st', '3rd', 'random', 'mix'], default='3rd')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)

if args.dataset == 'wikipedia':
    config = WikiConfig()
elif args.dataset == 'reddit':
    config = RedditConfig()
elif args.dataset == 'escorts':
    config = ESConfig()
else:
    raise Exception('Dataset Error')

DATA = args.dataset
AGG_METHOD = args.agg_method
ATTN_MODE = args.attn_mode
UNIFORM = args.uniform
USE_TIME = args.time

BATCH_SIZE = config.BATCH_SIZE
NUM_EPOCH = config.EPOCH
LEARNING_RATE = config.LR

NUM_NEIGHBORS = config.N_DEGREE
NUM_HEADS = config.N_HEAD
NUM_LAYER = config.N_LAYER
DROP_OUT = config.DROPOUT
NODE_DIM = config.NODE_DIM
TIME_DIM = config.TIME_DIM
SEED = args.seed
PATIENCE = args.patience
TOLERANCE = args.tolerance
TAU = args.tau
GTAU = args.gtau
K = args.K
SSL_BATCH_SIZE = args.batch_size
COE = args.coe
RATIO = args.ratio
INFER_BS = args.infer_bs
NUM_CAN_NN = args.can_nn
NUM_RNN_NN = args.rnn_nn
CAN_TYPE = args.can_type
NUM_RNN_LAYER = args.rnn_layer

NUM_NEG = 1
SEQ_LEN = NUM_NEIGHBORS

set_seed(SEED)

os.makedirs(f"./saved_models/", exist_ok=True)
os.makedirs(f"./saved_checkpoints/", exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.dataset}.pth'
get_checkpoint_path = lambda epoch: f'./saved_checkpoints/{args.prefix}-{args.agg_method}-{args.attn_mode}-{args.dataset}-{epoch}.pth'



def eval_one_epoch(hint, tgan, view_learner, edge_rnn, sample_time_encoder, sampler, src, dst, ts, label):
    val_acc, val_ap, val_f1, val_auc = [], [], [], []
    with torch.no_grad():
        tgan = tgan.eval()
        view_learner = view_learner.eval()
        edge_rnn = edge_rnn.eval()
        sample_time_encoder = sample_time_encoder.eval()
        TEST_BATCH_SIZE = INFER_BS
        num_test_instance = len(src)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        train_edge_feat = view_learner(g)
        full_edge_feat = torch.zeros(e_feat.shape[0] + 1, train_edge_feat.shape[1], device=device)
        full_edge_feat[train_e_idx_l - 1] = train_edge_feat
        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
            src_l_cut = src[s_idx:e_idx]
            dst_l_cut = dst[s_idx:e_idx]
            ts_l_cut = ts[s_idx:e_idx]
            size = len(src_l_cut)
            src_l_fake, dst_l_fake = sampler.sample(size)

            train_node_cut = np.array(list(set(np.append(src_l_cut, dst_l_cut)).intersection(train_node_set)))
            max_ts = np.array([max_train_ts_l + 1] * len(train_node_cut))
            neighbor_finder = NeighborFinder(adj_list, uniform=True)
            neighbor_node_idx, neighbor_edge_idx, neighbor_ts = neighbor_finder.get_temporal_neighbor(train_node_cut, max_ts, num_neighbors=NUM_RNN_NN)

            neighbor_edge_idx = neighbor_edge_idx.reshape(-1) - 1
            neighbor_edge_feat = full_edge_feat[neighbor_edge_idx]  # [bs * 20, 172]
            neighbor_edge_feat = neighbor_edge_feat.reshape(neighbor_node_idx.shape[0], neighbor_node_idx.shape[1], -1)  # [bs, 20, 172]

            neighbor_edge_feat = neighbor_edge_feat.transpose(0, 1)
            _, (h_n, _) = edge_rnn(neighbor_edge_feat)
            context_vec = h_n[-1]  # [bs, 172]

            neighbor_finder.uniform = True
            if CAN_TYPE == '1st':
                candidate_node_idx, candidate_edge_idx, candidate_ts = neighbor_finder.get_temporal_neighbor(train_node_cut, max_ts, num_neighbors=NUM_CAN_NN)

                src_node_idx_aug = np.repeat(train_node_cut.reshape(train_node_cut.shape[0], 1), candidate_node_idx.shape[1], axis=1)  # [bs, 20]
                dst_node_idx_aug = candidate_node_idx  # [bs, 20]
            elif CAN_TYPE == '3rd':
                candidate_node_idx, candidate_edge_idx, candidate_ts = neighbor_finder.find_k_hop(3, train_node_cut, max_ts, num_neighbors=NUM_CAN_NN)
                candidate_node_idx = candidate_node_idx[-1].reshape(train_node_cut.shape[0], -1)
                candidate_edge_idx = candidate_edge_idx[-1].reshape(train_node_cut.shape[0], -1)
                candidate_ts = candidate_ts[-1].reshape(train_node_cut.shape[0], -1)

                src_node_idx_aug = np.repeat(train_node_cut.reshape(train_node_cut.shape[0], 1), candidate_node_idx.shape[1], axis=1)  # [bs, 20]
                dst_node_idx_aug = candidate_node_idx  # [bs, 20]
            elif CAN_TYPE == 'random':
                candidate_node_idx = np.random.choice(np.array(list(train_node_set)), size=train_node_cut.shape[0] * NUM_CAN_NN, replace=True).reshape(train_node_cut.shape[0], -1)  # [bs, 20]
                candidate_edge_idx = np.array([0] * (train_node_cut.shape[0] * NUM_CAN_NN)).reshape(train_node_cut.shape[0], -1)
                candidate_ts = np.random.rand(train_node_cut.shape[0], NUM_CAN_NN) * max_train_ts_l

                src_node_idx_aug = np.repeat(train_node_cut.reshape(train_node_cut.shape[0], 1), candidate_node_idx.shape[1], axis=1)  # [bs, 20]
                dst_node_idx_aug = candidate_node_idx  # [bs, 20]
            elif CAN_TYPE == 'mix':
                candidate_node_idx_1st, candidate_edge_idx_1st, candidate_ts_1st = neighbor_finder.get_temporal_neighbor(train_node_cut, max_ts, num_neighbors=NUM_CAN_NN)
                candidate_node_idx_3rd, candidate_edge_idx_3rd, candidate_ts_3rd = neighbor_finder.find_k_hop(3, train_node_cut, max_ts, num_neighbors=NUM_CAN_NN)
                candidate_node_idx_3rd = candidate_node_idx_3rd[-1].reshape(train_node_cut.shape[0], -1)
                candidate_edge_idx_3rd = candidate_edge_idx_3rd[-1].reshape(train_node_cut.shape[0], -1)
                candidate_ts_3rd = candidate_ts_3rd[-1].reshape(train_node_cut.shape[0], -1)

                candidate_node_idx = np.concatenate((candidate_node_idx_1st, candidate_node_idx_3rd), axis=-1)
                candidate_edge_idx = np.concatenate((candidate_edge_idx_1st, candidate_edge_idx_3rd), axis=-1)
                candidate_ts = np.concatenate((candidate_ts_1st, candidate_ts_3rd), axis=-1)

                src_node_idx_aug = np.repeat(train_node_cut.reshape(train_node_cut.shape[0], 1), candidate_node_idx.shape[1], axis=1)  # [bs, 20]
                dst_node_idx_aug = candidate_node_idx  # [bs, 20]
            else:
                pass

            candidate_edge_idx = candidate_edge_idx.reshape(-1) - 1
            candidate_edge_feat = full_edge_feat[candidate_edge_idx]  # [bs * 20, 172]
            candidate_edge_feat = candidate_edge_feat.reshape(candidate_node_idx.shape[0], candidate_node_idx.shape[1], -1)  # [bs, 20, 172]

            ts_aug = np.random.rand(candidate_ts.shape[0], candidate_ts.shape[1]) * max_train_ts_l
            delta_ts_sample = ts_aug - candidate_ts
            delta_ts_sample_context = ts_aug - np.ones_like(candidate_ts) * max_train_ts_l
            delta_ts_sample_embedding = sample_time_encoder(torch.tensor(delta_ts_sample.reshape(-1, 1), dtype=torch.float32).to(device)).reshape(ts_aug.shape[0], ts_aug.shape[1], -1)
            delta_ts_sample_context_embedding = sample_time_encoder(torch.tensor(delta_ts_sample_context.reshape(-1, 1), dtype=torch.float32).to(device)).reshape(ts_aug.shape[0], ts_aug.shape[1], -1)

            context_vec = context_vec.unsqueeze(1).expand_as(candidate_edge_feat)
            context_vec = context_vec * delta_ts_sample_context_embedding
            candidate_edge_feat = candidate_edge_feat * delta_ts_sample_embedding
            aug_edge_logits = torch.sum(context_vec * candidate_edge_feat, dim=-1)  # [bs, 20, 1]

            # Gumble-Top-K
            bias = 0.0 + 0.0001  # If bias is 0, we run into problems
            eps = (bias - (1 - bias)) * torch.rand(aug_edge_logits.size()) + (1 - bias)
            gate_inputs = torch.log(eps) - torch.log(1 - eps)
            gate_inputs = gate_inputs.to(aug_edge_logits.device)
            gate_inputs = (gate_inputs + aug_edge_logits) / GTAU
            z = torch.sigmoid(gate_inputs).squeeze()  # [bs, 20]
            __, sorted_idx = z.sort(dim=-1, descending=True)
            keep = sorted_idx[:, :int(RATIO * z.size(1))]  # [bs, k]

            aug_edge_logits = torch.sigmoid(gate_inputs).squeeze()  # [bs, 20]
            aug_edge_weight = torch.gather(aug_edge_logits, dim=1, index=keep)  # [bs, k]
            ts_aug = torch.gather(torch.tensor(ts_aug, device=device), dim=1, index=keep).detach().cpu().numpy()  # [bs, k]
            src_node_idx_aug = torch.gather(torch.tensor(src_node_idx_aug, device=device), dim=1, index=keep).detach().cpu().numpy()  # [bs, k]
            dst_node_idx_aug = torch.gather(torch.tensor(dst_node_idx_aug, device=device), dim=1, index=keep).detach().cpu().numpy()  # [bs, k]
            candidate_edge_feat = torch.gather(candidate_edge_feat, dim=1, index=keep.unsqueeze(2).repeat(1, 1, candidate_edge_feat.shape[2]))  # [bs, k, 172]

            aug_edge_weight = aug_edge_weight.reshape(-1)
            ts_aug = ts_aug.reshape(-1)
            src_node_idx_aug = src_node_idx_aug.reshape(-1)
            dst_node_idx_aug = dst_node_idx_aug.reshape(-1)

            temp_eid = e_feat.shape[0]
            new_eid_list = []
            adj_list_aug = pickle.loads(full_adj_list_pickle)
            for src_aug, dst_aug, ts_aug_temp in zip(src_node_idx_aug, dst_node_idx_aug, ts_aug):
                adj_list_aug[src_aug].append((dst_aug, temp_eid, ts_aug_temp))
                adj_list_aug[dst_aug].append((src_aug, temp_eid, ts_aug_temp))
                new_eid_list.append(temp_eid)
                temp_eid += 1
            train_ngh_finder_aug = NeighborFinder(adj_list_aug, uniform=tgan.ngh_finder.uniform)

            new_eid_list = np.array(new_eid_list)
            full_aug_edge_weight = torch.ones(temp_eid, device=device)
            full_aug_edge_weight[new_eid_list - 1] = aug_edge_weight

            candidate_edge_feat = candidate_edge_feat.reshape(-1, candidate_edge_feat.shape[
                2]).detach().cpu().numpy()  # [bs * k, 172]
            e_feat_aug = np.concatenate((e_feat, candidate_edge_feat), axis=0)
            e_feat_th_aug = torch.nn.Parameter(torch.from_numpy(e_feat_aug.astype(np.float32)))
            edge_raw_embed_aug = torch.nn.Embedding.from_pretrained(e_feat_th_aug, padding_idx=0, freeze=True).to(
                device)

            ngh_finder_ori = tgan.ngh_finder
            tgan.ngh_finder = train_ngh_finder_aug
            edge_raw_embed_ori = tgan.edge_raw_embed
            tgan.edge_raw_embed = edge_raw_embed_aug

            pos_prob, neg_prob = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS, full_aug_edge_weight=full_aug_edge_weight)

            # Recover
            tgan.ngh_finder = ngh_finder_ori
            tgan.edge_raw_embed = edge_raw_embed_ori
            
            pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            
            val_acc.append((pred_label == true_label).mean())
            val_ap.append(average_precision_score(true_label, pred_score))
            val_auc.append(roc_auc_score(true_label, pred_score))

    return np.mean(val_acc), np.mean(val_ap), np.mean(val_f1), np.mean(val_auc)

# Load data and train val test split
g_df = pd.read_csv('./processed/ml_{}.csv'.format(DATA))
e_feat = np.load('./processed/ml_{}.npy'.format(DATA))
n_feat = np.load('./processed/ml_{}_node.npy'.format(DATA))

if e_feat.shape[1] < 172:
    edge_zero_padding = np.zeros((e_feat.shape[0], 172 - e_feat.shape[1]))
    e_feat = np.concatenate([e_feat, edge_zero_padding], axis=1)
if n_feat.shape[1] < 172:
    node_zero_padding = np.zeros((n_feat.shape[0], 172 - n_feat.shape[1]))
    n_feat = np.concatenate([n_feat, node_zero_padding], axis=1)

val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))

src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values

max_src_index = src_l.max()
max_idx = max(src_l.max(), dst_l.max())

total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
num_total_unique_nodes = len(total_node_set)

mask_node_set = set(random.sample(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])), int(0.1 * num_total_unique_nodes)))
mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
none_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)

valid_train_flag = (ts_l <= val_time) * (none_node_flag > 0)

train_src_l = src_l[valid_train_flag]
train_dst_l = dst_l[valid_train_flag]
train_ts_l = ts_l[valid_train_flag]
train_e_idx_l = e_idx_l[valid_train_flag]
train_label_l = label_l[valid_train_flag]

# define the new nodes sets for testing inductiveness of the model
train_node_set = set(train_src_l).union(train_dst_l)
assert(len(train_node_set - mask_node_set) == len(train_node_set))
new_node_set = total_node_set - train_node_set

# select validation and test dataset
valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
valid_test_flag = ts_l > test_time

is_new_node_edge = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(src_l, dst_l)])
nn_val_flag = valid_val_flag * is_new_node_edge
nn_test_flag = valid_test_flag * is_new_node_edge

# validation and test with all edges
val_src_l = src_l[valid_val_flag]
val_dst_l = dst_l[valid_val_flag]
val_ts_l = ts_l[valid_val_flag]
val_e_idx_l = e_idx_l[valid_val_flag]
val_label_l = label_l[valid_val_flag]

test_src_l = src_l[valid_test_flag]
test_dst_l = dst_l[valid_test_flag]
test_ts_l = ts_l[valid_test_flag]
test_e_idx_l = e_idx_l[valid_test_flag]
test_label_l = label_l[valid_test_flag]
# validation and test with edges that at least has one new node (not in training set)
nn_val_src_l = src_l[nn_val_flag]
nn_val_dst_l = dst_l[nn_val_flag]
nn_val_ts_l = ts_l[nn_val_flag]
nn_val_e_idx_l = e_idx_l[nn_val_flag]
nn_val_label_l = label_l[nn_val_flag]

nn_test_src_l = src_l[nn_test_flag]
nn_test_dst_l = dst_l[nn_test_flag]
nn_test_ts_l = ts_l[nn_test_flag]
nn_test_e_idx_l = e_idx_l[nn_test_flag]
nn_test_label_l = label_l[nn_test_flag]

# Initialize the data structure for graph and edge sampling
# build the graph for fast query
# graph only contains the training data (with 10% nodes removal)
adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
    adj_list[src].append((dst, eidx, ts))
    adj_list[dst].append((src, eidx, ts))
train_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM)

# full graph with all the data for the test and validation purpose
full_adj_list = [[] for _ in range(max_idx + 1)]
for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
    full_adj_list[src].append((dst, eidx, ts))
    full_adj_list[dst].append((src, eidx, ts))
full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)

adj_list_pickle = pickle.dumps(adj_list, -1)
full_adj_list_pickle = pickle.dumps(full_adj_list, -1)


train_rand_sampler = RandEdgeSampler(train_src_l, train_dst_l)
val_rand_sampler = RandEdgeSampler(src_l, dst_l)
nn_val_rand_sampler = RandEdgeSampler(nn_val_src_l, nn_val_dst_l)
test_rand_sampler = RandEdgeSampler(src_l, dst_l)
nn_test_rand_sampler = RandEdgeSampler(nn_test_src_l, nn_test_dst_l)


device = get_device(index=0)
max_train_ts_l = max(train_ts_l)
'''
+++++++++++++++++++++++++++++ MTL Stage +++++++++++++++++++++++++++++
'''
# DGL Graph Construction
g = dgl.graph((train_src_l, train_dst_l))

ndata = []
for ind in range(g.num_nodes()):
    if ind in train_node_set:
        ndata.append(n_feat[ind])
    else:
        ndata.append([0] * n_feat.shape[1])

edata_feat = e_feat[train_e_idx_l]
edata_ts = train_ts_l.reshape(train_ts_l.shape[0], -1)

g.ndata['feat'] = torch.tensor(np.array(ndata), dtype=torch.float32)
g.edata['edge_feat'] = torch.tensor(np.array(edata_feat), dtype=torch.float32)
g.edata['ts'] = torch.tensor(np.array(edata_ts), dtype=torch.float32)

g = dgl.add_self_loop(g)
g = dgl.add_reverse_edges(g, copy_ndata=True, copy_edata=True)
g = g.to(device)

SSL_Encoder_k = TGAN(train_ngh_finder, n_feat, e_feat, num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD,
                     attn_mode=ATTN_MODE, seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=0.0)
view_learner = ETGNN(in_dim=n_feat.shape[1], hidden_dim=n_feat.shape[1], train_src_l=train_src_l, train_dst_l=train_dst_l)
tgan = TGAN(train_ngh_finder, n_feat, e_feat, num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD,
            attn_mode=ATTN_MODE, seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT)
edge_rnn = torch.nn.LSTM(input_size=n_feat.shape[1], hidden_size=n_feat.shape[1], num_layers=NUM_RNN_LAYER, bidirectional=False)
sample_time_encoder = TimeMapping()
model = MTL(base_encoder_k=SSL_Encoder_k, encoder=tgan, view_learner=view_learner, edge_rnn=edge_rnn,
            sample_time_encoder=sample_time_encoder, len_full_edge=e_feat.shape[0], train_e_idx_l=train_e_idx_l,
            train_node_set=train_node_set, train_ts_l=train_ts_l, e_feat=e_feat, device=device, K=K, ratio=RATIO,
            can_nn=NUM_CAN_NN, rnn_nn=NUM_RNN_NN, can_type=CAN_TYPE, tau=TAU, gtau=GTAU)
model = model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
ssl_criterion = torch.nn.CrossEntropyLoss().to(device)
criterion = torch.nn.BCELoss().to(device)
early_stopper = EarlyStopMonitor(max_round=PATIENCE, tolerance=TOLERANCE)


num_instance = len(train_src_l)
num_batch = math.ceil(num_instance / BATCH_SIZE)

print('Fine-tuning Stage: num of training instances: {}'.format(num_instance))
print('Fine-tuning Stage: num of batches per epoch: {}'.format(num_batch))

for epoch in range(NUM_EPOCH):
    start_time = time.time()
    model.encoder.ngh_finder = train_ngh_finder
    acc, ap, f1, auc, m_loss = [], [], [], [], []
    print('{} start {} epoch'.format(show_time(), epoch))
    for k in range(num_batch):
        s_idx = k * BATCH_SIZE
        e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
        src_l_cut, dst_l_cut = train_src_l[s_idx:e_idx], train_dst_l[s_idx:e_idx]
        ts_l_cut = train_ts_l[s_idx:e_idx]
        label_l_cut = train_label_l[s_idx:e_idx]
        size = len(src_l_cut)
        src_l_fake, dst_l_fake = train_rand_sampler.sample(size)

        with torch.no_grad():
            pos_label = torch.ones(size, dtype=torch.float, device=device)
            neg_label = torch.zeros(size, dtype=torch.float, device=device)
        
        optimizer.zero_grad()
        model = model.train()

        pos_prob, neg_prob, pos_prob_ed, neg_prob_ed, output, target = model(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS, g, adj_list, adj_list_pickle)

        loss = criterion(pos_prob, pos_label)
        loss += criterion(neg_prob, neg_label)
        loss += criterion(pos_prob_ed, pos_label)
        loss += criterion(neg_prob_ed, neg_label)
        loss += ssl_criterion(output, target) * COE
        loss.backward()
        optimizer.step()
        # get training results
        with torch.no_grad():
            model = model.eval()
            pred_score = np.concatenate([(pos_prob).cpu().detach().numpy(), (neg_prob).cpu().detach().numpy()])
            pred_label = pred_score > 0.5
            true_label = np.concatenate([np.ones(size), np.zeros(size)])
            acc.append((pred_label == true_label).mean())
            ap.append(average_precision_score(true_label, pred_score))
            m_loss.append(loss.item())
            auc.append(roc_auc_score(true_label, pred_score))


    end_time = time.time()
    print('epoch: {} took {:.2f}s'.format(epoch, end_time - start_time))
    # validation phase use all information
    tgan.ngh_finder = full_ngh_finder
    val_acc, val_ap, val_f1, val_auc = eval_one_epoch('val for old nodes', tgan, view_learner, edge_rnn, sample_time_encoder, val_rand_sampler, val_src_l,
    val_dst_l, val_ts_l, val_label_l)

    nn_val_acc, nn_val_ap, nn_val_f1, nn_val_auc = eval_one_epoch('val for new nodes', tgan, view_learner, edge_rnn, sample_time_encoder, val_rand_sampler, nn_val_src_l,
    nn_val_dst_l, nn_val_ts_l, nn_val_label_l)

    print('epoch: {}:'.format(epoch))
    print('Epoch mean loss: {}'.format(np.mean(m_loss)))
    print('train acc: {}, val acc: {}, new node val acc: {}'.format(np.mean(acc), val_acc, nn_val_acc))
    print('train auc: {}, val auc: {}, new node val auc: {}'.format(np.mean(auc), val_auc, nn_val_auc))
    print('train ap: {}, val ap: {}, new node val ap: {}'.format(np.mean(ap), val_ap, nn_val_ap))
    # print('train f1: {}, val f1: {}, new node val f1: {}'.format(np.mean(f1), val_f1, nn_val_f1))

    if early_stopper.early_stop_check(val_ap):
        print('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
        print(f'Loading the best model at epoch {early_stopper.best_epoch}')
        best_model_path = get_checkpoint_path(early_stopper.best_epoch)
        tgan.load_state_dict(torch.load(best_model_path))
        view_learner.load_state_dict(torch.load(best_model_path[:-4] + '-ViewLearner.pth'))
        edge_rnn.load_state_dict(torch.load(best_model_path[:-4] + '-edgernn.pth'))
        sample_time_encoder.load_state_dict(torch.load(best_model_path[:-4] + '-ste.pth'))
        print(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
        tgan.eval()
        view_learner.eval()
        edge_rnn.eval()
        sample_time_encoder.eval()
        break
    else:
        torch.save(tgan.state_dict(), get_checkpoint_path(epoch))
        torch.save(view_learner.state_dict(), get_checkpoint_path(epoch)[:-4] + '-ViewLearner.pth')
        torch.save(edge_rnn.state_dict(), get_checkpoint_path(epoch)[:-4] + '-edgernn.pth')
        torch.save(sample_time_encoder.state_dict(), get_checkpoint_path(epoch)[:-4] + '-ste.pth')


# testing phase use all information
tgan.ngh_finder = full_ngh_finder
test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for old nodes', tgan, view_learner, edge_rnn, sample_time_encoder, test_rand_sampler, test_src_l,
test_dst_l, test_ts_l, test_label_l)

nn_test_acc, nn_test_ap, nn_test_f1, nn_test_auc = eval_one_epoch('test for new nodes', tgan, view_learner, edge_rnn, sample_time_encoder, nn_test_rand_sampler, nn_test_src_l,
nn_test_dst_l, nn_test_ts_l, nn_test_label_l)

print('Test statistics: Old nodes -- acc: {}, auc: {}, ap: {}'.format(test_acc, test_auc, test_ap))
print('Test statistics: New nodes -- acc: {}, auc: {}, ap: {}'.format(nn_test_acc, nn_test_auc, nn_test_ap))

print('Saving TGAN model')
torch.save(tgan.state_dict(), MODEL_SAVE_PATH)
torch.save(view_learner.state_dict(), MODEL_SAVE_PATH[:-4] + '-ViewLearner.pth')
torch.save(edge_rnn.state_dict(), MODEL_SAVE_PATH[:-4] + '-edgernn.pth')
torch.save(sample_time_encoder.state_dict(), MODEL_SAVE_PATH[:-4] + '-ste.pth')
print('TGAN models saved')
