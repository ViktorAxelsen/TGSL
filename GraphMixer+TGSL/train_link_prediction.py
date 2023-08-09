import logging
import time
import sys
import os
import numpy as np
import warnings
import json
import torch
import torch.nn as nn
import pickle
import dgl

from torch.utils.data import DataLoader
from models.GraphMixer import GraphMixer
from models.modules import MergeLayer
from utils.utils import set_random_seed, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler, NeighborSampler
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data, Data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args

from view_learner import ETGNN, TimeMapping
from MTL import MTL


def evaluate_model_link_prediction(model_name: str, model: nn.Module, view_learner: nn.Module, edge_rnn: nn.Module,
                                   sample_time_encoder: nn.Module, neighbor_sampler: NeighborSampler,
                                   evaluate_idx_data_loader: DataLoader, evaluate_neg_edge_sampler: NegativeEdgeSampler,
                                   evaluate_data: Data, loss_func: nn.Module, num_neighbors: int = 20, time_gap: int = 2000):
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()
    model[0].set_neighbor_sampler(neighbor_sampler)
    model.eval()
    view_learner.eval()
    edge_rnn.eval()
    sample_time_encoder.eval()

    with torch.no_grad():
        evaluate_losses, evaluate_metrics = [], []
        train_edge_feat = view_learner(g)
        full_edge_feat = torch.zeros(edge_raw_features.shape[0] + 1, train_edge_feat.shape[1], device=device)
        full_edge_feat[train_data.edge_ids - 1] = train_edge_feat
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader):
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]

            _, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_src_node_ids))
            batch_neg_src_node_ids = batch_src_node_ids

            train_node_cut = np.array(list(set(np.append(batch_src_node_ids, batch_dst_node_ids)).intersection(train_data.unique_node_ids)))
            if len(train_node_cut) != 0:
                # control rnn/ candidate sample, cannot change to val/test, or data leak
                max_ts = np.array([max_train_ts_l + 1] * len(train_node_cut))

                neighbor_finder = NeighborSampler(adj_list, sample_neighbor_strategy='uniform')
                neighbor_node_idx, neighbor_edge_idx, neighbor_ts = neighbor_finder.get_historical_neighbors(train_node_cut, max_ts, num_neighbors=NUM_RNN_NN)

                neighbor_edge_idx = neighbor_edge_idx.reshape(-1) - 1
                neighbor_edge_feat = full_edge_feat[neighbor_edge_idx]  # [bs * 20, 172]
                neighbor_edge_feat = neighbor_edge_feat.reshape(neighbor_node_idx.shape[0], neighbor_node_idx.shape[1], -1)  # [bs, 20, 172]

                neighbor_edge_feat = neighbor_edge_feat.transpose(0, 1)
                _, (h_n, _) = edge_rnn(neighbor_edge_feat)
                context_vec = h_n[-1]  # [bs, 172]

                neighbor_finder.sample_neighbor_strategy = 'uniform'
                if CAN_TYPE == '1st':
                    candidate_node_idx, candidate_edge_idx, candidate_ts = neighbor_finder.get_historical_neighbors(train_node_cut, max_ts, num_neighbors=NUM_CAN_NN)

                    src_node_idx_aug = np.repeat(train_node_cut.reshape(train_node_cut.shape[0], 1), candidate_node_idx.shape[1], axis=1)  # [bs, 20]
                    dst_node_idx_aug = candidate_node_idx  # [bs, 20]
                elif CAN_TYPE == '3rd':
                    candidate_node_idx, candidate_edge_idx, candidate_ts = neighbor_finder.get_multi_hop_neighbors(3, train_node_cut, max_ts, num_neighbors=NUM_CAN_NN)
                    candidate_node_idx = candidate_node_idx[-1].reshape(train_node_cut.shape[0], -1)
                    candidate_edge_idx = candidate_edge_idx[-1].reshape(train_node_cut.shape[0], -1)
                    candidate_ts = candidate_ts[-1].reshape(train_node_cut.shape[0], -1)

                    src_node_idx_aug = np.repeat(train_node_cut.reshape(train_node_cut.shape[0], 1), candidate_node_idx.shape[1], axis=1)  # [bs, 20]
                    dst_node_idx_aug = candidate_node_idx  # [bs, 20]
                elif CAN_TYPE == 'random':
                    candidate_node_idx = np.random.choice(np.array(list(train_data.unique_node_ids)), size=train_node_cut.shape[0] * NUM_CAN_NN, replace=True).reshape(train_node_cut.shape[0], -1)  # [bs, 20]
                    candidate_edge_idx = np.array([0] * (train_node_cut.shape[0] * NUM_CAN_NN)).reshape(train_node_cut.shape[0], -1)
                    candidate_ts = np.random.rand(train_node_cut.shape[0], NUM_CAN_NN) * max_train_ts_l

                    src_node_idx_aug = np.repeat(train_node_cut.reshape(train_node_cut.shape[0], 1), candidate_node_idx.shape[1], axis=1)  # [bs, 20]
                    dst_node_idx_aug = candidate_node_idx  # [bs, 20]
                elif CAN_TYPE == 'mix':
                    candidate_node_idx_1st, candidate_edge_idx_1st, candidate_ts_1st = neighbor_finder.get_historical_neighbors(train_node_cut, max_ts, num_neighbors=NUM_CAN_NN)
                    candidate_node_idx_3rd, candidate_edge_idx_3rd, candidate_ts_3rd = neighbor_finder.get_multi_hop_neighbors(3, train_node_cut, max_ts, num_neighbors=NUM_CAN_NN)
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

                temp_eid = edge_raw_features.shape[0]
                new_eid_list = []
                adj_list_aug = pickle.loads(full_adj_list_pickle)
                for src_aug, dst_aug, ts_aug_temp in zip(src_node_idx_aug, dst_node_idx_aug, ts_aug):
                    adj_list_aug[src_aug].append((dst_aug, temp_eid, ts_aug_temp))
                    adj_list_aug[dst_aug].append((src_aug, temp_eid, ts_aug_temp))
                    new_eid_list.append(temp_eid)
                    temp_eid += 1

                train_ngh_finder_aug = NeighborSampler(adj_list_aug, sample_neighbor_strategy=model[0].neighbor_sampler.sample_neighbor_strategy, seed=model[0].neighbor_sampler.seed)

                new_eid_list = np.array(new_eid_list)
                full_aug_edge_weight = torch.ones(temp_eid, device=device)
                full_aug_edge_weight[new_eid_list - 1] = aug_edge_weight

                candidate_edge_feat = candidate_edge_feat.reshape(-1, candidate_edge_feat.shape[2]).detach().cpu().numpy()  # [bs * k, 172]
                e_feat_aug = np.concatenate((edge_raw_features, candidate_edge_feat), axis=0)
                edge_raw_embed_aug = torch.from_numpy(e_feat_aug.astype(np.float32)).to(device)

                ngh_finder_ori = model[0].neighbor_sampler
                model[0].set_neighbor_sampler(train_ngh_finder_aug)
                edge_raw_embed_ori = model[0].edge_raw_features
                model[0].edge_raw_features = edge_raw_embed_aug

                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap,
                                                                      full_aug_edge_weight=full_aug_edge_weight)

                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap,
                                                                      full_aug_edge_weight=full_aug_edge_weight)

                # Recover
                model[0].set_neighbor_sampler(ngh_finder_ori)
                model[0].edge_raw_features = edge_raw_embed_ori
            else:
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)

                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)

            # get positive and negative probabilities, shape (batch_size, )
            positive_probabilities = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
            negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

            predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
            labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)], dim=0)

            loss = loss_func(input=predicts, target=labels)
            evaluate_losses.append(loss.item())
            evaluate_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))
            # evaluate_idx_data_loader_tqdm.set_description(f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

    return evaluate_losses, evaluate_metrics


if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_link_prediction_args(is_evaluation=False)

    TAU = args.tau
    GTAU = args.gtau
    K = args.K
    COE = args.coe
    RATIO = args.ratio
    INFER_BS = args.infer_bs
    NUM_CAN_NN = args.can_nn
    NUM_RNN_NN = args.rnn_nn
    CAN_TYPE = args.can_type
    NUM_RNN_LAYER = args.rnn_layer
    LOG_NAME = args.log_name

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data = \
        get_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler, adj_list = get_neighbor_sampler(data=train_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler, full_adj_list = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    adj_list_pickle = pickle.dumps(adj_list, -1)
    full_adj_list_pickle = pickle.dumps(full_adj_list, -1)

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
    train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids)
    val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0)
    new_node_val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_val_data.src_node_ids, dst_node_ids=new_node_val_data.dst_node_ids, seed=1)
    test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2)
    new_node_test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=new_node_test_data.src_node_ids, dst_node_ids=new_node_test_data.dst_node_ids, seed=3)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_val_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)
    new_node_test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(new_node_test_data.src_node_ids))), batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, new_node_val_metric_all_runs, test_metric_all_runs, new_node_test_metric_all_runs = [], [], [], []

    device = torch.device(args.device)
    max_train_ts_l = max(train_data.node_interact_times)

    for run in range(args.num_runs):
        set_random_seed(seed=2023 + run)
        args.save_model_name = f'{args.model_name}_seed{2023 + run}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler("./logs/{}.log".format(LOG_NAME))
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        '''
        +++++++++++++++++++++++++++++ MTL Stage +++++++++++++++++++++++++++++
        '''
        # DGL Graph Construction
        g = dgl.graph((train_data.src_node_ids, train_data.dst_node_ids))

        ndata = []
        for ind in range(g.num_nodes()):
            if ind in train_data.unique_node_ids:
                ndata.append(node_raw_features[ind])
            else:
                ndata.append([0] * node_raw_features.shape[1])

        edata_feat = edge_raw_features[train_data.edge_ids]
        edata_ts = train_data.node_interact_times.reshape(train_data.node_interact_times.shape[0], -1)

        g.ndata['feat'] = torch.tensor(np.array(ndata), dtype=torch.float32)
        g.edata['edge_feat'] = torch.tensor(np.array(edata_feat), dtype=torch.float32)
        g.edata['ts'] = torch.tensor(np.array(edata_ts), dtype=torch.float32)

        g = dgl.add_self_loop(g)
        g = dgl.add_reverse_edges(g, copy_ndata=True, copy_edata=True)
        g = g.to(device)

        dynamic_backbone_k = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                      neighbor_sampler=train_neighbor_sampler,
                                      time_feat_dim=args.time_feat_dim, num_tokens=args.num_neighbors,
                                      num_layers=args.num_layers, dropout=0.0, device=args.device)

        link_predictor_k = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                    hidden_dim=node_raw_features.shape[1], output_dim=1)
        model_k = nn.Sequential(dynamic_backbone_k, link_predictor_k)
        dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                      neighbor_sampler=train_neighbor_sampler, time_feat_dim=args.time_feat_dim,
                                      num_tokens=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout,
                                      device=args.device)

        link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                    hidden_dim=node_raw_features.shape[1], output_dim=1)
        model = nn.Sequential(dynamic_backbone, link_predictor)
        view_learner = ETGNN(in_dim=node_raw_features.shape[1], hidden_dim=node_raw_features.shape[1],
                             train_src_l=train_data.src_node_ids, train_dst_l=train_data.dst_node_ids)
        edge_rnn = torch.nn.LSTM(input_size=node_raw_features.shape[1], hidden_size=node_raw_features.shape[1],
                                 num_layers=NUM_RNN_LAYER, bidirectional=False)
        sample_time_encoder = TimeMapping()
        mtl = MTL(base_encoder_k=model_k, encoder=model, view_learner=view_learner, edge_rnn=edge_rnn,
                sample_time_encoder=sample_time_encoder,
                len_full_edge=edge_raw_features.shape[0], train_e_idx_l=train_data.edge_ids,
                train_node_set=train_data.unique_node_ids,
                train_ts_l=train_data.node_interact_times, e_feat=edge_raw_features, device=device, K=K,
                ratio=RATIO, can_nn=NUM_CAN_NN, rnn_nn=NUM_RNN_NN, can_type=CAN_TYPE, tau=TAU, gtau=GTAU)
        mtl = mtl.to(device)

        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        optimizer = create_optimizer(model=mtl, optimizer_name=args.optimizer, learning_rate=args.learning_rate, weight_decay=args.weight_decay)
        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=3, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name, prefix=args.prefix)

        loss_func = nn.BCELoss().to(device)
        ssl_criterion = torch.nn.CrossEntropyLoss().to(device)

        for epoch in range(args.num_epochs):
            mtl.train()
            mtl.encoder[0].set_neighbor_sampler(train_neighbor_sampler)
            train_losses, train_metrics = [], []
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader):
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                    train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]

                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

                pos_prob, neg_prob, pos_prob_ed, neg_prob_ed, output, target = mtl(batch_src_node_ids, batch_dst_node_ids, batch_neg_src_node_ids, batch_neg_dst_node_ids, batch_node_interact_times, batch_edge_ids, args.num_neighbors, args.time_gap, g, adj_list, adj_list_pickle)
                predicts = torch.cat([pos_prob, neg_prob], dim=0)
                predicts_ed = torch.cat([pos_prob_ed, neg_prob_ed], dim=0)
                labels = torch.cat([torch.ones_like(pos_prob), torch.zeros_like(neg_prob)], dim=0)

                loss = loss_func(input=predicts, target=labels)
                loss += loss_func(input=predicts_ed, target=labels)
                loss += ssl_criterion(output, target) * COE

                train_losses.append(loss.item())
                train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            val_losses, val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                     model=model,
                                                                     view_learner=view_learner,
                                                                     edge_rnn=edge_rnn,
                                                                     sample_time_encoder=sample_time_encoder,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=loss_func,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap)

            new_node_val_losses, new_node_val_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                       model=model,
                                                                                       view_learner=view_learner,
                                                                                       edge_rnn=edge_rnn,
                                                                                       sample_time_encoder=sample_time_encoder,
                                                                                       neighbor_sampler=full_neighbor_sampler,
                                                                                       evaluate_idx_data_loader=new_node_val_idx_data_loader,
                                                                                       evaluate_neg_edge_sampler=new_node_val_neg_edge_sampler,
                                                                                       evaluate_data=new_node_val_data,
                                                                                       loss_func=loss_func,
                                                                                       num_neighbors=args.num_neighbors,
                                                                                       time_gap=args.time_gap)

            logger.info(f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
            for metric_name in train_metrics[0].keys():
                logger.info(f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                logger.info(f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')
            logger.info(f'new node validate loss: {np.mean(new_node_val_losses):.4f}')
            for metric_name in new_node_val_metrics[0].keys():
                logger.info(f'new node validate {metric_name}, {np.mean([new_node_val_metric[metric_name] for new_node_val_metric in new_node_val_metrics]):.4f}')

            # perform testing once after test_interval_epochs
            if (epoch + 1) % args.test_interval_epochs == 0:
                test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                           model=model,
                                                                           view_learner=view_learner,
                                                                           edge_rnn=edge_rnn,
                                                                           sample_time_encoder=sample_time_encoder,
                                                                           neighbor_sampler=full_neighbor_sampler,
                                                                           evaluate_idx_data_loader=test_idx_data_loader,
                                                                           evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                           evaluate_data=test_data,
                                                                           loss_func=loss_func,
                                                                           num_neighbors=args.num_neighbors,
                                                                           time_gap=args.time_gap)

                new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                             model=model,
                                                                                             view_learner=view_learner,
                                                                                             edge_rnn=edge_rnn,
                                                                                             sample_time_encoder=sample_time_encoder,
                                                                                             neighbor_sampler=full_neighbor_sampler,
                                                                                             evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                             evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                             evaluate_data=new_node_test_data,
                                                                                             loss_func=loss_func,
                                                                                             num_neighbors=args.num_neighbors,
                                                                                             time_gap=args.time_gap)

                logger.info(f'test loss: {np.mean(test_losses):.4f}')
                for metric_name in test_metrics[0].keys():
                    logger.info(f'test {metric_name}, {np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}')
                logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
                for metric_name in new_node_test_metrics[0].keys():
                    logger.info(f'new node test {metric_name}, {np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics]):.4f}')

            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics[0].keys():
                if metric_name == 'acc' or metric_name == 'roc_auc':
                    continue
                val_metric_indicator.append((metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
            early_stop = early_stopping.step(val_metric_indicator, mtl)

            if early_stop:
                break

        # load the best model
        early_stopping.load_checkpoint(mtl)

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        # the saved best model of memory-based models cannot perform validation since the stored memory has been updated by validation data
        test_losses, test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                   model=model,
                                                                   view_learner=view_learner,
                                                                   edge_rnn=edge_rnn,
                                                                   sample_time_encoder=sample_time_encoder,
                                                                   neighbor_sampler=full_neighbor_sampler,
                                                                   evaluate_idx_data_loader=test_idx_data_loader,
                                                                   evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                   evaluate_data=test_data,
                                                                   loss_func=loss_func,
                                                                   num_neighbors=args.num_neighbors,
                                                                   time_gap=args.time_gap)

        new_node_test_losses, new_node_test_metrics = evaluate_model_link_prediction(model_name=args.model_name,
                                                                                     model=model,
                                                                                     view_learner=view_learner,
                                                                                     edge_rnn=edge_rnn,
                                                                                     sample_time_encoder=sample_time_encoder,
                                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                                     evaluate_idx_data_loader=new_node_test_idx_data_loader,
                                                                                     evaluate_neg_edge_sampler=new_node_test_neg_edge_sampler,
                                                                                     evaluate_data=new_node_test_data,
                                                                                     loss_func=loss_func,
                                                                                     num_neighbors=args.num_neighbors,
                                                                                     time_gap=args.time_gap)
        # store the evaluation metrics at the current run
        val_metric_dict, new_node_val_metric_dict, test_metric_dict, new_node_test_metric_dict = {}, {}, {}, {}

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        logger.info(f'new node test loss: {np.mean(new_node_test_losses):.4f}')
        for metric_name in new_node_test_metrics[0].keys():
            average_new_node_test_metric = np.mean([new_node_test_metric[metric_name] for new_node_test_metric in new_node_test_metrics])
            logger.info(f'new node test {metric_name}, {average_new_node_test_metric:.4f}')
            new_node_test_metric_dict[metric_name] = average_new_node_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        test_metric_all_runs.append(test_metric_dict)
        new_node_test_metric_all_runs.append(new_node_test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        result_json = {
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
            "new node test metrics": {metric_name: f'{new_node_test_metric_dict[metric_name]:.4f}' for metric_name in new_node_test_metric_dict}
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                    f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    for metric_name in new_node_test_metric_all_runs[0].keys():
        logger.info(f'new node test {metric_name}, {[new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]}')
        logger.info(f'average new node test {metric_name}, {np.mean([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs]):.4f} '
                    f'± {np.std([new_node_test_metric_single_run[metric_name] for new_node_test_metric_single_run in new_node_test_metric_all_runs], ddof=1):.4f}')

    sys.exit()
