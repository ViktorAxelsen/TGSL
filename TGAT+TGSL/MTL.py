import torch
import torch.nn as nn
import numpy as np
import pickle

from neighbor_finder import NeighborFinder


class MTL(nn.Module):
    def __init__(self, base_encoder_k, encoder, view_learner, edge_rnn, sample_time_encoder, len_full_edge,
                 train_e_idx_l, train_node_set, train_ts_l, e_feat, device, dim=172, K=600, m=0.999, tau=0.1, gtau=1.0,
                 ratio=0.9, can_nn=20, rnn_nn=20, can_type='3rd'):
        super(MTL, self).__init__()
        self.K = K
        self.m = m
        self.tau = tau
        self.gtau = gtau
        self.ratio = ratio
        self.can_nn = can_nn
        self.rnn_nn = rnn_nn
        self.can_type = can_type

        self.encoder_k = base_encoder_k
        self.encoder = encoder
        self.view_learner = view_learner
        self.edge_rnn = edge_rnn
        self.sample_time_encoder = sample_time_encoder

        self.e_feat = e_feat
        self.len_full_edge = len_full_edge
        self.train_e_idx_l = train_e_idx_l
        self.train_node_set = np.array(list(train_node_set))
        self.train_ts_l = train_ts_l
        self.max_train_ts_l = max(train_ts_l)
        self.device = device

        for param_q, param_k in zip(
            self.encoder.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(
            self.encoder.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if ptr + batch_size > self.K:
            keys_prev = keys[:self.K - ptr]
            keys_aft = keys[self.K - ptr:]
            self.queue[:, ptr: self.K] = keys_prev.T
            self.queue[:, : len(keys_aft)] = keys_aft.T
            ptr = len(keys_aft)
            self.queue_ptr[0] = ptr
        else:
            self.queue[:, ptr : ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K  # move pointer
            self.queue_ptr[0] = ptr

    def forward(self, src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS, g, adj_list, adj_list_pickle):
        src_embed = self.encoder.tem_conv(src_l_cut, ts_l_cut, 2, NUM_NEIGHBORS)
        target_embed = self.encoder.tem_conv(dst_l_cut, ts_l_cut, 2, NUM_NEIGHBORS)
        background_embed = self.encoder.tem_conv(dst_l_fake, ts_l_cut, 2, NUM_NEIGHBORS)
        pos_score = self.encoder.affinity_score(src_embed, target_embed).squeeze(dim=-1)
        neg_score = self.encoder.affinity_score(src_embed, background_embed).squeeze(dim=-1)

        train_node_cut = np.array(list(set(np.append(src_l_cut, dst_l_cut))))
        max_ts = np.array([self.max_train_ts_l + 1] * len(train_node_cut))

        neighbor_finder = NeighborFinder(adj_list, uniform=True)
        neighbor_node_idx, neighbor_edge_idx, neighbor_ts = neighbor_finder.get_temporal_neighbor(train_node_cut, max_ts, num_neighbors=self.rnn_nn)

        train_edge_feat = self.view_learner(g)
        full_edge_feat = torch.zeros(self.len_full_edge + 1, train_edge_feat.shape[1], device=self.device)
        full_edge_feat[self.train_e_idx_l - 1] = train_edge_feat

        neighbor_edge_idx = neighbor_edge_idx.reshape(-1) - 1
        neighbor_edge_feat = full_edge_feat[neighbor_edge_idx]
        neighbor_edge_feat = neighbor_edge_feat.reshape(neighbor_node_idx.shape[0], neighbor_node_idx.shape[1], -1)

        neighbor_edge_feat = neighbor_edge_feat.transpose(0, 1)
        _, (h_n, _) = self.edge_rnn(neighbor_edge_feat)
        context_vec = h_n[-1]

        neighbor_finder.uniform = True
        if self.can_type == '1st':
            candidate_node_idx, candidate_edge_idx, candidate_ts = neighbor_finder.get_temporal_neighbor(train_node_cut, max_ts, num_neighbors=self.can_nn)

            src_node_idx_aug = np.repeat(train_node_cut.reshape(train_node_cut.shape[0], 1), candidate_node_idx.shape[1], axis=1)  # [bs, 20]
            dst_node_idx_aug = candidate_node_idx  # [bs, 20]
        elif self.can_type == '3rd':
            candidate_node_idx, candidate_edge_idx, candidate_ts = neighbor_finder.find_k_hop(3, train_node_cut, max_ts, num_neighbors=self.can_nn)
            candidate_node_idx = candidate_node_idx[-1].reshape(train_node_cut.shape[0], -1)
            candidate_edge_idx = candidate_edge_idx[-1].reshape(train_node_cut.shape[0], -1)
            candidate_ts = candidate_ts[-1].reshape(train_node_cut.shape[0], -1)

            src_node_idx_aug = np.repeat(train_node_cut.reshape(train_node_cut.shape[0], 1), candidate_node_idx.shape[1], axis=1)  # [bs, 20]
            dst_node_idx_aug = candidate_node_idx  # [bs, 20]
        elif self.can_type == 'random':
            candidate_node_idx = np.random.choice(self.train_node_set, size=train_node_cut.shape[0] * self.can_nn, replace=True).reshape(train_node_cut.shape[0], -1)  # [bs, 20]
            candidate_edge_idx = np.array([0] * (train_node_cut.shape[0] * self.can_nn)).reshape(train_node_cut.shape[0], -1)
            candidate_ts = np.random.rand(train_node_cut.shape[0], self.can_nn) * self.max_train_ts_l

            src_node_idx_aug = np.repeat(train_node_cut.reshape(train_node_cut.shape[0], 1), candidate_node_idx.shape[1], axis=1)  # [bs, 20]
            dst_node_idx_aug = candidate_node_idx  # [bs, 20]
        elif self.can_type == 'mix':
            candidate_node_idx_1st, candidate_edge_idx_1st, candidate_ts_1st = neighbor_finder.get_temporal_neighbor(train_node_cut, max_ts, num_neighbors=self.can_nn)
            candidate_node_idx_3rd, candidate_edge_idx_3rd, candidate_ts_3rd = neighbor_finder.find_k_hop(3, train_node_cut, max_ts, num_neighbors=self.can_nn)
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
        candidate_edge_feat = candidate_edge_feat.reshape(candidate_node_idx.shape[0], candidate_node_idx.shape[1], -1)

        ts_aug = np.random.rand(candidate_ts.shape[0], candidate_ts.shape[1]) * self.max_train_ts_l
        delta_ts_sample = ts_aug - candidate_ts
        delta_ts_sample_context = ts_aug - np.ones_like(candidate_ts) * self.max_train_ts_l
        delta_ts_sample_embedding = self.sample_time_encoder(torch.tensor(delta_ts_sample.reshape(-1, 1), dtype=torch.float32).to(self.device)).reshape(ts_aug.shape[0], ts_aug.shape[1], -1)
        delta_ts_sample_context_embedding = self.sample_time_encoder(torch.tensor(delta_ts_sample_context.reshape(-1, 1), dtype=torch.float32).to(self.device)).reshape(ts_aug.shape[0], ts_aug.shape[1], -1)

        context_vec = context_vec.unsqueeze(1).expand_as(candidate_edge_feat)
        context_vec = context_vec * delta_ts_sample_context_embedding
        candidate_edge_feat = candidate_edge_feat * delta_ts_sample_embedding
        aug_edge_logits = torch.sum(context_vec * candidate_edge_feat, dim=-1)  # [bs, 20, 1]

        # Gumble-Top-K
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(aug_edge_logits.size()) + (1 - bias)
        gate_inputs = torch.log(eps) - torch.log(1 - eps)
        gate_inputs = gate_inputs.to(aug_edge_logits.device)
        gate_inputs = (gate_inputs + aug_edge_logits) / self.gtau
        z = torch.sigmoid(gate_inputs).squeeze()  # [bs, 20]
        __, sorted_idx = z.sort(dim=-1, descending=True)
        k = int(self.ratio * z.size(1))
        keep = sorted_idx[:, :k]  # [bs, k]

        aug_edge_logits = torch.sigmoid(gate_inputs).squeeze()  # [bs, 20]
        aug_edge_weight = torch.gather(aug_edge_logits, dim=1, index=keep)  # [bs, k]
        ts_aug = torch.gather(torch.tensor(ts_aug, device=self.device), dim=1, index=keep).detach().cpu().numpy()  # [bs, k]
        src_node_idx_aug = torch.gather(torch.tensor(src_node_idx_aug, device=self.device), dim=1, index=keep).detach().cpu().numpy()  # [bs, k]
        dst_node_idx_aug = torch.gather(torch.tensor(dst_node_idx_aug, device=self.device), dim=1, index=keep).detach().cpu().numpy()  # [bs, k]
        candidate_edge_feat = torch.gather(candidate_edge_feat, dim=1, index=keep.unsqueeze(2).repeat(1, 1, candidate_edge_feat.shape[2]))  # [bs, k, 172]

        aug_edge_weight = aug_edge_weight.reshape(-1)
        ts_aug = ts_aug.reshape(-1)
        src_node_idx_aug = src_node_idx_aug.reshape(-1)
        dst_node_idx_aug = dst_node_idx_aug.reshape(-1)

        temp_eid = self.len_full_edge
        new_eid_list = []
        adj_list_aug = pickle.loads(adj_list_pickle)
        for src, dst, ts in zip(src_node_idx_aug, dst_node_idx_aug, ts_aug):
            adj_list_aug[src].append((dst, temp_eid, ts))
            adj_list_aug[dst].append((src, temp_eid, ts))
            new_eid_list.append(temp_eid)
            temp_eid += 1

        train_ngh_finder_aug = NeighborFinder(adj_list_aug, uniform=self.encoder.ngh_finder.uniform)

        new_eid_list = np.array(new_eid_list)
        full_aug_edge_weight = torch.ones(temp_eid, device=self.device)
        full_aug_edge_weight[new_eid_list - 1] = aug_edge_weight

        candidate_edge_feat = candidate_edge_feat.reshape(-1, candidate_edge_feat.shape[2]).detach().cpu().numpy()  # [bs * k, 172]
        e_feat_aug = np.concatenate((self.e_feat, candidate_edge_feat), axis=0)
        e_feat_th_aug = torch.nn.Parameter(torch.from_numpy(e_feat_aug.astype(np.float32)))
        edge_raw_embed_aug = torch.nn.Embedding.from_pretrained(e_feat_th_aug, padding_idx=0, freeze=True).to(self.device)

        ngh_finder_ori = self.encoder.ngh_finder
        self.encoder.ngh_finder = train_ngh_finder_aug
        edge_raw_embed_ori = self.encoder.edge_raw_embed
        self.encoder.edge_raw_embed = edge_raw_embed_aug

        max_ts = np.array([self.max_train_ts_l + 1] * len(train_node_cut))

        src_embed_ed = self.encoder.tem_conv(src_l_cut, ts_l_cut, 2, NUM_NEIGHBORS, full_aug_edge_weight)
        target_embed_ed = self.encoder.tem_conv(dst_l_cut, ts_l_cut, 2, NUM_NEIGHBORS, full_aug_edge_weight)
        background_embed_ed = self.encoder.tem_conv(dst_l_fake, ts_l_cut, 2, NUM_NEIGHBORS, full_aug_edge_weight)
        pos_score_ed = self.encoder.affinity_score(src_embed_ed, target_embed_ed).squeeze(dim=-1)
        neg_score_ed = self.encoder.affinity_score(src_embed_ed, background_embed_ed).squeeze(dim=-1)

        q = self.encoder.tem_conv(train_node_cut, max_ts, 2, 20, full_aug_edge_weight)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # Recover
        self.encoder.ngh_finder = ngh_finder_ori
        self.encoder.edge_raw_embed = edge_raw_embed_ori

        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.encoder_k.tem_conv(train_node_cut, max_ts, 2, 20)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.tau
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        self._dequeue_and_enqueue(k)

        return pos_score.sigmoid(), neg_score.sigmoid(), pos_score_ed.sigmoid(), neg_score_ed.sigmoid(), logits, labels
