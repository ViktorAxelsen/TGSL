import numpy as np
import torch
import torch.nn as nn


class GraphMixerTE(torch.nn.Module):
    def __init__(self, time_dim):
        super(GraphMixerTE, self).__init__()
        self.w = nn.Parameter(torch.tensor([int(np.sqrt(time_dim)) ** (-(i - 1) / int(np.sqrt(time_dim))) for i in range(1, time_dim + 1)]), requires_grad=False)
        self.transformation = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.ReLU(inplace=True),
            nn.Linear(time_dim, time_dim)
        )

    def forward(self, ts):
        map_ts = ts * self.w.view(1, -1)
        harmonic = torch.sin(map_ts)
        harmonic = self.transformation(harmonic)

        return harmonic


class TimeMapping(torch.nn.Module):
    def __init__(self, time_dim=172):
        super(TimeMapping, self).__init__()
        self.w = nn.Parameter(torch.tensor([int(np.sqrt(time_dim)) ** (-(i - 1) / int(np.sqrt(time_dim))) for i in range(1, time_dim + 1)]), requires_grad=False)
        self.transformation = nn.Linear(time_dim, time_dim, bias=False)

    def forward(self, ts):
        map_ts = ts * self.w.view(1, -1)
        harmonic = torch.sin(map_ts)
        harmonic = self.transformation(harmonic)

        return harmonic + 1


def gcn_reduce(nodes):
    selfh = nodes.data['h']
    msgs = torch.mean(nodes.mailbox['h'], dim=1)
    msgs = torch.cat((msgs, selfh), dim=1)

    return {'h': msgs}


def gcn_msg(edges):
    h = torch.cat((edges.src['h'], edges.data['h'], edges.data['ts_enc']), dim=1)

    return {'h': h}


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(NodeApplyModule, self).__init__()
        self.fc = nn.Linear(in_feats, out_feats, bias=True)

    def forward(self, node):
        h = self.fc(node.data['h'])

        return {'h': h}


class EdgeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(EdgeApplyModule, self).__init__()
        self.fc = nn.Linear(in_feats * 4, out_feats, bias=True)

    def forward(self, edge):
        h = torch.cat((edge.src['h'], edge.data['h'], edge.data['ts_enc'], edge.dst['h']), dim=1)
        h = self.fc(h)

        return {'h': h}


class TimeAwareGCN(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(TimeAwareGCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats * 4, out_feats)
        self.apply_mod_e = EdgeApplyModule(in_feats, out_feats)

    def forward(self, g, features, efeatures):
        g.ndata['h'] = features
        g.edata['h'] = efeatures

        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        g.apply_edges(func=self.apply_mod_e)

        return g.ndata.pop('h'), g.edata.pop('h')


class ETGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, train_src_l, train_dst_l, mlp_dim=64, time_dim=172):
        super(ETGNN, self).__init__()
        self.train_src_l = train_src_l
        self.train_dst_l = train_dst_l
        self.gcn1 = TimeAwareGCN(in_dim, hidden_dim)
        self.gcn2 = TimeAwareGCN(hidden_dim, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, 1)
        )
        self.time_encoder = GraphMixerTE(time_dim=time_dim)
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, g):
        g.edata['ts_enc'] = self.time_encoder(g.edata['ts'])
        res, eres = self.gcn1(g, g.ndata['feat'], g.edata['edge_feat'])
        res = self.act(res)
        eres = self.act(eres)
        res, eres = self.gcn2(g, res, eres)
        eres = eres[:len(self.train_src_l), :]

        return eres
