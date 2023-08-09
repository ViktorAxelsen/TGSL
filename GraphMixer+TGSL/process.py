import os
import argparse
import numpy as np
import pandas as pd


def preprocess(path, data_name):
    u_list, i_list, ts_list, label_list = [], [], [], []
    feat_l = []
    idx_list = []

    with open(path) as f:
        s = next(f)
        print(s)
        if data_name == 'escorts':
            for _ in range(5):
                s = next(f)
                print(s)
        previous_time = -1
        for idx, line in enumerate(f):
            if data_name == 'escorts':
                e = line.strip().split()
                u = int(e[0])
                i = int(e[1])
                ts = float(e[3])
                assert ts >= previous_time
                previous_time = ts
                label = int(e[2])
                feat = np.zeros(172)
            else:
                e = line.strip().split(',')
                u = int(e[0])
                i = int(e[1])
                ts = float(e[2])
                label = int(e[3])
                feat = np.array([float(x) for x in e[4:]])

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)

            feat_l.append(feat)
    return pd.DataFrame({'u': u_list,
                         'i': i_list,
                         'ts': ts_list,
                         'label': label_list,
                         'idx': idx_list}), np.array(feat_l)


def reindex(df):
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1
    new_i = df.i + upper_u

    new_df = df.copy()
    print(new_df.u.max())
    print(new_df.i.max())

    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

    print(new_df.u.max())
    print(new_df.i.max())

    return new_df


def run(data_name):
    if data_name == 'escorts':
        PATH = './data/{}.edges'.format(data_name)
    else:
        PATH = './data/{}.csv'.format(data_name)
    OUT_DF = './processed_data/{}/ml_{}.csv'.format(data_name, data_name)
    OUT_FEAT = './processed_data/{}/ml_{}.npy'.format(data_name, data_name)
    OUT_NODE_FEAT = './processed_data/{}/ml_{}_node.npy'.format(data_name, data_name)

    df, feat = preprocess(PATH, data_name)
    new_df = reindex(df)

    print(feat.shape)
    empty = np.zeros(feat.shape[1])[np.newaxis, :]
    feat = np.vstack([empty, feat])

    max_idx = max(new_df.u.max(), new_df.i.max())
    rand_feat = np.zeros((max_idx + 1, feat.shape[1]))

    print(feat.shape)
    new_df.to_csv(OUT_DF)
    np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, rand_feat)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset', choices=['wikipedia', 'reddit', 'escorts'], default='wikipedia')
    args = parser.parse_args()
    os.makedirs("./processed_data/{}/".format(args.dataset), exist_ok=True)
    run(args.dataset)