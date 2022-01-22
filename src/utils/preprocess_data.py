import argparse
import os
import sys

sys.path.append(os.path.join(sys.path[0], ".."))
import numpy as np
import pandas as pd

from path_manager.path_manager import DataPathManager


def preprocessA(PATH_EDGES, PATH_NODE_FEATS, PATH_EDGE_FEATS, model="train", init_last_update="min_time"):
    """
    preprocess dataA
    map node id to [1:max_node_id]; id 0 represent null
    map edge type id to [1:max_edge_type_id]; id 0 represent null
    :param PATH_EDGES:
    :param PATH_NODE_FEATS:
    :param PATH_EDGE_FEATS:
    :return:
     edges_np[num_edge,4]:(src_node,dest_node,edge_type,timestamp),dtype=int32
     node_features_np[max_node_id+1,num_node_feats]: dtype=float64
     edge_type_feat_np[max_type_id+1,num_edge_feats]:dtype=float64
    """

    # preprocess node
    node_features_csv = pd.read_csv(PATH_NODE_FEATS, header=None)
    # get a map : old_node_id to new_node_id
    node_id_map = np.zeros(max(node_features_csv[0]) + 1, dtype=np.int32)
    for new_node_id, old_node_id in enumerate(node_features_csv[0], start=1):
        node_id_map[old_node_id] = new_node_id
    # node feats: index of 0 is null; index from 1 to node_num is valid
    empty = np.zeros(node_features_csv.shape[1] - 1)[np.newaxis, :]
    node_features_np = np.vstack([empty, np.array(node_features_csv.loc[:, 1:])])

    # preprocess edges_feats
    # edge type feats: index of 0 is null; index from 1 to num_edge_type is valid
    edge_type_feat_csv = pd.read_csv(PATH_EDGE_FEATS, header=None)
    edge_type_feat_csv.sort_values(by=0, inplace=True)
    empty = np.zeros(edge_type_feat_csv.shape[1] - 1)[np.newaxis, :]
    edge_type_feat_np = np.vstack([empty, np.array(edge_type_feat_csv.loc[:, 1:])])

    # preprocess edges
    edges_csv = pd.read_csv(PATH_EDGES, header=None)
    # if model == "train":
    #     # only sort for train data
    #     edges_csv.sort_values(by=3, inplace=True)
    # transformer old edge type if to new edge type id
    edges_csv[2] += 1
    # transformer old node id to new node id
    edges_csv[0] = edges_csv[0].map(lambda x: node_id_map[x])
    edges_csv[1] = edges_csv[1].map(lambda x: node_id_map[x])
    # split the edges
    # 0: src, 1: dst, 2: relation type, 3: timestamp, 4: edge idx, 5: label
    if model != "train":
        u_list, v_list, type_list, ts_list, id_list, label_list = [], [], [], [], [], []
        for id, row in edges_csv.iterrows():
            assert row[4] + 1 > row[3] + (3600 - row[3] % 3600) % 3600, "error"
            for split_time in range(row[3] + (3600 - row[3] % 3600) % 3600, row[4] + 1, 3600):
                u_list.append(row[0])
                v_list.append(row[1])
                type_list.append(row[2])
                ts_list.append(split_time)
                id_list.append(id)
                if model == "test":
                    label_list.append(row[5])
        if model == "test":
            return np.array([u_list, v_list, type_list, ts_list, id_list, label_list], dtype=np.int32).T
        else:
            return np.array([u_list, v_list, type_list, ts_list, id_list], dtype=np.int32).T

    edges_np = np.array(edges_csv, dtype=np.int32)
    ## add time diff col
    if init_last_update == "zero":
        last_update = np.zeros(node_features_np.shape[0], dtype=np.int32)
    elif init_last_update == "min_time":
        last_update = np.ones(node_features_np.shape[0], dtype=np.int32)
        last_update *= min(edges_np[:, 3])
    elif init_last_update == "fist_occur":
        last_update = np.ones(node_features_np.shape[0], dtype=np.int32)
        last_update *= -1
    else:
        assert 1 == 0, "error init method"
    src_time_diff = []
    dst_time_diff = []
    for row in edges_np:
        if init_last_update == "fist_occur":
            if last_update[row[0]] == -1:
                last_update[row[0]] = row[3]
            if last_update[row[1]] == -1:
                last_update[row[1]] = row[3]
        src_time_diff.append(row[3] - last_update[row[0]])
        dst_time_diff.append(row[3] - last_update[row[1]])
        last_update[row[0]] = row[3]
        last_update[row[1]] = row[3]
    src_time_diff = np.array(src_time_diff).reshape(-1, 1)
    dst_time_diff = np.array(dst_time_diff).reshape(-1, 1)
    edges_np = np.hstack([edges_np, src_time_diff, dst_time_diff])
    print("OK")
    return edges_np, node_features_np, edge_type_feat_np


def preprocessB(PATH_EDGES, PATH_NODE_FEATS, PATH_EDGE_FEATS, model="train", init_last_update="min_time"):
    # preprocess edges
    edges_csv = pd.read_csv(PATH_EDGES, header=None)
    if model == "train":
        # only sort for train data
        # edges_csv.sort_values(by=3, inplace=True)
        edge_feats_csv = edges_csv[4]
        edges_csv = edges_csv.iloc[:, 0:4]
        # edge_feats_csv = edge_feats_csv.str.split(pat=',', expand=True)
        # empty = np.zeros(edge_feats_csv.shape[1])[np.newaxis, :]
        # edge_feats_np = np.vstack([empty, np.array(edge_feats_csv)])
        edge_feats_np = None
    # merge src,dst id namespace
    edges_csv[0] += 1
    edges_csv[1] += (791332 + 1)
    edges_csv[3] //= 3600
    edges_csv[3] *= 3600
    # split the edges
    # 0: src, 1: dst, 2: relation type, 3: timestamp, 4: edge idx, 5: label
    if model != "train":
        edges_csv[4] //= 3600
        edges_csv[4] *= 3600
        u_list, v_list, type_list, ts_list, id_list, label_list = [], [], [], [], [], []
        for id, row in edges_csv.iterrows():
            assert row[4] + 1 > row[3], "error"
            for split_time in range(row[3], row[4] + 1, 3600):
                u_list.append(row[0])
                v_list.append(row[1])
                type_list.append(row[2])
                ts_list.append(split_time)
                id_list.append(id)
                if model == "test":
                    label_list.append(row[5])
        if model == "test":
            return np.array([u_list, v_list, type_list, ts_list, id_list, label_list], dtype=np.int32).T
        else:
            return np.array([u_list, v_list, type_list, ts_list, id_list], dtype=np.int32).T

    edges_np = np.array(edges_csv, dtype=np.int32)
    ## add time diff col
    max_node_id = max(edges_np[1])
    if init_last_update == "zero":
        last_update = np.zeros(max_node_id + 1, dtype=np.int32)
    elif init_last_update == "min_time":
        last_update = np.ones(max_node_id + 1, dtype=np.int32)
        last_update *= min(edges_np[:, 3])
    elif init_last_update == "fist_occur":
        last_update = np.ones(max_node_id + 1, dtype=np.int32)
        last_update *= -1
    else:
        assert 1 == 0, "error init method"
    src_time_diff = []
    dst_time_diff = []
    for row in edges_np:
        if init_last_update == "fist_occur":
            if last_update[row[0]] == -1:
                last_update[row[0]] = row[3]
            if last_update[row[1]] == -1:
                last_update[row[1]] = row[3]
        src_time_diff.append(row[3] - last_update[row[0]])
        dst_time_diff.append(row[3] - last_update[row[1]])
        last_update[row[0]] = row[3]
        last_update[row[1]] = row[3]
    src_time_diff = np.array(src_time_diff).reshape(-1, 1)
    dst_time_diff = np.array(dst_time_diff).reshape(-1, 1)
    edges_np = np.hstack([edges_np, src_time_diff, dst_time_diff])
    print("OK")
    return edges_np, edge_feats_np


def run(data_name, data_type, init_last_update):
    data_path_manager = DataPathManager()
    PATH_EDGES = data_path_manager.get_raw_edges_path(data_name, data_type)
    PATH_NODE_FEATS = data_path_manager.get_raw_node_feats_path(data_name)
    PATH_EDGE_FEATS = data_path_manager.get_raw_edge_feats_path(data_name)
    OUT_EDGES = data_path_manager.get_ml_edges_path(data_name, data_type)
    OUT_EDGE_FEAT = data_path_manager.get_ml_edge_feats_path(data_name)
    OUT_NODE_FEAT = data_path_manager.get_ml_node_feats_path(data_name)
    OUT_TMP_ANS = data_path_manager.get_output_csv_path(data_name)
    if data_name == "A":
        preprocess = preprocessA
    else:
        preprocess = preprocessB
    if data_type == "train":
        np_list = preprocess(PATH_EDGES, PATH_NODE_FEATS, PATH_EDGE_FEATS, model=data_type,
                             init_last_update=init_last_update)
        if data_name == "A":
            edges_np, node_features_np, edge_type_feat_np = np_list
            np.save(OUT_NODE_FEAT, node_features_np)
        else:
            edges_np, edge_type_feat_np = np_list
        np.save(OUT_EDGES, edges_np)
        np.save(OUT_EDGE_FEAT, edge_type_feat_np)
    else:
        edges_np = preprocess(PATH_EDGES, PATH_NODE_FEATS, PATH_EDGE_FEATS, model=data_type)
        np.save(OUT_EDGES, edges_np)
    if data_type == "predict":
        pd.Series(np.random.rand(edges_np[:, 4].max()+1)).to_csv(OUT_TMP_ANS, header=None, index=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
    parser.add_argument('--all', default=False, action="store_true")
    parser.add_argument('--data', type=str, help='Dataset name (eg. A or B)',
                        default='A')
    parser.add_argument('--data_type', type=str, choices=["train", "test", "predict", "final_predict"],
                        help='Dataset name (eg. train, initial, test)', default='test')
    parser.add_argument('--init_last_update', type=str, choices=["zero", "min_time", "fist_occur"],
                        help='Init last update method (eg. train, initial, test)', default='fist_occur')
    args = parser.parse_args()
    if args.all:
        for data in ["A", "B"]:
            for data_type in ["train", "test", "predict", "final_predict"]:
                run(data, data_type, 'fist_occur')
    else:
        run(args.data, args.data_type, args.init_last_update)
