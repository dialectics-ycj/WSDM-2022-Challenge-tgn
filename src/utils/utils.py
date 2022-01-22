import os
import pickle

import numpy as np
import torch

from path_manager.path_manager import DataPathManager


class MergeLayer(torch.nn.Module):
    class MergeLayer(torch.nn.Module):
        """
        merge dim1||dim2 -> dim3 -> dim4
        """

    def __init__(self, dim1, dim2, dim3, dim4):
        super().__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(dim3, dim4)

        torch.nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        torch.nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="relu")

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.act(self.fc1(x))
        return self.fc2(x)


class MLP_Classifier(torch.nn.Module):
    def __init__(self, dim, output=1, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 2 * output)
        self.fc_2 = torch.nn.Linear(2 * output, output)
        self.fc_3 = torch.nn.Linear(output, output)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)


class MLP(torch.nn.Module):
    def __init__(self, dim, output=1, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, 80)
        self.fc_2 = torch.nn.Linear(80, 10)
        self.fc_3 = torch.nn.Linear(10, output)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)


class SingleMLP(torch.nn.Module):
    def __init__(self, dim, output=1, drop=0.3):
        super().__init__()
        self.fc_1 = torch.nn.Linear(dim, dim // 2)
        self.fc_2 = torch.nn.Linear(dim // 2, 10)
        self.fc_3 = torch.nn.Linear(10, output)
        self.act = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop, inplace=False)

    def forward(self, x):
        x = self.act(self.fc_1(x))
        x = self.dropout(x)
        x = self.act(self.fc_2(x))
        x = self.dropout(x)
        return self.fc_3(x).squeeze(dim=1)


class MultiMLP(torch.nn.Module):
    def __init__(self, in_features, n_relation):
        super().__init__()
        self.n_relation = n_relation
        self.weight_1 = torch.nn.Parameter(torch.empty(n_relation, in_features, in_features // 2))
        self.bias_1 = torch.nn.Parameter(torch.empty(n_relation, 1, in_features // 2))
        for i in range(0, n_relation):
            torch.nn.init.kaiming_normal_(self.weight_1[i], mode='fan_in', nonlinearity='relu')
            torch.nn.init.kaiming_uniform_(self.bias_1[i], mode='fan_in', nonlinearity='relu')

        in_features = in_features // 2
        self.weight_2 = torch.nn.Parameter(torch.empty((n_relation, in_features, in_features // 2)))
        self.bias_2 = torch.nn.Parameter(torch.empty(n_relation, 1, in_features // 2))
        for i in range(0, n_relation):
            torch.nn.init.kaiming_normal_(self.weight_2[i], mode='fan_in', nonlinearity='relu')
            torch.nn.init.kaiming_uniform_(self.bias_2[i], mode='fan_in', nonlinearity='relu')

        self.act = torch.nn.ReLU()
        in_features = in_features // 2
        self.weight_3 = torch.nn.Parameter(torch.empty(n_relation, in_features, 1))
        self.bias_3 = torch.nn.Parameter(torch.empty(n_relation, 1, 1))
        for i in range(0, n_relation):
            torch.nn.init.xavier_normal_(self.weight_3[i], gain=1)
            torch.nn.init.xavier_uniform_(self.bias_3[i], gain=1)

    def forward(self, x):
        # x.shape: (b, m0)
        x = torch.matmul(x, self.weight_1)  # broadcast: (k, b, m0) * (k, m0, m1)
        x.add_(self.bias_1)  # broadcast:(k,b,m1) + (k,1,m1)
        x = self.act(x)
        x = torch.matmul(x, self.weight_2)
        x.add_(self.bias_2)
        x = self.act(x)
        x = torch.matmul(x, self.weight_3)
        x.add_(self.bias_3)
        return x.squeeze()  # (k,b)


class EarlyStopMonitor(object):
    """
    对每个epoch的训练结果进行记录
    记录最好的模型所在的epoch
    记录耐受情况
    """

    def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
        """

        :param max_round: 耐受上限
        :param higher_better: 是否值越大越好
        :param tolerance: 变化率
        """
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better  # 是否curr_val越高越好
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:  # 转化为负数后，统一为值越大越好
            curr_val *= -1
        if self.last_best is None:  # 初始化记录的最好值
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:  # 变化率
            self.last_best = curr_val
            self.num_round = 0  # 耐受值归零
            self.best_epoch = self.epoch_count  # 最好模型所在的epoch
        else:
            self.num_round += 1  # 耐受值增1

        self.epoch_count += 1  # epoch 记数

        return self.num_round >= self.max_round


class RandEdgeSampler(object):

    def __init__(self, src_list, dst_list, seed=None):
        self.seed = None
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def super_sample(self, batch_size, sample_times):
        pass

    def sample(self, size):
        if self.seed is None:
            src_index = np.random.randint(0, len(self.src_list), size)
            dst_index = np.random.randint(0, len(self.dst_list), size)
        else:
            src_index = self.random_state.randint(0, len(self.src_list), size)
            dst_index = self.random_state.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


def get_neighbor_finder(data, uniform, max_node_idx=None, seed=None, *, dataName, model):
    """
    从data数据集中统计所有节点(仅限于data中的source、destination节点)的邻接列表
    :param model: :["train","val","full"]
    :param dataName: not None, try load from file and save to file
    :param seed:
    :param data:
    :param uniform:
    :param max_node_idx:
    :return:
    """
    # 确保数据本身有序以尝试加速处理过程
    data_path_manager = DataPathManager()
    neighborFinder_path = data_path_manager.get_ngh_finder_path(dataName, model)
    if os.path.exists(neighborFinder_path):
        return NeighborFinder(None, None, None, None).load(neighborFinder_path)

    max_node_idx = max(data.max_sources_id, data.max_destination_id) + 1 if max_node_idx is None else max_node_idx
    node_to_neighbors = [[] for _ in range(max_node_idx + 1)]
    node_to_edge_idxs = [[] for _ in range(max_node_idx + 1)]
    node_to_edge_timestamps = [[] for _ in range(max_node_idx + 1)]
    node_to_edge_types = [[] for _ in range(max_node_idx + 1)]
    for source, destination, edge_idx, timestamp, relation_type in zip(data.sources, data.destinations,
                                                                       data.edge_idxs,
                                                                       data.timestamps,
                                                                       data.relation_types):
        node_to_neighbors[source].append(destination)
        node_to_edge_idxs[source].append(edge_idx)
        node_to_edge_timestamps[source].append(timestamp)
        node_to_edge_types[source].append(relation_type)

        node_to_neighbors[destination].append(source)
        node_to_edge_idxs[destination].append(edge_idx)
        node_to_edge_timestamps[destination].append(timestamp)
        node_to_edge_types[destination].append(relation_type)
    for i in range(max_node_idx + 1):
        node_to_neighbors[i] = np.array(node_to_neighbors[i], dtype=np.int32)
        node_to_edge_idxs[i] = np.array(node_to_edge_idxs[i], dtype=np.int32)
        node_to_edge_timestamps[i] = np.array(node_to_edge_timestamps[i], dtype=np.int32)
        node_to_edge_types[i] = np.array(node_to_edge_types[i], dtype=np.int32)

    res = NeighborFinder(node_to_neighbors, node_to_edge_idxs, node_to_edge_timestamps, node_to_edge_types,
                         uniform=uniform, seed=seed)
    # try to save
    res.save(neighborFinder_path)
    return res


class NeighborFinder:
    """
    node idx->neighbors idx list
    node idx->edge idx  list
    node idx->timestamp lsit  sort on timestamp
    node idx->edge types list
    """

    def __init__(self, node_to_neighbors, node_to_edge_idxs, node_to_edge_timestamps, node_to_edge_types, uniform=False,
                 seed=None):
        self.node_to_neighbors = node_to_neighbors
        self.node_to_edge_idxs = node_to_edge_idxs
        self.node_to_edge_timestamps = node_to_edge_timestamps
        self.node_to_edge_types = node_to_edge_types
        self.uniform = uniform
        self.seed = seed
        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def save(self, path):
        pickle.dump([
            self.node_to_neighbors, self.node_to_edge_idxs,
            self.node_to_edge_timestamps, self.node_to_edge_types,
            self.uniform, self.seed],
            open(path, 'wb'))

    def load(self, path):
        self.node_to_neighbors, self.node_to_edge_idxs, \
        self.node_to_edge_timestamps, self.node_to_edge_types, \
        self.uniform, self.seed = pickle.load(open(path, "rb"))
        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)
        return self

    def find_before(self, src_idx, cut_time):
        """
        Extracts all the interactions happening before cut_time for user src_idx in the overall interaction graph. The returned interactions are sorted by time.
        Returns 3 lists: neighbors, edge_idxs, timestamps,edge_types
        """
        i = np.searchsorted(self.node_to_edge_timestamps[src_idx],
                            cut_time)  # time[0~i-1]< cut_time Binary search is used

        return self.node_to_neighbors[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], \
               self.node_to_edge_timestamps[src_idx][:i], self.node_to_edge_types[src_idx][:i]

    def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
        """
        Given a list of users ids and relative cut times, extracts a sampled temporal neighborhood of each user in the list.
        Params
        ------
        src_idx_l: List[int]
        cut_time_l: List[float],
        num_neighbors: int
        :return
        neighbors:[len(source_nodes),max(1,n_neighbors)]
        edge_idxs:[len(source_nodes),max(1,n_neighbors)]
        edge_times:[len(source_nodes),max(1,n_neighbors)]
        edge_types:[len(source_nodes),max(1,n_neighbors)]
        sorted on time, 紧后，前置index为零表示邻居不存在，若为均匀采样，邻居可能被重复采样
        """
        assert (len(source_nodes) == len(timestamps))

        tmp_n_neighbors = n_neighbors if n_neighbors > 0 else 1
        # NB! All interactions described in these matrices are sorted in each row by time
        neighbors = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.int32)  # each entry in position (i,j) represent the id of the item targeted by user src_idx_l[i] with an interaction happening before cut_time_l[i]
        edge_times = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.int32)  # each entry in position (i,j) represent the timestamp of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
        edge_idxs = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.int32)  # each entry in position (i,j) represent the interaction index of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
        edge_types = np.zeros((len(source_nodes), tmp_n_neighbors)).astype(
            np.int32)  # each entry in position (i,j) represent the interaction type of an interaction between user src_idx_l[i] and item neighbors[i,j] happening before cut_time_l[i]
        for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
            source_neighbors, source_edge_idxs, source_edge_times, source_edge_types = self.find_before(source_node,
                                                                                                        timestamp)  # extracts all neighbors, interactions indexes ,timestamps,edge_types of all interactions of user source_node happening before cut_time
            # 目前看来极有可能，index 0：null,若邻居节点的index为零则意味着莫得邻居了，所以强制让邻居数量为1
            if len(source_neighbors) > 0 and n_neighbors > 0:
                if self.uniform:  # if we are applying uniform sampling, shuffles the data above before sampling
                    sampled_idx = np.random.randint(0, len(source_neighbors), n_neighbors)  # 可能造成重复采样，即一个连接被多次采样

                    neighbors[i, :] = source_neighbors[sampled_idx]
                    edge_times[i, :] = source_edge_times[sampled_idx]
                    edge_idxs[i, :] = source_edge_idxs[sampled_idx]
                    edge_types[i, :] = source_edge_types[sampled_idx]
                    # re-sort based on time
                    pos = edge_times[i, :].argsort()
                    neighbors[i, :] = neighbors[i, :][pos]
                    edge_times[i, :] = edge_times[i, :][pos]
                    edge_idxs[i, :] = edge_idxs[i, :][pos]
                    edge_types[i, :] = edge_types[i, :][pos]
                else:
                    # Take most recent interactions
                    source_edge_times = source_edge_times[-n_neighbors:]  # 邻居数量可能少于n_neighbors
                    source_neighbors = source_neighbors[-n_neighbors:]
                    source_edge_idxs = source_edge_idxs[-n_neighbors:]
                    source_edge_types = source_edge_types[-n_neighbors:]

                    assert (len(source_neighbors) <= n_neighbors)
                    assert (len(source_edge_times) <= n_neighbors)
                    assert (len(source_edge_idxs) <= n_neighbors)
                    assert (len(source_edge_types) <= n_neighbors)

                    neighbors[i, n_neighbors - len(source_neighbors):] = source_neighbors  # index置为0，表示邻居不存在
                    edge_times[i, n_neighbors - len(source_edge_times):] = source_edge_times
                    edge_idxs[i, n_neighbors - len(source_edge_idxs):] = source_edge_idxs
                    edge_types[i, n_neighbors - len(source_edge_types):] = source_edge_types
        return neighbors, edge_idxs, edge_times, edge_types
