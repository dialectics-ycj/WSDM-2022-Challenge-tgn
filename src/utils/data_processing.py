import copy

import numpy as np

from path_manager.path_manager import DataPathManager


class Data:
    def __init__(self, sources, destinations, timestamps, relation, *, edge_idxs,
                 src_time_diffs=None, dst_time_diffs=None, bipartite=False, cover_type):
        self.sources = sources
        self.destinations = destinations
        self.timestamps = timestamps
        self.src_time_diffs = src_time_diffs
        self.dst_time_diffs = dst_time_diffs
        self.edge_idxs = edge_idxs
        self.relation_types = relation
        self.min_timestamp = min(timestamps)
        self.max_timestamp = max(timestamps)
        if self.src_time_diffs is not None:
            self.mean_time_diff = np.mean(np.concatenate([self.src_time_diffs, self.dst_time_diffs]))
            self.std_time_diff = np.std(np.concatenate([self.src_time_diffs, self.dst_time_diffs]))
        self.n_relations = self.relation_types.max() + 1  # max id +1 is num
        self.n_interactions = len(sources)
        if bipartite:
            self.min_sources_id = sources.min()
            self.max_sources_id = sources.max()
            self.min_destination_id = destinations.min()
            self.max_destination_id = destinations.max()
            assert self.max_sources_id < self.min_destination_id, "Illegal bipartite graph"
        else:
            self.min_sources_id = min(sources.min(), destinations.min())
            self.max_sources_id = max(sources.max(), destinations.max())
            self.min_destination_id = self.min_sources_id
            self.max_destination_id = self.max_sources_id
        self.max_node_id = max(self.max_sources_id, self.max_destination_id)

    def init_timestamp_to_index(self):
        self.timestamps_first = np.ones(self.max_timestamp + 1, dtype=np.int32) * -1
        self.timestamps_last = np.ones(self.max_timestamp + 1, dtype=np.int32) * -1
        for index, timestamp in enumerate(self.timestamps):
            if self.timestamps_first[timestamp] == -1:
                self.timestamps_first[timestamp] = index
            self.timestamps_last[timestamp] = index + 1
        for i in range(len(self.timestamps_first) - 1, -1, -1):
            if self.timestamps_first[i] == -1:
                self.timestamps_first[i] = self.timestamps_first[i + 1]
        if self.timestamps_last[0] == -1:
            self.timestamps_last[0] = 0
        for i in range(len(self.timestamps_last)):
            if self.timestamps_last[i] == -1:
                self.timestamps_last[i] = self.timestamps_last[i - 1]

    def set_seed(self, seed):
        self.seed = seed
        self.random_state = np.random.RandomState(self.seed)

    def reset_seed(self):
        self.random_state = np.random.RandomState(self.seed)

    def concatenated(self, other_data):
        assert isinstance(self.sources, type(np.array([1])))
        data = copy.deepcopy(self)
        data.sources = np.concatenate([data.sources, other_data.sources], dtype=np.int32)
        data.destinations = np.concatenate([data.destinations, other_data.destinations], dtype=np.int32)
        data.timestamps = np.concatenate([data.timestamps, other_data.timestamps], dtype=np.int32)
        data.edge_idxs = np.concatenate([data.edge_idxs, other_data.edge_idxs], dtype=np.int32)
        data.relation_types = np.concatenate([data.relation_types, other_data.relation_types], dtype=np.int32)
        data.n_interactions += other_data.n_interactions
        return data

    def get_data(self, batch_size, *, reverse=False, begin_batch_id=0):
        for start_idx in range(begin_batch_id * batch_size, self.n_interactions, batch_size):
            end_idx = min(start_idx + batch_size, self.n_interactions)
            if reverse:
                yield np.concatenate([self.sources[start_idx:end_idx], self.destinations[start_idx:end_idx]]), \
                      np.concatenate([self.destinations[start_idx:end_idx], self.sources[start_idx:end_idx]]), \
                      np.concatenate([self.edge_idxs[start_idx: end_idx], self.edge_idxs[start_idx: end_idx]]), \
                      np.concatenate(
                          [self.relation_types[start_idx:end_idx], self.relation_types[start_idx:end_idx]]), \
                      np.concatenate([self.timestamps[start_idx:end_idx], self.timestamps[start_idx:end_idx]])
            else:
                yield self.sources[start_idx:end_idx], \
                      self.destinations[start_idx:end_idx], \
                      self.edge_idxs[start_idx: end_idx], \
                      self.relation_types[start_idx:end_idx], \
                      self.timestamps[start_idx:end_idx], \
                      self.src_time_diffs[start_idx:end_idx], \
                      self.dst_time_diffs[start_idx:end_idx]

    def get_future_data(self, timestamp, min_period, max_period, batch_size, jump_method="jump"):
        """
        select samples from [timestamp+min_period,timestamp+max_period]
        left close right close
        """
        if jump_method == "jump" and timestamp + max_period > self.max_timestamp:
            return None
        elif jump_method == "zoom" and timestamp + max_period > self.max_timestamp:
            batch_size *= (self.max_timestamp - timestamp - min_period + 1) / (max_period - min_period + 1)
            batch_size = int(batch_size)
            if batch_size <= 0:
                return None
        min_timestamp = min(timestamp + min_period, self.max_timestamp)
        max_timestamp = min(timestamp + max_period, self.max_timestamp)
        if min_timestamp > max_timestamp:
            return None
        start_index = self.timestamps_first[min_timestamp]
        end_index = self.timestamps_last[max_timestamp]
        if start_index >= end_index:
            return None
        indexs = self.random_state.randint(start_index, end_index, batch_size)
        return self.sources[indexs], \
               self.destinations[indexs], \
               self.relation_types[indexs], \
               self.timestamps[indexs]

    def random_select(self, select_method, batch_size, *, min_timestamp, max_timestamp, use_random_relation=None):
        if select_method == "random":
            random_sources = self.random_state.randint(self.min_sources_id, self.max_sources_id + 1,
                                                       batch_size, dtype=np.int32)
            random_destinations = self.random_state.randint(self.min_destination_id, self.max_destination_id + 1,
                                                            batch_size, dtype=np.int32)
            random_relations = self.random_state.randint(1, self.n_relations, batch_size,
                                                         dtype=np.int32)
        elif select_method == "global":
            random_sources = self.random_state.randint(0, self.n_interactions, batch_size,
                                                       dtype=np.int32)
            random_sources = self.sources[random_sources]
            random_destinations = self.random_state.randint(0, self.n_interactions, batch_size,
                                                            dtype=np.int32)
            random_destinations = self.destinations[random_destinations]
            if use_random_relation:
                random_relations = self.random_state.randint(1, self.n_relations, batch_size,
                                                             dtype=np.int32)
            else:
                random_relations = self.random_state.randint(0, self.n_interactions, batch_size,
                                                             dtype=np.int32)
                random_relations = self.relation_types[random_relations]
        elif select_method == "local":
            start_index = self.timestamps_first[min_timestamp]
            end_index = self.timestamps_last[max_timestamp]
            if start_index >= end_index:
                return None
            random_sources = self.random_state.randint(start_index, end_index, batch_size,
                                                       dtype=np.int32)
            random_sources = self.sources[random_sources]
            random_destinations = self.random_state.randint(start_index, end_index, batch_size,
                                                            dtype=np.int32)
            random_destinations = self.destinations[random_destinations]
            if use_random_relation:
                random_relations = self.random_state.randint(1, self.n_relations, batch_size,
                                                             dtype=np.int32)
            else:
                random_relations = self.random_state.randint(start_index, end_index, batch_size,
                                                             dtype=np.int32)
                random_relations = self.relation_types[random_relations]
        random_timestamps = self.random_state.randint(min_timestamp, max_timestamp + 1, batch_size,
                                                      dtype=np.int32)
        return random_sources, random_destinations, random_relations, random_timestamps

    def get_negative_data(self, strategy="random", *, random_method=None, batch_size=None,
                          timestamp=None, min_period=None, max_period=None, jump_method=None,
                          positive_sources=None, positive_destinations=None,
                          positive_types=None, positive_timestamps=None, use_random_relation=False,
                          shift_base=None, shift_span=None):
        """
        random_method: ["global","local","random"]
        global: random select by global distribution
        local: random select by local distribution
        random: random select between min_value and max_value
        left close right close, [min timestamp, max timestamp]
        """
        if strategy == "random":
            if jump_method == "jump" and timestamp + max_period > self.max_timestamp:
                return None
            elif jump_method == "zoom" and timestamp + max_period > self.max_timestamp:
                batch_size *= (self.max_timestamp - timestamp - min_period + 1) / (
                        max_period - min_period + 1)
                batch_size = int(batch_size)
                if batch_size <= 0:
                    return None
            min_timestamp = min(timestamp + min_period, self.max_timestamp)
            max_timestamp = min(timestamp + max_period, self.max_timestamp)
            if min_timestamp > max_timestamp:
                return None
            return self.random_select(random_method, batch_size, min_timestamp=min_timestamp,
                                      max_timestamp=max_timestamp, use_random_relation=use_random_relation)
        elif strategy == "from_positive":
            index = self.random_state.randint(0, len(positive_sources), batch_size)
            positive_sources = positive_sources[index]
            positive_destinations = positive_destinations[index]
            positive_types = positive_types[index]
            positive_timestamps = positive_timestamps[index]
            min_timestamp = min(timestamp + min_period, self.max_timestamp)
            max_timestamp = min(timestamp + max_period, self.max_timestamp)
            if min_timestamp > max_timestamp:
                return None
            random_batch = self.random_select(random_method,
                                              batch_size,
                                              min_timestamp=min_timestamp,
                                              max_timestamp=max_timestamp,
                                              use_random_relation=use_random_relation)
            assert random_batch is not None, "some error"
            random_sources, random_destinations, random_relation_types, _ = random_batch
            negative_sources = np.concatenate([positive_sources, random_sources, random_sources])
            negative_destinations = np.concatenate(
                [random_destinations, positive_destinations, random_destinations])
            negative_relation_types = np.concatenate([positive_types, positive_types, random_relation_types])
            negative_timestamps = np.concatenate([positive_timestamps, positive_timestamps, positive_timestamps])
            return negative_sources, \
                   negative_destinations, \
                   negative_relation_types, \
                   negative_timestamps
        elif strategy == "from_mem":
            min_timestamp = min(timestamp + min_period, self.max_timestamp)
            max_timestamp = min(timestamp + max_period, self.max_timestamp)
            if min_timestamp > max_timestamp:
                return None
            nodes = np.unique(np.concatenate([positive_sources, positive_destinations]))
            random_batch = self.random_select("global",
                                              len(nodes),
                                              min_timestamp=min_timestamp,
                                              max_timestamp=max_timestamp,
                                              use_random_relation=True)
            assert random_batch is not None, "some error in from mem"
            random_sources, random_destinations, random_relation_types, random_timestamps = random_batch
            negative_sources = nodes
            negative_destinations = random_destinations
            negative_relation_types = random_relation_types
            negative_timestamps = random_timestamps
            return negative_sources, \
                   negative_destinations, \
                   negative_relation_types, \
                   negative_timestamps
        elif strategy == "time_sharp":
            # shift_range: [-shift_base-shift_span,-shift_base)+(shift_base,shift_base+shift_span]
            shift_range = np.concatenate(
                [np.arange(-shift_base - shift_span, -shift_base),
                 np.arange(shift_base + 1, shift_base + shift_span + 1)], dtype=np.int32)
            shift_times = np.random.randint(0, len(shift_range), len(positive_timestamps))
            shift_times = shift_range[shift_times]
            mask = np.logical_and(shift_times != 0, positive_timestamps + shift_times > timestamp)
            negative_sources = positive_sources[mask]
            negative_destinations = positive_destinations[mask]
            negative_relation_types = positive_types[mask]
            negative_timestamps = (positive_timestamps + shift_times)[mask]
            return negative_sources, \
                   negative_destinations, \
                   negative_relation_types, \
                   negative_timestamps

    def period_data(self):
        # used for markov 
        src = self.sources[0]
        dst = self.destinations[0]
        relation = self.relation_types[0]
        min_time = self.timestamps[0]
        max_time = None
        for i in range(1, self.n_interactions):
            if self.edge_idxs[i] != self.edge_idxs[i - 1]:
                yield src, dst, relation, min_time, self.timestamps[i - 1]
                src = self.sources[i]
                dst = self.destinations[i]
                relation = self.relation_types[i]
                min_time = self.timestamps[i]
        yield src, dst, relation, min_time, self.timestamps[-1]


def get_data(dataset_name, split_points=None, action_missing_feats=None, data_type="train", split_time=None):
    """
    split the dataset for train, val and test
    :param split_time: (max_time - split_time, max_time] is used for validation
    :param split_points:
    :param dataset_name: "A"  or "B"
    :param action_missing_feats: None;Zero;randomize
    :return:
    """
    data_path_manager = DataPathManager()
    if dataset_name == "B":
        bipartite = True
    else:
        bipartite = False

    if data_type == "train":
        # treat args
        if split_points is None:
            split_points = [0.70, 0.85]

        # load
        OUT_EDGES = data_path_manager.get_ml_edges_path(dataset_name, data_type)
        OUT_EDGE_FEAT = data_path_manager.get_ml_edge_feats_path(dataset_name)
        OUT_NODE_FEAT = data_path_manager.get_ml_node_feats_path(dataset_name)
        edges_np = np.load(OUT_EDGES)
        if dataset_name == "A":
            edges_feats_np = np.load(OUT_EDGE_FEAT)
            node_feats_np = np.load(OUT_NODE_FEAT)
            cover_type = "cover_range"
        else:
            edges_feats_np = None
            node_feats_np = None
            cover_type = "non_cover_range"

        # missing feats: randomize or zero
        if action_missing_feats is None:
            pass

        # get full data
        sources = edges_np[:, 0]
        destinations = edges_np[:, 1]
        relation_types = edges_np[:, 2]
        timestamps = edges_np[:, 3] // 3600
        timestamps -= min(timestamps)
        src_time_diffs = edges_np[:, 4] // 3600
        dst_time_diffs = edges_np[:, 5] // 3600
        if data_type == "train":
            edge_idxs = np.array(range(1, len(sources) + 1), dtype=np.int32)
        else:
            edge_idxs = edges_np[:, 4]

        # timestamp split point
        if split_time is not None:
            val_time = max(timestamps) - split_time
            test_time = max(timestamps) - split_time
        else:
            val_time, test_time = list(np.quantile(timestamps, split_points))
        # get full data
        full_data = Data(sources, destinations, timestamps, relation_types,
                         src_time_diffs=src_time_diffs, dst_time_diffs=dst_time_diffs, edge_idxs=edge_idxs,
                         bipartite=bipartite, cover_type=cover_type)

        # get train,val and test
        train_mask = timestamps <= val_time
        if split_time is not None:
            val_mask = timestamps > test_time
        else:
            val_mask = np.logical_and(timestamps >= val_time, timestamps < test_time)
        test_mask = timestamps > test_time

        train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                          relation_types[train_mask], edge_idxs=edge_idxs[train_mask],
                          src_time_diffs=src_time_diffs[train_mask], dst_time_diffs=dst_time_diffs[train_mask],
                          bipartite=bipartite, cover_type=cover_type)
        train_data.init_timestamp_to_index()
        val_data, test_data = None, None
        if split_time != 0:
            val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                            relation_types[val_mask], edge_idxs=edge_idxs[val_mask],
                            src_time_diffs=src_time_diffs[val_mask], dst_time_diffs=dst_time_diffs[val_mask],
                            bipartite=bipartite, cover_type=cover_type)
            val_data.init_timestamp_to_index()
            test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                             relation_types[test_mask], edge_idxs=edge_idxs[test_mask],
                             src_time_diffs=src_time_diffs[test_mask], dst_time_diffs=dst_time_diffs[test_mask],
                             bipartite=bipartite, cover_type=cover_type)

        return node_feats_np, edges_feats_np, full_data, train_data, val_data, test_data
    if data_type != "train":
        OUT_EDGES = data_path_manager.get_ml_edges_path(dataset_name, data_type)
        if dataset_name == "A":
            cover_type = "cover_range"
        else:
            cover_type = "non_cover_range"
        edges_np = np.load(OUT_EDGES)
        sources = edges_np[:, 0].astype(np.int32)
        destinations = edges_np[:, 1].astype(np.int32)
        relation_types = edges_np[:, 2].astype(np.int32)
        timestamps = edges_np[:, 3].astype(np.int32)
        TRAIN_DATASET = data_path_manager.get_ml_edges_path(dataset_name, "train")
        train_dataset = np.load(TRAIN_DATASET)
        timestamps -= min(train_dataset[:, 3])
        timestamps //= 3600
        edge_idxs = edges_np[:, 4].astype(np.int32)
        full_data = Data(sources, destinations, timestamps, relation_types, edge_idxs=edge_idxs, bipartite=bipartite,
                         cover_type=cover_type)
        if data_type == "test":
            edge_labels = edges_np[:, 5].astype(np.int32)
            full_data.labels = edge_labels
        return full_data


def compute_time_statistics(sources, destinations, timestamps):
    last_timestamp_sources = dict()
    last_timestamp_dst = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    for k in range(len(sources)):
        source_id = sources[k]
        dest_id = destinations[k]
        c_timestamp = timestamps[k]
        if source_id not in last_timestamp_sources.keys():
            last_timestamp_sources[source_id] = 0
        if dest_id not in last_timestamp_dst.keys():
            last_timestamp_dst[dest_id] = 0
        all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
        all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
        last_timestamp_sources[source_id] = c_timestamp
        last_timestamp_dst[dest_id] = c_timestamp
    assert len(all_timediffs_src) == len(sources)
    assert len(all_timediffs_dst) == len(sources)

    mean_time_shift_src = np.mean(all_timediffs_src)
    std_time_shift_src = np.std(all_timediffs_src)
    mean_time_shift_dst = np.mean(all_timediffs_dst)
    std_time_shift_dst = np.std(all_timediffs_dst)
    time_shift = {"mean_src": mean_time_shift_src,
                  "std_src": std_time_shift_src,
                  "mean_dst": mean_time_shift_dst,
                  "std_dst": std_time_shift_dst}
    return time_shift


def get_labels(data):
    labels = []

    pre_id = data.edge_idxs[0]
    labels.append(data.labels[0])
    for id, label in zip(data.edge_idxs, data.labels):
        if id != pre_id:
            labels.append(label)
        pre_id = id
    return np.array(labels)


def get_ans(data, pos_probs):
    sum_probs = []
    max_probs = []
    mean_probs = []
    # stds = []
    # diffs = []
    # labels = []

    pre_id = data.edge_idxs[0]
    tmp_pros = []

    for id, prob in zip(data.edge_idxs, pos_probs):
        if id == pre_id:
            tmp_pros.append(prob)
        else:
            sum_probs.append(np.sum(tmp_pros))
            mean_probs.append(np.mean(tmp_pros))
            max_probs.append(np.max(tmp_pros))
            # stds.append(np.std(tmp_pros))
            # diffs.append(np.max(tmp_pros) - np.min(tmp_pros))
            tmp_pros.clear()
            tmp_pros.append(prob)
        pre_id = id
    sum_probs.append(np.sum(tmp_pros))
    mean_probs.append(np.mean(tmp_pros))
    max_probs.append(np.max(tmp_pros))
    # stds.append(np.std(tmp_pros))
    # diffs.append(np.max(tmp_pros) - np.min(tmp_pros))
    # print("sum score' mean:{}".format(np.mean(sum_probs)))
    # print("mean score' mean:{}".format(np.mean(mean_probs)))
    # print("max score' mean:{}".format(np.mean(max_probs)))
    # print("score std's mean:{}".format(np.mean(stds)))
    # print("diff's mean:{}".format(np.mean(diffs)))
    sum_probs = list(map(lambda x: min(1.0, x), sum_probs))
    return {"sum": np.array(sum_probs), "mean": np.array(mean_probs), "max": np.array(max_probs)}
