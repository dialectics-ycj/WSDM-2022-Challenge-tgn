import torch
import numpy as np


class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, data, args):
        super(MyIterableDataset).__init__()
        self.data = data
        self.args = args
        self.memory_update_data_loader = data.get_data(args.bs, begin_batch_id=0)
        # get relation distribution
        _, self.relation_counts = np.unique(np.concatenate([self.data.relation_types, np.zeros(1)]), return_counts=True)

    def get_predict_data(self, memory_update_data):
        # init variable
        sources_batch, destinations_batch, edge_idxs_batch, edge_types_batch, timestamps_batch, src_time_diffs, dst_time_diffs = memory_update_data
        train_data = self.data
        args = self.args
        # judge existence of predict data
        if timestamps_batch[-1] + args.min_period > train_data.max_timestamp:
            return None
        # get positive batch
        positive_data = train_data.get_future_data(timestamps_batch[-1], args.min_period, args.max_period,
                                                   args.bs_predict, jump_method=args.jump_method)
        if positive_data is None:
            return None
        positive_sources, positive_destinations, positive_relations, positive_timestamps = positive_data
        positive_size = len(positive_sources)
        # false positive sampling
        false_positive_sources = np.array([], dtype=np.int32)
        false_positive_destinations = np.array([], dtype=np.int32)
        false_positive_relations = np.array([], dtype=np.int32)
        false_positive_timestamps = np.array([], dtype=np.int32)
        for i in range(args.shift_times_p):
            false_positive_data = train_data.get_negative_data(strategy="time_sharp",
                                                               timestamp=timestamps_batch[-1],
                                                               positive_sources=positive_sources,
                                                               positive_destinations=positive_destinations,
                                                               positive_types=positive_relations,
                                                               positive_timestamps=positive_timestamps,
                                                               shift_base=args.shift_base_p,
                                                               shift_span=args.shift_span_p,
                                                               )
            false_positive_sources = np.concatenate([false_positive_sources, false_positive_data[0]])
            false_positive_destinations = np.concatenate([false_positive_destinations, false_positive_data[1]])
            false_positive_relations = np.concatenate([false_positive_relations, false_positive_data[2]])
            false_positive_timestamps = np.concatenate([false_positive_timestamps, false_positive_data[3]])
        false_positive_len = len(false_positive_sources)
        # negative sampling
        # get negative batch by random
        negative_data = train_data.get_negative_data(strategy="random", random_method=args.random_method_random,
                                                     timestamp=timestamps_batch[-1], min_period=args.min_period,
                                                     max_period=args.max_period,
                                                     batch_size=args.bs_neg_from_random,
                                                     jump_method=args.jump_method)
        if negative_data is None:
            return None
        negative_sources, negative_destinations, negative_relation_types, negative_timestamps = negative_data
        # get negative batch from positive
        if args.bs_neg_from_positive != 0:
            negative_data = train_data.get_negative_data(strategy="from_positive",
                                                         random_method=args.random_method_positive,
                                                         timestamp=timestamps_batch[-1], min_period=args.min_period,
                                                         max_period=args.max_period,
                                                         positive_sources=positive_sources,
                                                         positive_destinations=positive_destinations,
                                                         positive_types=positive_relations,
                                                         positive_timestamps=positive_timestamps,
                                                         batch_size=min(args.bs_neg_from_positive, positive_size),
                                                         jump_method=args.jump_method)
            negative_sources = np.concatenate([negative_sources, negative_data[0]])
            negative_destinations = np.concatenate([negative_destinations, negative_data[1]])
            negative_relation_types = np.concatenate([negative_relation_types, negative_data[2]])
            negative_timestamps = np.concatenate([negative_timestamps, negative_data[3]])
        # get negative batch from mem
        if args.mem_as_predict:
            # for cover updated mem
            negative_data = train_data.get_negative_data(strategy="from_mem",
                                                         timestamp=timestamps_batch[-1], min_period=args.min_period,
                                                         max_period=args.max_period,
                                                         positive_sources=sources_batch,
                                                         positive_destinations=destinations_batch,
                                                         positive_types=edge_types_batch,
                                                         positive_timestamps=timestamps_batch)
            negative_sources = np.concatenate([negative_sources, negative_data[0]])
            negative_destinations = np.concatenate([negative_destinations, negative_data[1]])
            negative_relation_types = np.concatenate([negative_relation_types, negative_data[2]])
            negative_timestamps = np.concatenate([negative_timestamps, negative_data[3]])
        # get negative batch by time sharp
        for i in range(args.shift_times_n):
            negative_data = train_data.get_negative_data(strategy="time_sharp",
                                                         timestamp=timestamps_batch[-1],
                                                         positive_sources=positive_sources,
                                                         positive_destinations=positive_destinations,
                                                         positive_types=positive_relations,
                                                         positive_timestamps=positive_timestamps,
                                                         shift_base=args.shift_base_n,
                                                         shift_span=args.shift_span_n,
                                                         )
            negative_sources = np.concatenate([negative_sources, negative_data[0]])
            negative_destinations = np.concatenate([negative_destinations, negative_data[1]])
            negative_relation_types = np.concatenate([negative_relation_types, negative_data[2]])
            negative_timestamps = np.concatenate([negative_timestamps, negative_data[3]])
        negative_size = len(negative_sources)
        assert min(positive_timestamps) > max(timestamps_batch)
        assert min(negative_timestamps) > max(timestamps_batch)

        # concatenate predict data
        predict_source_nodes = np.concatenate([positive_sources, false_positive_sources, negative_sources])
        predict_destination_nodes = np.concatenate(
            [positive_destinations, false_positive_destinations, negative_destinations])
        predict_edge_types = np.concatenate([positive_relations, false_positive_relations, negative_relation_types])
        predict_edge_times = np.concatenate([positive_timestamps, false_positive_timestamps, negative_timestamps])
        # concatenate predict labels
        pos_label_np = np.ones(positive_size + false_positive_len)
        neg_label_np = np.zeros(negative_size)
        labels_np = np.concatenate([pos_label_np, neg_label_np])

        # shuffle
        perm = np.arange(0, labels_np.shape[0])
        np.random.shuffle(perm)
        predict_source_nodes = predict_source_nodes[perm]
        predict_destination_nodes = predict_destination_nodes[perm]
        predict_edge_types = predict_edge_types[perm]
        predict_edge_times = predict_edge_times[perm]
        labels_np = labels_np[perm]
        # get loss weights
        if args.bl_relation == "none":
            weights = np.ones_like(labels_np)
        elif args.bl_relation == "cs":
            weights = self.relation_counts.min() / self.relation_counts[predict_edge_types]
        elif args.bl_relation == "focal":
            weights = np.ones_like(labels_np)
        elif args.bl_relation == "ghm":
            weights = np.ones_like(labels_np)
        # combination
        predict_data = predict_source_nodes, predict_destination_nodes, predict_edge_types, predict_edge_times, labels_np, weights
        return predict_data

    def get_data(self):
        for memory_update_data in self.memory_update_data_loader:
            yield memory_update_data, "memory_batch"
            for i in range(self.args.reply_times):
                yield self.get_predict_data(memory_update_data), "predict_batch"

    def __iter__(self):
        return self.get_data()


if __name__ == '__main__':
    # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
    ds = MyIterableDataset(bs=100)

    loader = torch.utils.data.DataLoader(ds, num_workers=1)
    next(loader)
    for data in loader:
        print(len(data))
