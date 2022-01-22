import math
import time

import torch

from utils.recoder import *

torch.manual_seed(0)
np.random.seed(0)


# for test
def eval_edge_prediction(*, model, data, args, logger=None, ):
    batch_recoder = BatchRecoder(0.5, False)
    with torch.no_grad():
        model = model.eval()
        model.predict_model_on()
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / args.bs_predict)
        data_loader = data.get_data(args.bs_predict, begin_batch_id=0)
        if logger is not None:
            logger.info("test batch size: {}".format(args.bs_predict))
            logger.info("test batch num: {}".format(num_test_batch))
            logger.info("test instance num: {}".format(num_test_instance))

        for k in range(num_test_batch):
            # positive
            predict_sources_batch, predict_destinations_batch, _, predict_edge_types_batch, \
            predict_timestamps_batch, _, _ = next(data_loader)
            positive_size = len(predict_sources_batch)
            # negative samples
            # negative from random
            negative_data = data.get_negative_data(strategy="random", random_method=args.random_method_random,
                                                   timestamp=data.min_timestamp,
                                                   min_period=0, max_period=data.max_timestamp - data.min_timestamp,
                                                   batch_size=min(args.bs_neg_from_random, positive_size),
                                                   jump_method=args.jump_method)
            negative_sources, negative_destinations, negative_relation_types, negative_timestamps = negative_data
            # negative from positive
            if args.bs_neg_from_positive != 0:
                negative_data = data.get_negative_data(strategy="from_positive",
                                                       random_method=args.random_method_positive,
                                                       timestamp=data.min_timestamp,
                                                       min_period=0,
                                                       max_period=data.max_timestamp - data.min_timestamp,
                                                       positive_sources=predict_sources_batch,
                                                       positive_destinations=predict_destinations_batch,
                                                       positive_types=predict_edge_types_batch,
                                                       positive_timestamps=predict_timestamps_batch,
                                                       batch_size=min(args.bs_neg_from_positive, positive_size),
                                                       jump_method=args.jump_method,
                                                       )
                negative_sources = np.concatenate([negative_sources, negative_data[0]])
                negative_destinations = np.concatenate([negative_destinations, negative_data[1]])
                negative_relation_types = np.concatenate([negative_relation_types, negative_data[2]])
                negative_timestamps = np.concatenate([negative_timestamps, negative_data[3]])

            negative_size = len(negative_sources)
            # predict
            probs_torch = model.compute_edge_probabilities_v2(
                source_nodes=np.concatenate([predict_sources_batch, negative_sources]),
                destination_nodes=np.concatenate([predict_destinations_batch, negative_destinations]),
                edge_times=np.concatenate([predict_timestamps_batch, negative_timestamps]),
                edge_types=np.concatenate([predict_edge_types_batch, negative_relation_types]),
            )

            # ONE
            pos_label_np = np.ones(positive_size)
            neg_label_np = np.zeros(negative_size)
            labels_np = np.concatenate([pos_label_np, neg_label_np])

            pred_score = probs_torch.cpu().numpy()
            batch_recoder.record(labels_np, pred_score)
    return batch_recoder


# for prediction
def predict_edge_perdiction(model, data, batch_size=200, logger=None):
    pos_probs = []
    with torch.no_grad():
        model = model.eval()
        model.predict_model_on()
        PREDICT_BATCH_SIZE = batch_size
        num_predict_instance = len(data.sources)
        num_predict_batch = math.ceil(num_predict_instance / PREDICT_BATCH_SIZE)
        if logger is not None:
            logger.info("predict batch size: {}".format(batch_size))
            logger.info("predict batch num: {}".format(num_predict_batch))
            logger.info("predict instance num: {}".format(num_predict_instance))
        predict_start_time = time.time()
        for k in range(num_predict_batch):
            s_idx = k * PREDICT_BATCH_SIZE
            e_idx = min(num_predict_instance, s_idx + PREDICT_BATCH_SIZE)

            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_types_batch = data.relation_types[s_idx: e_idx]

            # don't update memory
            probs_torch = model.compute_edge_probabilities_v2(
                source_nodes=sources_batch,
                destination_nodes=destinations_batch,
                edge_times=timestamps_batch,
                edge_types=edge_types_batch,
            )

            pos_prob = probs_torch.sigmoid().cpu().numpy().squeeze().tolist()
            pos_probs.extend(pos_prob)

        if logger is not None:
            logger.info("predict cost:{:.2f}s".format(time.time() - predict_start_time))
    return pos_probs



# for experiment
def exp_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200, update_memory_rate=0,
                        logger=None, *, n_relation):
    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    val_ap, val_auc = [], []
    res_record = [0 for _ in range(9)]
    with torch.no_grad():
        model = model.eval()
        # While usually the test batch size is as big as it fits in memory, here we keep it the same
        # size as the training batch size, since it allows the memory to be updated more frequently,
        # and later test batches to access information from interactions in previous test batches
        # through the memory
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = min(math.ceil(num_test_instance / TEST_BATCH_SIZE), 100)
        data_loader = data.get_data(TEST_BATCH_SIZE)
        if logger is not None:
            logger.info("test batch size: {}".format(batch_size))
            logger.info("test batch num: {}".format(2 * num_test_batch))
            logger.info("test instance num: {}".format(num_test_instance))

        for batch_idx in range(num_test_batch):
            sources_batch, destinations_batch, edge_idxs_batch, edge_types_batch, timestamps_batch = next(
                data_loader)
            size = len(sources_batch)
            negatives_src_batch, negatives_dst_batch = negative_edge_sampler.sample(size)

            print("timestamps_batch*=0")
            timestamps_batch *= 0

            # print("timestamps_batch+=100h")
            # timestamps_batch+=100

            # print("timestamps_batch-=100h")
            # timestamps_batch -= 100

            # print("random type")
            # edge_types_batch = np.random.randint(1, 248, len(edge_types_batch))

            # print("zero type")
            # edge_types_batch = np.ones_like(edge_types_batch)
            # don't update memory

            probs_torch = model.compute_edge_probabilities(
                source_nodes=np.concatenate(
                    [sources_batch, sources_batch, sources_batch, destinations_batch, destinations_batch,
                     destinations_batch, negatives_dst_batch, negatives_dst_batch, negatives_dst_batch]),
                destination_nodes=np.concatenate(
                    [sources_batch, destinations_batch, negatives_dst_batch, sources_batch, destinations_batch,
                     negatives_dst_batch, sources_batch, destinations_batch, negatives_dst_batch]),
                edge_times=np.concatenate(
                    [timestamps_batch, timestamps_batch, timestamps_batch, timestamps_batch, timestamps_batch,
                     timestamps_batch, timestamps_batch, timestamps_batch, timestamps_batch]),
                edge_types=np.concatenate(
                    [edge_types_batch, edge_types_batch, edge_types_batch, edge_types_batch, edge_types_batch,
                     edge_types_batch, edge_types_batch, edge_types_batch, edge_types_batch]),
                edge_idxs=np.concatenate(
                    [edge_idxs_batch, edge_idxs_batch, edge_idxs_batch, edge_idxs_batch, edge_idxs_batch,
                     edge_idxs_batch, edge_idxs_batch, edge_idxs_batch, edge_idxs_batch]),
                num_true_pair=0)
            probs_torch = probs_torch.cpu().numpy()
            for i in range(3):
                for j in range(3):
                    start_idx = (i * 3 + j)
                    res_record[start_idx] += sum(probs_torch[start_idx * size:start_idx * size + size])
            print("batch: {}".format(batch_idx))
        for i, name1 in zip(range(3), ["src", "dst", "random"]):
            for j, name2 in zip(range(3), ["src", "dst", "random"]):
                start_idx = (i * 3 + j)
                print("{} {}:".format(name1, name2), res_record[start_idx] / (num_test_batch * TEST_BATCH_SIZE))
    return np.mean(val_ap), np.mean(val_auc)  # mean of auc is meaningless
