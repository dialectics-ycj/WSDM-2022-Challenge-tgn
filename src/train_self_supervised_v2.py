import logging
import math
import random
import time
import shutil
import torch
import pandas as pd

from model.tgn import TGN
from myDataLoader import MyIterableDataset
from path_manager.path_manager import OutPutPathManager
from predict import predict_base
from utils.data_processing import get_data
from utils.myParser import MyParser
from utils.recoder import *
from utils.utils import EarlyStopMonitor, get_neighbor_finder

plt.set_loglevel("info")
# it's use for quick find whether there is a bug
DEBUG_MODEL = True
if (DEBUG_MODEL):
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("QUICK DEBUG MODEL")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

DEBUG_N_RUN = 1
DEBUG_N_EPOCH = 2
DEBUG_N_BATCH = 10

### set args
myParser = MyParser()
args = myParser.init_args()

torch.cuda.set_device(args.gpu)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

model_idx = f"{args.prefix}-{args.data}-{time.time()}"  # model index, unique

# init path manager
output_path_manager = OutPutPathManager(model_idx)
shutil.copy2("remote_run_A.sh", os.path.join(output_path_manager.get_log_dir(), "remote_run_A.sh"))
shutil.copy2("remote_run_B.sh", os.path.join(output_path_manager.get_log_dir(), "remote_run_B.sh"))
myParser.save_args(output_path_manager.get_args_path())

## set logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(output_path_manager.get_log_path())
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(f"model idx:{model_idx}")
logger.info(f"pid: {os.getpid()}")
logger.info(args)

# Set device
device_string = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

### Extract data for training
start_get_data = time.time()
node_features, edge_features, full_data, train_data, val_data, test_data = get_data(args.data, data_type="train",
                                                                                    split_time=args.split_time)
train_data.set_seed(1)
logger.info("get_data time:{:.2f}s".format(time.time() - start_get_data))

# Initialize training neighbor finder to retrieve temporal graph
start_get_neighbor_finder_time = time.time()
# find neighborhood before memory, it has no relation about data type
ngh_finder = get_neighbor_finder(full_data, args.uniform, dataName=args.data, model="full")
logger.info("finder time:{:.2f}s".format(time.time() - start_get_neighbor_finder_time))

# Compute time statistics
start_time = time.time()
if args.normal_time_diff:
    logger.info("use normalization for time diffs")
    time_shift_dict = {"mean_src": train_data.mean_time_diff,
                       "std_src": train_data.std_time_diff,
                       "mean_dst": train_data.mean_time_diff,
                       "std_dst": train_data.std_time_diff}
else:
    assert 1 == 0, "must normal time diffs"
logger.info("shift time:{:.2f}s".format(time.time() - start_time))

### start run model
# 1 RUN = x epoch
# 1 epoch = all batch
# 1 batch = m instance
for i in range(args.n_runs) if not DEBUG_MODEL else range(DEBUG_N_RUN):
    ## prepare for epoch
    # Initialize Model
    tgn = TGN.load_tgn(args, ngh_finder, node_features, edge_features, time_shift_dict, None,
                       n_nodes=full_data.max_node_id + 1)
    criterion = torch.nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(tgn.parameters(), lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    num_instance = len(train_data.sources)
    num_batch = math.ceil(num_instance / args.bs)
    # log message about run
    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))

    # log recoder on epoch level for train set and validation set
    epoch_train_recoder = SetBatchesRecoder(0, True, logger, "epoch train", output_path_manager.get_fig_dir(),
                                            output_path_manager.get_fig_info_dir(),
                                            "train-epoch")
    epoch_test_from_train_recoder = {
        "sum": BatchRecoder(0.5, False, logger, "epoch test from train-sum", output_path_manager.get_fig_dir(),
                            output_path_manager.get_fig_info_dir(),
                            "test_from_train-sum"),
        "mean": BatchRecoder(0.5, False, logger, "epoch test from train-mean", output_path_manager.get_fig_dir(),
                             output_path_manager.get_fig_info_dir(),
                             "test_from_train-mean"),
        "max": BatchRecoder(0.5, False, logger, "epoch test from train-max", output_path_manager.get_fig_dir(),
                            output_path_manager.get_fig_info_dir(),
                            "test_from_train-max")}
    epoch_predict_from_train_recoder = PredictRecoder(args.data, model_idx, logger, "epoch predict from train",
                                                      figs_dir=output_path_manager.get_fig_dir(),
                                                      fig_infos_dir=output_path_manager.get_fig_info_dir(),
                                                      title="predict_from_train")
    # get monitor
    early_stopper = EarlyStopMonitor(max_round=args.patience)
    for epoch in range(args.n_epoch) if not DEBUG_MODEL else range(DEBUG_N_EPOCH):
        logger.info('epoch {} start'.format(epoch))
        logger.info("epoch {} lr: {}".format(epoch, optimizer.param_groups[0]['lr']))
        ### Training
        start_epoch_train_time = time.time()
        set_batch_size = 100
        # log recoder on batch and set batch level
        train_batch_recoder = BatchRecoder(0.5, True)
        set_batch_recoder = SetBatchesRecoder(set_batch_size, True, logger, "set batches",
                                              output_path_manager.get_fig_train_dir(),
                                              output_path_manager.get_fig_info_train_dir(), f"epoch_{epoch}")

        # Reinitialize memory of the model at the start of each epoch
        if args.use_memory:
            tgn.memory.__init_memory__()

        # init data loader
        data_loader = iter(
            torch.utils.data.DataLoader(MyIterableDataset(train_data, args), batch_size=None, collate_fn=lambda x: x,
                                        num_workers=0 if DEBUG_MODEL else 1))
        # train on every batches
        tgn.train()
        tgn.predict_model_off()
        predict_jump_num = 0
        predict_zoom_num = 0
        bp_cost = 0
        fw_cost = 0
        start_set_batch = time.time()
        for memory_batch_idx in range(0, num_batch) if not DEBUG_MODEL else range(DEBUG_N_BATCH):
            ## load memory update batch
            memory_update_batch, memory_update_label = next(data_loader, (None, None))
            sources_batch, destinations_batch, edge_idxs_batch, edge_types_batch, timestamps_batch, src_time_diffs, dst_time_diffs = memory_update_batch
            # reply update predict
            for predict_batch_idx in range(args.reply_times):
                # update memory
                tgn.update_memory_v2_without_store(sources_batch, destinations_batch, edge_types_batch,
                                                   timestamps_batch, src_time_diffs,
                                                   dst_time_diffs, edge_idxs_batch)
                # predict
                predict_batch, predict_label = next(data_loader, (None, None))
                assert predict_label == "predict_batch", "batch type match error"
                if predict_batch is None:
                    predict_jump_num += 1
                    continue
                start_fw_time = time.time()
                predict_source_nodes, predict_destination_nodes, predict_edge_types, predict_edge_times, labels_np, weights = predict_batch
                with torch.no_grad():
                    labels = torch.FloatTensor(labels_np).to(device)
                probs_torch = tgn.compute_edge_probabilities_v2(
                    source_nodes=predict_source_nodes,
                    destination_nodes=predict_destination_nodes,
                    edge_types=predict_edge_types,
                    edge_times=predict_edge_times)
                probs_torch_np = probs_torch.detach().sigmoid().cpu().numpy()
                fw_cost += time.time() - start_fw_time
                # BP
                start_bp_time = time.time()
                if args.bl_relation == "cs":
                    reduction = 'sum'
                elif args.bl_relation == "none":
                    reduction = "mean"
                loss = torch.nn.functional.binary_cross_entropy_with_logits(probs_torch, labels, reduction=reduction,
                                                                            pos_weight=torch.tensor(args.pos_weight))
                optimizer.zero_grad()
                loss.backward(retain_graph=False)
                optimizer.step()
                bp_cost += time.time() - start_bp_time
                # record info
                train_batch_recoder.record(labels_np, probs_torch_np, loss.item())
                loss = 0
                # active set batch recoder
                if train_batch_recoder.num_records % set_batch_size == 0:
                    set_batch_recoder.record(train_batch_recoder)
                    set_batch_recoder.log()
            # update memory with store
            tgn.update_memory_v2_with_store(sources_batch, destinations_batch, edge_types_batch,
                                            timestamps_batch, src_time_diffs, dst_time_diffs, edge_idxs_batch)
            if (memory_batch_idx + 1) % set_batch_size == 0:
                avg_batch_time = (time.time() - start_epoch_train_time) / (memory_batch_idx + 1)
                logger.info("set batch cost: {:.2f}s".format(time.time() - start_set_batch))
                logger.info("set batch bp cost: {:.2f}s".format(bp_cost))
                logger.info("set batch fw cost: {:.2f}s".format(fw_cost))
                start_set_batch = time.time()
                bp_cost = 0
                fw_cost = 0
                logger.info('avg batch time:{:.2f}s, remaining time:{:.2f}s'.format(avg_batch_time,
                                                                                    avg_batch_time * (
                                                                                            num_batch - memory_batch_idx - 1)))
        # record epoch train message
        epoch_train_time = time.time() - start_epoch_train_time
        epoch_train_recoder.record(train_batch_recoder)

        # log epoch train message
        logger.info("-----------------------------train message-------------------------------------")
        logger.info('epoch: {} train took {:.2f}s'.format(epoch, epoch_train_time))
        logger.info('epoch: {} jump batch num {}'.format(epoch, predict_jump_num))
        logger.info('epoch: {} zoom batch num {}'.format(epoch, predict_zoom_num))
        epoch_train_recoder.log()
        # Save model
        torch.save(tgn.state_dict(), output_path_manager.get_checkpoint_path(epoch, "train"))
        logger.info("epoch {} model save {}".format(epoch, output_path_manager.get_checkpoint_path(epoch, "train")))
        # test set
        test_labels, test_pros_dict = predict_base(args, tgn, logger, data_type="test")
        for name, probs in test_pros_dict.items():
            epoch_test_from_train_recoder[name].record(test_labels, probs)
            epoch_test_from_train_recoder[name].log()
        # predict set
        predict_pros_dict = predict_base(args, tgn, logger, data_type="predict")
        epoch_predict_from_train_recoder.record(predict_pros_dict, output_path_manager)

        # record total message
        epoch_total_time = time.time() - start_epoch_train_time
        # log total message
        logger.info("-----------------------------total message------------------------------------")
        logger.info('epoch: {} took {:.2f}s'.format(epoch, epoch_total_time))
    # get final predict ans
    best_model_epoch = epoch_predict_from_train_recoder.aucs_dict["max"].index(
        max(epoch_predict_from_train_recoder.aucs_dict["max"]))
    tgn.load_state_dict(torch.load(output_path_manager.get_checkpoint_path(best_model_epoch, "train")))
    pros_dict = predict_base(args, tgn, logger, "final_predict", "normal")
    for key, value in pros_dict.items():
        pd.Series(value).to_csv(output_path_manager.get_final_predict_csv_path(args.data, key, best_model_epoch),
                                header=None, index=None)
    logger.info(f"Over, you can find results in $PROJECT/output/logs/{model_idx}/predict-result/XXX.csv")
