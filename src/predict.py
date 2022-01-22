import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluation.evaluation import predict_edge_perdiction, PredictRecoder
from model.tgn import TGN
from path_manager.path_manager import OutPutPathManager
from utils.data_processing import get_data, get_ans, get_labels
from utils.get_predict_ans import get_predict_auc
from utils.myParser import MyParser
from utils.recoder import BatchRecoder
from utils.utils import get_neighbor_finder


def get_args(model_idx, logger, ARGS_SAVE_PATH):
    args = MyParser().load_args(ARGS_SAVE_PATH)
    args.gpu = gpu
    logger.info(args)
    return args


def get_logger(model_idx, LOG_PATH):
    # get logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(LOG_PATH)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def load_models(model_idx, args, ngh_finder, node_features, edge_features, logger, model_type, min_epoch,
                max_epoch=None, n_nodes=None, time_shift=None, output_path_manager=None):
    MODEL_PATH_LST = []
    if model_type == "result":
        MODEL_SAVE_PATH = output_path_manager.get_model_path()
        MODEL_PATH_LST = [MODEL_SAVE_PATH]
    else:
        get_checkpoint_path = lambda epoch: output_path_manager.get_checkpoint_path(epoch, model_type)
        if max_epoch is None:
            model_epoch = min_epoch
            while os.path.exists(get_checkpoint_path(model_epoch)):
                MODEL_PATH_LST.append(get_checkpoint_path(model_epoch))
                model_epoch += 1
        else:
            MODEL_PATH_LST = [get_checkpoint_path(model_epoch) for model_epoch in
                              range(min_epoch, max_epoch)]

    for MODEL_PATH in MODEL_PATH_LST:
        logger.info(f'Loaded model from {MODEL_PATH} for predict')
        yield TGN.load_tgn(args, ngh_finder, node_features, edge_features, time_shift, torch.load(MODEL_PATH),
                           n_nodes=n_nodes)


def predict_base(args, tgn, logger, data_type, version="normal"):
    ## get predict_data
    predict_data = get_data(args.data, data_type=data_type)

    ### predict
    if version == "normal":
        pos_probs = predict_edge_perdiction(model=tgn,
                                            data=predict_data,
                                            batch_size=args.bs_predict,
                                            logger=logger)
        probs_dict = get_ans(predict_data, pos_probs)
    else:
        probs_dict = {"markov": []}
        with torch.no_grad():
            tgn = tgn.eval()
            tgn.predict_model_on()
            # test_datas=[]
            # for data in predict_data.period_data():
            #     test_datas.append(data)
            for src, dst, relation, min_time, max_time in predict_data.period_data():
                probs = tgn.compute_edge_probability_Markov(src, dst, relation, min_time, max_time, max_points=10,
                                                            speed_mode=True)
                probs_dict["markov"].append(min(probs.cpu().numpy(), 1.0))
            probs_dict["markov"] = np.array(probs_dict["markov"])
        tgn.predict_model_off()
    if data_type == "test":
        labels = get_labels(predict_data)
        return labels, probs_dict
    else:
        return probs_dict


def predict(args, tgn, logger, data_type):
    base_result = predict_base(args, tgn, logger, data_type)

    if data_type == "test":
        labels, probs_lst = base_result
        ap_dict = dict()
        auc_dict = dict()
        mean_score_dict = dict()
        for name, probs in probs_lst.items():
            probs = np.array(probs)
            ap = average_precision_score(labels, probs)
            auc = roc_auc_score(labels, probs)
            print(name, "predict ap:", ap, "predict auc", auc, "avg: ", np.mean(probs))
            ap_dict[name] = ap
            auc_dict[name] = auc
            mean_score_dict[name] = np.mean(probs)
        print("reverse ans")
        for name, probs in probs_lst.items():
            probs = np.array(probs)
            probs *= -1
            probs += 1
            ap = average_precision_score(labels, probs)
            auc = roc_auc_score(labels, probs)
            print(name, "predict ap:", ap, "predict auc", auc)
        return ap_dict, auc_dict, mean_score_dict
    else:
        return base_result


def get_time_shift(train_data):
    if args.normal_time_diff:
        time_shift = {"mean_src": train_data.mean_time_diff,
                      "std_src": train_data.std_time_diff,
                      "mean_dst": train_data.mean_time_diff,
                      "std_dst": train_data.std_time_diff}
    else:
        print("meaningless time_shift")
        time_shift = {"mean_src": 0.,
                      "std_src": 1.,
                      "mean_dst": 0.,
                      "std_dst": 1.}
    return time_shift


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    plt.set_loglevel("info")
    version = "normal"  # normal, Markov
    use_remote = False
    if use_remote:
        exp_name = None
        job_name = None
    else:
        exp_name = None
        job_name = None

    data_type = ["test", "predict"]  # test, predict, final_predict
    model_type_lst = ["train"]
    min_model_epoch = 4
    max_model_epoch = 5
    model_idx_list = [
        None,
    ]
    gpu = 0
    for model_idx in model_idx_list:
        output_path_manager = OutPutPathManager(model_idx, False, use_remote, exp_name=exp_name, job_name=job_name)
        logger = get_logger(model_idx, output_path_manager.get_predict_log_path())
        args = get_args(model_idx, logger, output_path_manager.get_args_path())
        # get base data
        start_get_data = time.time()
        node_features, edge_features, full_data, train_data, val_data, test_data = get_data(args.data,
                                                                                            data_type="train",
                                                                                            split_time=args.max_period)
        logger.info("get_data time:{:.2f}s".format(time.time() - start_get_data))
        ngh_finder = get_neighbor_finder(full_data, args.uniform, dataName=args.data, model="full")
        time_shift = get_time_shift(train_data)
        if version == "normal":
            epoch_test_from_train_recoder = {
                "sum": BatchRecoder(0.5, False, logger, "epoch test from train-sum",
                                    output_path_manager.get_predict_result_dir(),
                                    output_path_manager.get_predict_result_dir(),
                                    "test_from_train-sum"),
                "mean": BatchRecoder(0.5, False, logger, "epoch test from train-mean",
                                     output_path_manager.get_predict_result_dir(),
                                     output_path_manager.get_predict_result_dir(),
                                     "test_from_train-mean"),
                "max": BatchRecoder(0.5, False, logger, "epoch test from train-max",
                                    output_path_manager.get_predict_result_dir(),
                                    output_path_manager.get_predict_result_dir(),
                                    "test_from_train-max")}
            epoch_test_from_val_recoder = {
                "sum": BatchRecoder(0.5, False, logger, "epoch test from val-sum",
                                    output_path_manager.get_predict_result_dir(),
                                    output_path_manager.get_predict_result_dir(),
                                    "test_from_val-sum"),
                "mean": BatchRecoder(0.5, False, logger, "epoch test from val-mean",
                                     output_path_manager.get_predict_result_dir(),
                                     output_path_manager.get_predict_result_dir(),
                                     "test_from_val-mean"),
                "max": BatchRecoder(0.5, False, logger, "epoch test from val-max",
                                    output_path_manager.get_predict_result_dir(),
                                    output_path_manager.get_predict_result_dir(),
                                    "test_from_val-max")}
            epoch_predict_from_train_recoder = PredictRecoder(args.data, model_idx, logger, "epoch predict from train",
                                                              figs_dir=output_path_manager.get_predict_result_dir(),
                                                              fig_infos_dir=output_path_manager.get_predict_result_dir(),
                                                              title="predict_from_train")
            epoch_predict_from_val_recoder = PredictRecoder(args.data, model_idx, logger, "epoch predict from val",
                                                            figs_dir=output_path_manager.get_predict_result_dir(),
                                                            fig_infos_dir=output_path_manager.get_predict_result_dir(),
                                                            title="predict_from_val")
        else:
            epoch_test_from_train_recoder_Markov = BatchRecoder(0.5, False, logger, "epoch test from train-Markov",
                                                                output_path_manager.get_predict_result_dir(),
                                                                output_path_manager.get_predict_result_dir(),
                                                                "test_from_train-Markov")
            epoch_test_from_val_recoder_Markov = BatchRecoder(0.5, False, logger, "epoch test from val-Markov",
                                                              output_path_manager.get_predict_result_dir(),
                                                              output_path_manager.get_predict_result_dir(),
                                                              "test_from_val-Markov")
            epoch_predict_from_train_recoder_Markov = PredictRecoder(args.data, model_idx, logger,
                                                                     "epoch predict from train-Markov",
                                                                     figs_dir=output_path_manager.get_predict_result_dir(),
                                                                     fig_infos_dir=output_path_manager.get_predict_result_dir(),
                                                                     title="predict_from_train-Markov")
            epoch_predict_from_val_recoder_Markov = PredictRecoder(args.data, model_idx, logger,
                                                                   "epoch predict from val-Markov",
                                                                   figs_dir=output_path_manager.get_predict_result_dir(),
                                                                   fig_infos_dir=output_path_manager.get_predict_result_dir(),
                                                                   title="predict_from_val-Markov")
        for model_type in model_type_lst:
            epoch = min_model_epoch
            for tgn in load_models(model_idx, args, ngh_finder, node_features, edge_features, logger, model_type,
                                   min_model_epoch, max_model_epoch, n_nodes=full_data.max_node_id + 1,
                                   time_shift=time_shift, output_path_manager=output_path_manager):
                if "test" in data_type:
                    test_labels, pros_dict = predict_base(args, tgn, logger, data_type="test", version=version)
                    if version == "normal":
                        for name, probs in pros_dict.items():
                            if name == "special":
                                continue
                            if model_type == "train":
                                epoch_test_from_train_recoder[name].record(test_labels, probs)
                                epoch_test_from_train_recoder[name].log()
                            else:
                                epoch_test_from_val_recoder[name].record(test_labels, probs)
                                epoch_test_from_val_recoder[name].log()
                    else:
                        probs = pros_dict["markov"]
                        if model_type == "train":
                            epoch_test_from_train_recoder_Markov.record(test_labels, probs)
                            epoch_test_from_train_recoder_Markov.log()
                        else:
                            epoch_test_from_val_recoder_Markov.record(test_labels, probs)
                            epoch_test_from_val_recoder_Markov.log()
                    for key, value in pros_dict.items():
                        logger.info(
                            "epoch {} {} {} score' mean:{}, std:{}".format(epoch, "test", key, np.mean(value),
                                                                           np.std(value)))
                if "predict" in data_type:
                    # predict set
                    pros_dict = predict_base(args, tgn, logger, data_type="predict", version=version)
                    if version == "normal":
                        if model_type == "train":
                            epoch_predict_from_train_recoder.record(pros_dict, output_path_manager)
                        else:
                            epoch_predict_from_val_recoder.record(pros_dict, output_path_manager)
                    else:
                        if model_type == "train":
                            epoch_predict_from_train_recoder_Markov.record(pros_dict, output_path_manager)
                        else:
                            epoch_predict_from_val_recoder_Markov.record(pros_dict, output_path_manager)
                    for key, value in pros_dict.items():
                        logger.info(
                            "epoch {} {} {} score' mean:{}, std:{}".format(epoch, "predict", key, np.mean(value),
                                                                           np.std(value)))
                if "final_predict" in data_type:
                    pros_dict = predict_base(args, tgn, logger, data_type="final_predict", version=version)
                    for key, value in pros_dict.items():
                        pd.Series(value).to_csv(output_path_manager.get_final_predict_csv_path(args.data, key, epoch),
                                                header=None, index=None)
                    for key, value in pros_dict.items():
                        logger.info(
                            "epoch {} {} {} score' mean:{}, std:{}".format(epoch, "final_predict", key, np.mean(value),
                                                                           np.std(value)))
                epoch += 1
