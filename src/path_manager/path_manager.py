import os
from pathlib import Path


class OutPutPathManager:
    def __init__(self, model_idx, create=True, use_remote=False, *, exp_name=None, job_name=None):
        # mk dir in exps results for this exp's result
        # mk dir in exps logs for this exp's log
        # set results, logs dir
        if use_remote:
            None
        else:
            exp_results_dir = "../output/results"
            exp_logs_dir = '../output/logs'
        # judge
        if create and (os.path.exists(os.path.join(exp_results_dir, model_idx)) or os.path.exists(
                os.path.join(exp_logs_dir, model_idx))):
            assert 1 == 0, model_idx + " already exists"
        # mk dir for result
        Path(os.path.join(exp_results_dir, model_idx)).mkdir(parents=True, exist_ok=True)
        self.result_dir = os.path.join(exp_results_dir, model_idx)  # for this exp's result
        Path(os.path.join(self.result_dir, "saved_checkpoints")).mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = os.path.join(self.result_dir, "saved_checkpoints")  # for this exp's checkpoints
        Path(os.path.join(self.result_dir, "saved_models")).mkdir(parents=True, exist_ok=True)
        self.models_dir = os.path.join(self.result_dir, "saved_models")  # for this exp's best models
        # mk dir for log
        Path(os.path.join(exp_logs_dir, model_idx)).mkdir(parents=True, exist_ok=True)
        self.log_dir = os.path.join(exp_logs_dir, model_idx)  # for save log infos
        Path(os.path.join(self.log_dir, "board")).mkdir(parents=True, exist_ok=True)
        self.board_dir = os.path.join(self.log_dir, "board")  # for tensorboard
        Path(os.path.join(self.log_dir, "figs")).mkdir(parents=True, exist_ok=True)
        self.figs_dir = os.path.join(self.log_dir, "figs")  # for figs
        Path(os.path.join(self.figs_dir, "train")).mkdir(parents=True, exist_ok=True)
        self.figs_train_dir = os.path.join(self.figs_dir, "train")  # for figs of train
        Path(os.path.join(self.log_dir, "fig_infos")).mkdir(parents=True, exist_ok=True)
        self.fig_infos_dir = os.path.join(self.log_dir, "fig_infos")  # for fig infos will be deleted
        Path(os.path.join(self.fig_infos_dir, "train")).mkdir(parents=True, exist_ok=True)
        self.fig_infos_train_dir = os.path.join(self.fig_infos_dir, "train")  # for fig infos of tarin will be deleted

        # mk dir for predict
        Path(os.path.join(self.log_dir, "predict-result")).mkdir(parents=True, exist_ok=True)
        self.predict_result_dir = os.path.join(self.log_dir, "predict-result")  # for predict

    def get_log_dir(self):
        return self.log_dir

    def get_fig_dir(self):
        return self.figs_dir

    def get_fig_train_dir(self):
        return self.figs_train_dir

    def get_fig_info_dir(self):
        return self.fig_infos_dir

    def get_fig_info_train_dir(self):
        return self.fig_infos_train_dir

    def get_args_path(self):
        return os.path.join(self.log_dir, "args.pth")

    def get_log_path(self):
        return os.path.join(self.log_dir, "fileput.log")

    def get_model_path(self):
        return os.path.join(self.models_dir, "best_model.pth")

    def get_checkpoint_path(self, epoch, type):
        return os.path.join(self.checkpoints_dir, f"{epoch}-{type}.pth")

    def get_output_csv_path(self, data_name):
        return os.path.join(self.log_dir, f"output_{data_name}.csv")

    def get_final_predict_csv_path(self, data_name, predict_type, epoch):
        return os.path.join(self.predict_result_dir, f"output_{data_name}_{epoch}_{predict_type}.csv")

    def get_output_zip_path(self):
        return os.path.join(self.log_dir, f"output.zip")

    def get_predict_log_path(self):
        return os.path.join(self.log_dir, "predict.log")

    def get_predict_result_dir(self):
        return self.predict_result_dir


class DataPathManager:
    def __init__(self):
        self.data_dir = "../data"

    def get_ml_edges_path(self, data_name, data_type):
        # {ml/raw}_{data_name}_{data_type}_edges.npy/csv
        return os.path.join(self.data_dir, f"ml_{data_name}_{data_type}_edges.npy")

    def get_ml_node_feats_path(self, data_name):
        # {ml/raw}_{data_name}_node_feats.npy/csv
        return os.path.join(self.data_dir, f"ml_{data_name}_node_feats.npy")
        pass

    def get_ml_edge_feats_path(self, data_name):
        # {ml/raw}_{data_name}_edge_feats.npy/csv
        return os.path.join(self.data_dir, f"ml_{data_name}_edge_feats.npy")

    def get_raw_edges_path(self, data_name, data_type):
        # {ml/raw}_{data_name}_{data_type}_edges.npy/csv
        return os.path.join(self.data_dir, f"raw_{data_name}_{data_type}_edges.csv")

    def get_raw_node_feats_path(self, data_name):
        # {ml/raw}_{data_name}_node_feats.npy/csv
        return os.path.join(self.data_dir, f"raw_{data_name}_node_feats.csv")

    def get_raw_edge_feats_path(self, data_name):
        # {ml/raw}_{data_name}_edge_feats.npy
        return os.path.join(self.data_dir, f"raw_{data_name}_edge_feats.csv")

    def get_ngh_finder_path(self, data_name, model):
        return os.path.join(self.data_dir, "neighborFinder", f"neighborFinder_{data_name}_{model}.pth")

    def get_output_csv_path(self, data_name):
        return os.path.join(self.data_dir, f"output_{data_name}.csv")

    def get_output_zip_path(self):
        return os.path.join(self.data_dir, f"output.zip")
