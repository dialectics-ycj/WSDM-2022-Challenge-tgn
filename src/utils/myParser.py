import argparse
import os
import pickle
import sys


class MyParser():

    def __init__(self):
        def str2bool(str):
            return True if str.lower() == 'true' else False

        self.parser = argparse.ArgumentParser('TGN self-supervised training')
        self.parser.add_argument('--use_node_feat', type=str2bool, default=False, help='use_node_feat')

        self.parser.add_argument('--share_embedding', type=str2bool, default=False,
                                 help='share embedding between DistMult and Embedding if use DistMult')
        self.parser.add_argument('--use_edge_feat', type=str2bool, default=False, help='use_edge_feat')
        self.parser.add_argument('--use_edge_feat_as_emb', type=str2bool, default=False, help='use_edge_feat_as_emb')
        self.parser.add_argument('--jump_method', type=str, default="zoom",
                                 choices=["jump", "zoom", "full"], help='Method to treat Not Full predict time window')
        self.parser.add_argument('--time_dim', type=int, default=8, help='Dimensions of the time embedding')
        self.parser.add_argument('--relation_dim', type=int, default=128, help='relation_dims')
        self.parser.add_argument('--normal_time_diff', type=str2bool, default=True, help='normalization time diff')
        self.parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. A or B)',
                                 default='A')
        self.parser.add_argument('--bs', type=int, default=1024, help='Batch_size')

        self.parser.add_argument('--bs_predict', type=int, default=1024, help='Predict_batch_size')
        self.parser.add_argument('--bs_neg_from_random', type=int, default=2048, help='Negative_batch_size from random')
        self.parser.add_argument('--bs_neg_from_positive', type=int, default=512,
                                 help='3*Negative_batch_size from positive, should < bs_predict')
        self.parser.add_argument('--min_period', type=int, default=1, help='min time period to predict')
        self.parser.add_argument('--max_period', type=int, default=1200, help='max time period to predict')
        self.parser.add_argument('--split_time', type=int, default=1200,
                                 help='time to split train set and validation set, zeros means:no validation')
        self.parser.add_argument('--prefix', type=str, default='A', help='Prefix to name the checkpoints')
        self.parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
        self.parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
        self.parser.add_argument('--n_epoch', type=int, default=10, help='Number of epochs')
        self.parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
        self.parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
        self.parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
        self.parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
        self.parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
        self.parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                               'backprop')
        self.parser.add_argument('--use_memory', type=str2bool, default=True,
                                 help='Whether to augment the model with a node memory')
        self.parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
            "graph_attention", "graph_sum", "identity", "time", "time_merge"], help='Type of embedding module')
        self.parser.add_argument('--message_function', type=str, default="mlp", choices=[
            "mlp", "identity"], help='Type of message function')
        self.parser.add_argument('--memory_updater', type=str, default="gru_v2", choices=[
            "gru", "rnn", "gru_v2","attention"], help='Type of memory updater; gru_v2: for version 2')
        self.parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                                'aggregator')
        self.parser.add_argument('--memory_update_at_end', type=str2bool, default=False,
                                 help='Whether to update memory at the end or at the start of the batch')
        self.parser.add_argument('--different_new_nodes', type=str2bool, default=False,
                                 help='Whether to use disjoint set of new nodes for train and val')
        self.parser.add_argument('--uniform', type=str2bool, default=False,
                                 help='take uniform sampling from temporal neighbors')
        self.parser.add_argument('--randomize_features', type=str2bool, default=False,
                                 help='Whether to randomize node features')
        self.parser.add_argument('--use_destination_embedding_in_message', type=str2bool, default=False,
                                 help='Whether to use the embedding of the destination node as part of the message')
        self.parser.add_argument('--use_source_embedding_in_message', type=str2bool, default=False,
                                 help='Whether to use the embedding of the source node as part of the message')
        self.parser.add_argument('--dyrep', type=str2bool, default=False, help='Whether to run the dyrep model')
        self.parser.add_argument('--use_edge_type_feats', type=str2bool, default=True, help='Whether to use edge type')
        self.parser.add_argument('--node_dim', type=int, default=128, help='Dimensions of the node embedding')
        self.parser.add_argument('--message_dim', type=int, default=128, help='Dimensions of the messages')
        self.parser.add_argument('--memory_dim', type=int, default=128, help='Dimensions of the memory for '
                                                                             'each user')
        self.parser.add_argument('--init_emb_method', type=str, default="merge", help='Method of init embedding')

        self.parser.add_argument('--update_memory_rate_in_val', type=int, default=0,
                                 help='whether update memory in val')
        self.parser.add_argument('--update_memory_rate_in_test', type=int, default=0,
                                 help='whether update memory in test')

        self.parser.add_argument('--ngh_finder_in_val', type=str, default="train",
                                 choices=["train", "val", "full"], help='Type of memory updater')
        self.parser.add_argument('--ngh_finder_in_test', type=str, default="train",
                                 choices=["train", "val", "full"], help='Type of memory updater')
        self.parser.add_argument('--n_relation', type=int, default=248, help='num of relation')
        self.parser.add_argument('--classifier_fun', type=str,
                                 choices=["merge", "myMerge", "MLP", "myMLP", "DistMult", "multi_mlp", "single_mlp",
                                          "X_MMLP"],
                                 default="merge", help='num of relation')
        self.parser.add_argument('--time_diff_method', type=str, choices=["from_memory", "from_interaction"],
                                 default="from_memory",
                                 help='method to cal time diff for update memory not for preprocess')
        self.parser.add_argument('--method_init_memory', type=str, choices=["zero", "random"],
                                 default="random",
                                 help='method to init and reset memory')
        self.parser.add_argument('--use_time_sharp_sample', type=str2bool, default=False,
                                 help='Whether to open time sharp sampler')
        self.parser.add_argument('--random_method_random', type=str, choices=["global", "local", "random"],
                                 default="global",
                                 help='method to random select negative samples from negative sample from random')
        self.parser.add_argument('--random_method_positive', type=str, choices=["global", "local", "random"],
                                 default="global",
                                 help='method to random select negative samples from negative sample from positive')
        self.parser.add_argument('--use_random_relation', type=str2bool, default=True,
                                 help='use random relations in predict')
        self.parser.add_argument('--mem_as_predict', type=str2bool, default=True,
                                 help='use nodes in mem to predict')
        self.parser.add_argument('--reply_times', type=int, default=2,
                                 help='one memory update batch with reply_times predict batches')
        self.parser.add_argument('--attn_mode', type=str, default="n_loop",
                                 help='one memory update batch with reply_times predict batches')
        self.parser.add_argument('--bl_relation', type=str,
                                 help='one memory update batch with reply_times predict batches')

        self.parser.add_argument('--shift_times_p', type=int,
                                 help='shift times for false positive')
        self.parser.add_argument('--shift_base_p', type=int,
                                 help='base shift time for false positive')
        self.parser.add_argument('--shift_span_p', type=int,
                                 help='span of shift time for false positive')

        self.parser.add_argument('--shift_times_n', type=int,
                                 help='shift times for negative')
        self.parser.add_argument('--shift_base_n', type=int,
                                 help='base shift time for negative')
        self.parser.add_argument('--shift_span_n', type=int,
                                 help='span of shift time for negative')
        self.parser.add_argument('--pos_weight', type=float, help='pos_weight')
        self.parser.add_argument('--mem_weight', type=float, help='mem_weight')
        self.parser.add_argument('--memEmb_weight', type=float, help='memEmb_weight')
    def init_args(self):
        try:
            self.args = self.parser.parse_args()
            return self.args
        except:
            self.parser.print_help()
            sys.exit(0)

    def save_args(self, path):
        pickle.dump(self.args, open(path, 'wb'))

    def load_args(self, path):
        if os.path.exists(path):
            self.args = pickle.load(open(path, 'rb'))
            return self.args
        else:
            print("can't load args")
            sys.exit(0)
