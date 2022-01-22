import logging
from collections import defaultdict

import numpy as np
import torch

from model.DistMult import DistMult
from model.time_encoding import TimeEncode
from modules.embedding_module import get_embedding_module
from modules.memory import Memory
from modules.memory_updater import get_memory_updater
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from utils.utils import MergeLayer, MLP, MLP_Classifier, SingleMLP, MultiMLP


class TGN(torch.nn.Module):
    def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2,
                 n_heads=2, dropout=0.1, use_memory=False,
                 memory_update_at_start=True,
                 message_dimension=100,
                 memory_dimension=500,
                 embedding_module_type="graph_attention",
                 message_function="mlp",
                 mean_time_shift_src=0, std_time_shift_src=1,
                 mean_time_shift_dst=0, std_time_shift_dst=1,
                 n_neighbors=None, aggregator_type="last",
                 memory_updater_type="gru",
                 use_destination_embedding_in_message=False,
                 use_source_embedding_in_message=False,
                 use_edge_type_feats=False,
                 dyrep=False,
                 embedding_dimension=100,
                 init_embedding_method="merge",
                 n_relation=None,
                 classifier_fun="merge",
                 *,
                 n_nodes,
                 time_diff_method="from_memory",
                 use_node_feat,
                 use_edge_feat,
                 use_edge_feat_as_emb,
                 relation_dim,
                 time_dim,
                 share_embedding,
                 method_init_memory,
                 attn_mode,
                 mem_weight,
                 memEmb_weight, ):
        super(TGN, self).__init__()
        self.PREDICT_MODEL = False
        self.n_layers = n_layers
        self.neighbor_finder = neighbor_finder
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.classifier_fun = classifier_fun
        self.time_dim = time_dim
        self.n_nodes = n_nodes
        self.n_neighbors = n_neighbors
        self.n_relation = n_relation
        self.mean_time_shift_src = mean_time_shift_src
        self.std_time_shift_src = std_time_shift_src
        self.mean_time_shift_dst = mean_time_shift_dst
        self.std_time_shift_dst = std_time_shift_dst
        self.relation_dim = relation_dim
        self.embedding_dimension = embedding_dimension
        self.embedding_module_type = embedding_module_type
        self.use_destination_embedding_in_message = use_destination_embedding_in_message
        self.use_source_embedding_in_message = use_source_embedding_in_message
        self.dyrep = dyrep
        self.use_edge_type_feats = use_edge_type_feats
        self.time_diff_method = time_diff_method
        self.use_memory = use_memory
        self.memory = None
        self.use_edge_feat_as_emb = use_edge_feat_as_emb
        self.method_init_memory = method_init_memory
        self.attn_mode = attn_mode
        if use_node_feat:
            self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
            self.n_node_features = self.node_raw_features.shape[1]
        else:
            self.node_raw_features = None
            self.n_node_features = None
        if use_edge_feat:
            self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)
            self.n_edge_features = edge_features.shape[1]
        else:
            self.edge_raw_features = None
            self.n_edge_features = None

        if use_edge_feat_as_emb:
            self.relation_embedding = torch.from_numpy(edge_features.astype(np.float32)).to(device)
            self.relation_dim = edge_features.shape[1]
        else:
            self.relation_embedding = torch.nn.Embedding(self.n_relation, self.relation_dim)
            torch.nn.init.xavier_uniform_(self.relation_embedding.weight.data)

        self.time_encoder = TimeEncode(dimension=time_dim, mean_time_diff=self.mean_time_shift_src,
                                       std_time_diff=self.std_time_shift_src)

        if self.use_memory:
            self.mem_weight = torch.nn.Parameter(torch.tensor(mem_weight))
            self.memEmb_weight = torch.nn.Parameter(torch.tensor(memEmb_weight))
            self.memory_dimension = memory_dimension
            self.memory_update_at_start = memory_update_at_start
            raw_message_dimension = 2 * self.embedding_dimension + self.relation_dim + self.time_encoder.dimension \
                if self.use_source_embedding_in_message else \
                2 * self.memory_dimension + self.relation_dim + self.time_encoder.dimension
            message_dimension = message_dimension if message_function != "identity" else raw_message_dimension

            self.memory = Memory(n_nodes=self.n_nodes,
                                 memory_dimension=self.memory_dimension,
                                 message_dimension=message_dimension,
                                 device=device, method_init_memory=self.method_init_memory)

            self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                             device=device)

            self.message_function = get_message_function(module_type=message_function,
                                                         raw_message_dimension=raw_message_dimension,
                                                         message_dimension=message_dimension)
            self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                                     memory=self.memory,
                                                     message_dimension=message_dimension,
                                                     memory_dimension=self.memory_dimension,
                                                     device=device)

        self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                     node_features=self.node_raw_features,
                                                     relation_embedding=self.relation_embedding,
                                                     memory=self.memory,
                                                     neighbor_finder=self.neighbor_finder,
                                                     time_encoder=self.time_encoder,
                                                     n_layers=self.n_layers,
                                                     n_node_features=self.n_node_features,
                                                     n_edge_features=self.relation_dim,
                                                     n_time_features=self.time_dim,
                                                     embedding_dimension=self.embedding_dimension,
                                                     device=self.device,
                                                     n_heads=n_heads, dropout=dropout,
                                                     use_memory=use_memory,
                                                     n_neighbors=self.n_neighbors,
                                                     use_edge_type_feats=use_edge_type_feats,
                                                     init_embedding_method=init_embedding_method,
                                                     mean_time_diff=self.mean_time_shift_dst,
                                                     std_time_diff=self.std_time_shift_dst,
                                                     attn_mode=self.attn_mode)

        # MLP to compute probability on an edge given two node embeddings
        if self.classifier_fun == "merge":
            self.affinity_score = MergeLayer(self.embedding_dimension, self.embedding_dimension,
                                             self.embedding_dimension,
                                             self.n_relation)
        elif self.classifier_fun == "DistMult":
            self.affinity_score = DistMult(rel_tot=self.n_relation, dim=self.embedding_dimension)
            if share_embedding:
                self.relation_embedding = self.affinity_score.rel_embeddings
        elif self.classifier_fun == "MLP":
            self.affinity_score = MLP(self.embedding_dimension + self.embedding_dimension, self.n_relation)
        elif self.classifier_fun == "myMLP":
            self.affinity_score = MLP_Classifier(self.embedding_dimension + self.embedding_dimension, self.n_relation)
        elif self.classifier_fun == "multi_mlp":
            self.affinity_score = torch.nn.ModuleList([MergeLayer(self.embedding_dimension, self.embedding_dimension,
                                                                  self.embedding_dimension, 1)
                                                       for _ in range(self.n_relation)])
        elif self.classifier_fun == "single_mlp":
            self.affinity_score = SingleMLP(
                self.embedding_dimension + self.embedding_dimension + self.relation_dim, 1, dropout)
        elif self.classifier_fun == "X_MMLP":
            self.affinity_score = MultiMLP(self.embedding_dimension + self.embedding_dimension, self.n_relation)

    def get_edge_feats(self, edge_idxs, edge_types):
        if not self.use_edge_type_feats:
            if self.use_edge_feat_as_emb:
                return self.relation_embedding[edge_idxs]
            else:
                return self.relation_embedding(torch.LongTensor(edge_idxs).to(self.device))

        else:
            if self.use_edge_feat_as_emb:
                return self.relation_embedding[edge_types]
            else:
                return self.relation_embedding(torch.LongTensor(edge_types).to(self.device))


    def set_neighbor_finder(self, neighbor_finder):
        self.neighbor_finder = neighbor_finder
        self.embedding_module.neighbor_finder = neighbor_finder

    def load_tgn(args, neighbor_finder, node_features, edge_features, time_shift: dict, state_dict=None, logger=None, *,
                 n_nodes):
        # three element, first: data, second: args, third: model weights
        device_string = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
        device = torch.device(device_string)
        tgn = TGN(neighbor_finder=neighbor_finder,
                  node_features=node_features,
                  edge_features=edge_features,
                  device=device,
                  n_layers=args.n_layer,  # num of graph attention layers
                  n_heads=args.n_head,
                  dropout=args.drop_out,
                  use_memory=args.use_memory,  #
                  message_dimension=args.message_dim,
                  memory_dimension=args.memory_dim,
                  memory_update_at_start=not args.memory_update_at_end,
                  embedding_module_type=args.embedding_module,
                  message_function=args.message_function,
                  aggregator_type=args.aggregator,
                  memory_updater_type=args.memory_updater,
                  n_neighbors=args.n_degree,
                  mean_time_shift_src=time_shift["mean_src"], std_time_shift_src=time_shift["std_src"],
                  mean_time_shift_dst=time_shift["mean_dst"], std_time_shift_dst=time_shift["std_dst"],
                  use_destination_embedding_in_message=args.use_destination_embedding_in_message,
                  use_source_embedding_in_message=args.use_source_embedding_in_message,
                  use_edge_type_feats=args.use_edge_type_feats,
                  dyrep=args.dyrep,
                  embedding_dimension=args.node_dim,  # assign node embedding
                  init_embedding_method=args.init_emb_method,
                  n_relation=args.n_relation,
                  classifier_fun=args.classifier_fun,
                  relation_dim=args.relation_dim,

                  n_nodes=n_nodes,
                  time_dim=args.time_dim,
                  use_node_feat=args.use_node_feat,
                  use_edge_feat=args.use_edge_feat,
                  use_edge_feat_as_emb=args.use_edge_feat_as_emb,
                  share_embedding=args.share_embedding,
                  method_init_memory=args.method_init_memory,
                  attn_mode=args.attn_mode,
                  mem_weight=args.mem_weight,
                  memEmb_weight=args.memEmb_weight)
        if state_dict is not None:
            tgn.load_state_dict(state_dict)
        tgn = tgn.to(device)
        return tgn

    def get_raw_message_v2(self, source_nodes, destination_nodes, edge_types, edge_times, src_time_diffs,
                           dst_time_diffs, edge_idxs):
        edge_features = self.get_edge_feats(edge_idxs, edge_types)
        source_memory = self.memory.get_memory(source_nodes)
        destination_memory = self.memory.get_memory(destination_nodes)
        assert destination_memory.requires_grad == False, "Error here"
        # get time delate encoding
        if self.time_diff_method == "from_memory":
            edge_times = torch.from_numpy(edge_times).to(self.device)
            source_time_delta = (edge_times - self.memory.get_last_update(source_nodes)).float()
            dst_time_delate = (edge_times - self.memory.get_last_update(destination_nodes)).float()
        elif self.time_diff_method == "from_interaction":
            source_time_delta = torch.from_numpy(src_time_diffs).to(self.device).float()
            dst_time_delate = torch.from_numpy(dst_time_diffs).to(self.device).float()
        src_time_delta_encoding = self.time_encoder(source_time_delta.view(-1, 1)).view(len(
            source_nodes), -1)
        dst_time_delta_encoding = self.time_encoder(dst_time_delate.view(-1, 1)).view(len(
            source_nodes), -1)
        source_messages = torch.cat([source_memory, destination_memory, edge_features, src_time_delta_encoding], dim=1)
        destination_messages = torch.cat([destination_memory, source_memory, edge_features, dst_time_delta_encoding],
                                         dim=1)
        return source_messages, destination_messages

    def merge_messages_to_matrix(self, source_nodes, destination_nodes, source_messages, destination_messages,
                                 edge_times):
        unique_nodes = np.unique(np.concatenate([source_nodes, destination_nodes]))
        node_to_index_dict = dict()

        for index, node in enumerate(unique_nodes):
            node_to_index_dict[node] = index
        message_list = [[] for _ in unique_nodes]
        last_update = [0 for _ in unique_nodes]
        for i in range(len(source_nodes)):
            message_list[node_to_index_dict[source_nodes[i]]].append(source_messages[i])
            message_list[node_to_index_dict[destination_nodes[i]]].append(destination_messages[i])
            last_update[node_to_index_dict[source_nodes[i]]] = edge_times[i]
            last_update[node_to_index_dict[destination_nodes[i]]] = edge_times[i]
        message_tensor_list = []
        lens = []

        for node_messages in message_list:
            message_tensor_list.append(torch.stack(node_messages))
            lens.append(len(node_messages))
        message_tensor = torch.nn.utils.rnn.pad_sequence(message_tensor_list)
        return unique_nodes, message_tensor, lens, last_update

    def update_memory_v2_with_store(self, source_nodes, destination_nodes, edge_types, edge_times, src_time_diffs,
                                    dst_time_diffs, edge_idxs):
        # if unique_nodes is given, directly store updated memory and updated last updated
        unique_nodes = self.update_memory_v2_without_store(source_nodes, destination_nodes,
                                                           edge_types,
                                                           edge_times,
                                                           src_time_diffs,
                                                           dst_time_diffs, edge_idxs)
        # store memory
        self.memory.set_memory(unique_nodes, self.memory.updated_memory[unique_nodes])
        # store last update
        self.memory.set_last_update(unique_nodes, self.memory.updated_last_update[unique_nodes])

    def update_memory_v2_without_store(self, source_nodes, destination_nodes, edge_types, edge_times, src_time_diffs,
                                       dst_time_diffs, edge_idxs):
        # get raw message
        src_raw_msgs, dst_raw_msgs = self.get_raw_message_v2(source_nodes, destination_nodes, edge_types, edge_times,
                                                             src_time_diffs, dst_time_diffs, edge_idxs)
        # get message
        src_msgs = self.message_function.compute_message(src_raw_msgs)
        dst_msgs = self.message_function.compute_message(dst_raw_msgs)
        # sort message by node idx and timestamp, then merge
        unique_nodes, message_tensor, lens, last_update = self.merge_messages_to_matrix(source_nodes, destination_nodes,
                                                                                        src_msgs,
                                                                                        dst_msgs, edge_times)
        # get updated memory
        updated_memory = self.memory_updater.get_updated_memory(unique_nodes, message_tensor, lens)
        # store updated memory

        if self.memory.updated_memory is None:
            self.memory.updated_memory = self.memory.memory.data.clone()
        else:
            self.memory.updated_memory.detach_()
        self.memory.updated_memory[unique_nodes] = updated_memory
        if self.memory.updated_last_update is None:
            self.memory.updated_last_update = self.memory.last_update.data.clone()
        self.memory.updated_last_update[unique_nodes] = torch.tensor(last_update).to(self.device)
        return unique_nodes

    def compute_temporal_embeddings_v2(self, *, source_nodes, destination_nodes, edge_times, edge_types):
        n_samples = len(source_nodes)
        nodes = np.concatenate([source_nodes, destination_nodes])
        timestamps = np.concatenate([edge_times, edge_times])
        if self.PREDICT_MODEL:
            # don't learn
            memory = self.memory.memory
            last_update = self.memory.last_update.cpu().numpy()
        else:
            memory = self.memory.updated_memory
            last_update = self.memory.updated_last_update.cpu().numpy()
        memory = memory * self.mem_weight + self.memory.memory_embedding.weight * self.memEmb_weight
        node_embedding = self.embedding_module.compute_embedding(memory=memory,
                                                                 last_update=last_update,
                                                                 source_nodes=nodes,
                                                                 timestamps=timestamps,
                                                                 n_layers=self.n_layers,
                                                                 n_neighbors=self.n_neighbors,
                                                                 )
        # print("embedding time:{}".format(time.time() - mini_time))
        source_node_embedding = node_embedding[:n_samples]
        destination_node_embedding = node_embedding[n_samples: 2 * n_samples]
        return source_node_embedding, destination_node_embedding

    def decoder(self, source_node_embedding, destination_node_embedding, edge_types):
        if self.classifier_fun == "myMLP":
            score = self.affinity_score(
                torch.cat([source_node_embedding, destination_node_embedding], dim=1)).squeeze(dim=0)
            return score
        elif self.classifier_fun == "DistMult":
            score = self.affinity_score(source_node_embedding, destination_node_embedding,
                                        torch.tensor(edge_types).long().to(self.device))
            return score
        elif self.classifier_fun == "multi_mlp":
            scores = torch.zeros(len(edge_types), device=self.device).to(self.device)
            for relation in range(self.n_relation):
                masks = edge_types == relation
                part_scores = self.affinity_score[relation](source_node_embedding[masks],
                                                            destination_node_embedding[masks]).squeeze(dim=1)
                part_scores = part_scores
                scores[masks] = part_scores
            return scores
        elif self.classifier_fun == "single_mlp":
            score = self.affinity_score(torch.cat(
                [source_node_embedding, destination_node_embedding, self.get_edge_feats(None, edge_types)],
                dim=1)).squeeze(dim=0)
            return score
        elif self.classifier_fun == "X_MMLP":
            score = self.affinity_score(torch.cat([source_node_embedding, destination_node_embedding], dim=1))
            # score.shape: (n_relation,b)
            score = score.T
            mask = torch.tensor(edge_types, device=self.device).unsqueeze(dim=1) == torch.arange(0, self.n_relation,
                                                                                                 device=self.device)
            score = score[mask]
            return score
        else:
            score = self.affinity_score(source_node_embedding, destination_node_embedding).squeeze(dim=0)
            score = score[range(len(edge_types)), edge_types]
            return score

    def compute_edge_probabilities_v2(self, *, source_nodes, destination_nodes, edge_types, edge_times):
        """
        Compute probabilities for edges of edge_type between sources and destination
        by first computing temporal embeddings using the TGN encoder and then feeding them
        into the MLP decoder.
        """
        n_samples = len(source_nodes)
        # get embedding
        source_node_embedding, destination_node_embedding = self.compute_temporal_embeddings_v2(
            source_nodes=source_nodes,
            destination_nodes=destination_nodes,
            edge_times=edge_times,
            edge_types=edge_types)
        return self.decoder(source_node_embedding, destination_node_embedding, edge_types)

    def predict_model_on(self):
        self.PREDICT_MODEL = True

    def predict_model_off(self):
        self.PREDICT_MODEL = False

    def get_raw_message_Markov(self, src_mem, dst_mem, src_upts, dst_upts, relations, timestamp):
        relation_embs = self.get_edge_feats(None, relations)
        # get time delate encoding
        source_time_delta = torch.from_numpy(timestamp - src_upts).float().to(self.device)
        dst_time_delate = torch.from_numpy(timestamp - dst_upts).float().to(self.device)
        src_time_delta_encoding = self.time_encoder(source_time_delta.view(-1, 1)).view(len(src_mem), -1)
        dst_time_delta_encoding = self.time_encoder(dst_time_delate.view(-1, 1)).view(len(src_mem), -1)
        source_messages = torch.cat([src_mem, dst_mem, relation_embs, src_time_delta_encoding], dim=1)
        destination_messages = torch.cat([dst_mem, src_mem, relation_embs, dst_time_delta_encoding], dim=1)
        return source_messages, destination_messages

    def compute_edge_probability_Markov(self, src_node, dst_node, relation, start_time, end_time, max_points,
                                        speed_mode):
        # [start_time,end_time]
        if speed_mode:
            period = int(np.ceil((end_time - start_time + 1) / max_points))
        else:
            period = max(1, int(np.floor((end_time - start_time + 1) / max_points)))
        time_points = list(range(start_time, end_time + 1, period))
        V_set = {"src_mems": self.memory.get_memory(src_node).view(1, -1),
                 "dst_mems": self.memory.get_memory(dst_node).view(1, -1),
                 "src_upts": self.memory.get_last_update(src_node).cpu().numpy().reshape(1),
                 "dst_upts": self.memory.get_last_update(dst_node).cpu().numpy().reshape(1),
                 "probabilities": torch.tensor([1.0], device=self.device), "len": 1}
        probability = 0
        assert self.n_layers == 1, "Markov only opt for one layer"
        for predict_time in time_points:
            ## predict
            # get embedding
            src_emd = self.embedding_module.comput_embedding_Markov(np.repeat(src_node, V_set["len"]),
                                                                    V_set["src_mems"],
                                                                    V_set["src_upts"],
                                                                    np.repeat(predict_time, V_set["len"]),
                                                                    self.memory.memory,
                                                                    self.memory.last_update,
                                                                    self.n_neighbors, self.n_layers)
            dst_emd = self.embedding_module.comput_embedding_Markov(np.repeat(dst_node, V_set["len"]),
                                                                    V_set["dst_mems"],
                                                                    V_set["dst_upts"],
                                                                    np.repeat(predict_time, V_set["len"]),
                                                                    self.memory.memory,
                                                                    self.memory.last_update,
                                                                    self.n_neighbors, self.n_layers)
            # decoder
            tmp_probs = self.decoder(src_emd, dst_emd, np.repeat(relation, V_set["len"]))
            ## update result
            probability += (tmp_probs * V_set["probabilities"]).sum()
            ## update V_set
            src_msgs, dst_msgs = self.get_raw_message_Markov(V_set["src_mems"], V_set["dst_mems"],
                                                             V_set["src_upts"], V_set["dst_upts"],
                                                             np.repeat(relation, V_set["len"]),
                                                             np.repeat(predict_time, V_set["len"]))
            src_msgs = self.message_function.compute_message(src_msgs)
            dst_msgs = self.message_function.compute_message(dst_msgs)
            _, updated_src_mems = self.memory_updater.memory_updater(
                src_msgs.view(1, V_set["len"], self.memory.message_dimension),
                V_set["src_mems"].view(1, V_set["len"], self.memory.memory_dimension))
            updated_src_mems = updated_src_mems.view(-1, self.memory.memory_dimension)
            _, updated_dst_mems = self.memory_updater.memory_updater(
                dst_msgs.view(1, V_set["len"], self.memory.message_dimension),
                V_set["dst_mems"].view(1, V_set["len"], self.memory.memory_dimension))
            updated_dst_mems = updated_dst_mems.view(-1, self.memory.memory_dimension)
            V_set["src_mems"] = torch.cat([V_set["src_mems"], updated_src_mems])
            V_set["dst_mems"] = torch.cat([V_set["dst_mems"], updated_dst_mems])
            V_set["src_upts"] = np.concatenate([V_set["src_upts"], np.repeat(predict_time, V_set["len"])])
            V_set["dst_upts"] = np.concatenate([V_set["dst_upts"], np.repeat(predict_time, V_set["len"])])
            V_set["probabilities"] = torch.cat(
                [V_set["probabilities"] * (1 - tmp_probs), V_set["probabilities"] * tmp_probs])
            V_set["len"] *= 2
        return probability

    def detach_tmp_memory(self, nodes):
        self.memory.updated_memory[nodes] = self.memory.updated_memory[nodes].detach()
