import math
import time

import numpy as np
import torch
from torch import nn

from model.temporal_attention import TemporalAttentionLayer, TemporalAttentionLayer_EX
from model.time_encoding import TimeEncode
from utils.utils import MergeLayer


class EmbeddingModule(nn.Module):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device, use_edge_type_feats,
                 dropout):
        super(EmbeddingModule, self).__init__()
        self.node_features = node_features  # save node data
        self.edge_features = edge_features  # save edge data
        # self.memory = memory
        self.neighbor_finder = neighbor_finder  # neighbor_finder is on cpu
        self.time_encoder = time_encoder  # TGAT time encoder
        self.n_layers = n_layers  # attention layers,n-hop aggregate
        self.n_node_features = n_node_features  # node_feature dim
        self.n_edge_features = n_edge_features  # edge_feature dim
        self.n_time_features = n_time_features  #
        self.embedding_dimension = embedding_dimension
        self.dropout = dropout
        self.device = device
        self.use_edge_type_feats = use_edge_type_feats

    def get_edge_feats(self, edge_idxs, edge_types):
        if self.use_edge_type_feats:
            if isinstance(self.edge_features, torch.nn.Embedding):
                return self.edge_features(edge_types)
            else:
                return self.edge_features[edge_types]
        else:
            if isinstance(self.edge_features, torch.nn.Embedding):
                return self.edge_features(edge_idxs)
            else:
                return self.edge_features[edge_idxs]

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, last_update=None):
        pass


class IdentityEmbedding(EmbeddingModule):
    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, last_update=None):
        return memory[source_nodes, :]


class TimeEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device, use_edge_type_feats,
                 n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1, *, mean_time_diff, std_time_diff):
        super(TimeEmbedding, self).__init__(node_features, edge_features, memory,
                                            neighbor_finder, time_encoder, n_layers,
                                            n_node_features, n_edge_features, n_time_features,
                                            embedding_dimension, device, use_edge_type_feats, dropout)
        self.mean_time_diff = mean_time_diff
        self.std_time_diff = std_time_diff

        class NormalLinear(nn.Linear):
            # From Jodie code
            def reset_parameters(self):
                stdv = 1. / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.embedding_layer = NormalLinear(1, self.embedding_dimension)

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, last_update=None):
        time_diffs = (torch.FloatTensor(timestamps - last_update[source_nodes]).to(
            self.device) - self.mean_time_diff) / self.std_time_diff
        source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))
        return source_embeddings


class TimeMergeEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device, use_edge_type_feats,
                 n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1, *, mean_time_diff, std_time_diff):
        super(TimeMergeEmbedding, self).__init__(node_features, edge_features, memory,
                                                 neighbor_finder, time_encoder, n_layers,
                                                 n_node_features, n_edge_features, n_time_features,
                                                 embedding_dimension, device, use_edge_type_feats, dropout)
        self.mean_time_diff = mean_time_diff
        self.std_time_diff = std_time_diff
        self.embedding_layer = MergeLayer(self.time_encoder.dimension, memory.memory_dimension, embedding_dimension,
                                          embedding_dimension)
        self.source_time_encoder = TimeEncode(n_time_features, mean_time_diff=None, std_time_diff=None)

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, last_update=None):
        time_diffs = torch.FloatTensor(timestamps - last_update[source_nodes]).to(self.device)
        time_diff_embedding = self.source_time_encoder(time_diffs.unsqueeze(1))
        return self.embedding_layer(time_diff_embedding.squeeze(), memory[source_nodes, :])


class GraphEmbedding(EmbeddingModule):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device, use_edge_type_feats,
                 n_heads=2, dropout=0.1, use_memory=True, init_embedding_method="merge"):
        super(GraphEmbedding, self).__init__(node_features, edge_features, memory,
                                             neighbor_finder, time_encoder, n_layers,
                                             n_node_features, n_edge_features, n_time_features,
                                             embedding_dimension, device, use_edge_type_feats, dropout)

        self.use_memory = use_memory
        self.device = device
        self.source_time_encoder = TimeEncode(n_time_features, mean_time_diff=None, std_time_diff=None)
        self.ngh_time_encoder = TimeEncode(n_time_features, mean_time_diff=None, std_time_diff=None)
        if init_embedding_method == "just_sum":
            assert self.n_node_features == memory.memory_dimension, \
                "init_embedding_method is just_num,but embedding_dimension!=memory.memory_dimension"
            self.get_init_feats = self.get_init_feats_just_sum
            self.node_hidden_emb_dim = self.n_node_features

        elif init_embedding_method == "just_cat":
            self.get_init_feats = self.get_init_feats_just_cat
            self.node_hidden_emb_dim = memory.memory_dimension + self.n_node_features

        elif init_embedding_method == "merge":
            self.init_merge = MergeLayer(self.n_node_features, memory.memory_dimension, self.embedding_dimension,
                                         self.embedding_dimension)
            self.get_init_feats = self.get_init_feats_merge
            self.node_hidden_emb_dim = self.embedding_dimension
        elif init_embedding_method == "just_mem":
            assert memory.memory_dimension == embedding_dimension
            self.get_init_feats = self.get_init_feats_just_memory
            self.node_hidden_emb_dim = memory.memory_dimension
        else:
            assert 1 == 0, "embedding method error!"

    def get_init_feats_just_sum(self, source_node_features, source_node_memory):
        return source_node_features + source_node_memory

    def get_init_feats_just_cat(self, source_node_features, source_node_memory):
        return torch.cat([source_node_features, source_node_memory], dim=1)

    def get_init_feats_just_memory(self, source_node_features, source_node_memory):
        return source_node_memory

    def get_init_feats_merge(self, source_node_features, source_node_memory):
        return self.init_merge(source_node_features, source_node_memory)

    def comput_embedding_Markov(self, q_nodes, q_mems, q_upts, timestamps, memory, last_update, n_neighbors, n_layers):
        # q_nodes:被询问点，复制为n
        # q_mems:被询问点对应的mem
        # q_upts:被询问点的mem对应的时间，复制为n
        # timestamp:embedding的时间点
        # memory:全局的memory,用作邻居节点的memory
        # last_update:全局的last_update,用作邻居节点的last_update
        # 可以优化
        # prepare self
        source_embeddings = q_mems
        source_nodes_time_diffs = torch.FloatTensor(timestamps - q_upts).view(-1, 1).to(
            self.device)
        source_nodes_time_embedding = self.source_time_encoder(source_nodes_time_diffs)
        # prepare neighborhood
        neighbors, edge_idxs, edge_times, edge_types = self.neighbor_finder.get_temporal_neighbor(
            q_nodes,
            last_update[q_nodes].cpu().numpy(),  # 现在是预测阶段，所有的邻居都可以访问，所以挑选一个最大的时间戳即可
            n_neighbors=n_neighbors)
        neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
        edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)
        edge_types = torch.from_numpy(edge_types).long().to(self.device)
        edge_deltas = timestamps[:, np.newaxis] - edge_times
        edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)
        edge_time_deltas_embeddings = self.ngh_time_encoder(edge_deltas_torch)
        neighbors = neighbors.flatten()
        neighbor_embeddings = memory[neighbors].view(len(q_mems), n_neighbors, -1)
        edge_features = self.get_edge_feats(edge_idxs, edge_types)
        mask = neighbors_torch == 0
        source_embeddings = self.aggregate(n_layers, source_embeddings,
                                           source_nodes_time_embedding,
                                           neighbor_embeddings,
                                           edge_time_deltas_embeddings,
                                           edge_features,
                                           mask)
        return source_embeddings

    def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, last_update=None):
        assert (n_layers >= 0)
        if n_layers == 0:
            if self.node_features is not None:
                source_embeddings = self.node_features[source_nodes, :]

                if self.use_memory:
                    source_embeddings = self.get_init_feats(source_embeddings, memory[source_nodes, :])
            else:
                source_embeddings = memory[source_nodes, :]
            return source_embeddings
        else:
            source_embeddings = self.compute_embedding(memory=memory,
                                                       last_update=last_update,
                                                       source_nodes=source_nodes,
                                                       timestamps=timestamps,
                                                       n_layers=n_layers - 1,
                                                       n_neighbors=n_neighbors)
            # query node always has the start time -> time span == 0
            source_nodes_time_diffs = torch.FloatTensor(timestamps - last_update[source_nodes]).view(-1, 1).to(
                self.device)
            source_nodes_time_embedding = self.source_time_encoder(source_nodes_time_diffs)
            neighbors, edge_idxs, edge_times, edge_types = self.neighbor_finder.get_temporal_neighbor(
                source_nodes,
                last_update[source_nodes],
                n_neighbors=n_neighbors)
            # to device
            neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)
            edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)
            edge_types = torch.from_numpy(edge_types).long().to(self.device)
            edge_deltas = timestamps[:, np.newaxis] - edge_times
            edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)
            edge_time_deltas_embeddings = self.ngh_time_encoder(edge_deltas_torch)
            # print("edge_time_deltas_embeddings sum {}".format(edge_time_deltas_embeddings.sum()))
            neighbors = neighbors.flatten()
            neighbor_embeddings = self.compute_embedding(memory=memory,
                                                         last_update=last_update,
                                                         source_nodes=neighbors,
                                                         timestamps=edge_times.flatten(),
                                                         n_layers=n_layers - 1,
                                                         n_neighbors=n_neighbors)

            effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
            neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)

            edge_features = self.get_edge_feats(edge_idxs, edge_types)

            mask = neighbors_torch == 0

            source_embeddings = self.aggregate(n_layers, source_embeddings,
                                               source_nodes_time_embedding,
                                               neighbor_embeddings,
                                               edge_time_deltas_embeddings,
                                               edge_features,
                                               mask)
            return source_embeddings

    def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings,
                  edge_time_embeddings, edge_features, mask):
        return None


class GraphSumEmbedding(GraphEmbedding):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device, use_edge_type_feats,
                 n_heads=2, dropout=0.1, use_memory=True, init_embedding_method="merge"):
        super(GraphSumEmbedding, self).__init__(node_features=node_features,
                                                edge_features=edge_features,
                                                memory=memory,
                                                neighbor_finder=neighbor_finder,
                                                time_encoder=time_encoder, n_layers=n_layers,
                                                n_node_features=n_node_features,
                                                n_edge_features=n_edge_features,
                                                n_time_features=n_time_features,
                                                embedding_dimension=embedding_dimension,
                                                device=device,
                                                use_edge_type_feats=use_edge_type_feats,
                                                n_heads=n_heads, dropout=dropout,
                                                use_memory=use_memory,
                                                init_embedding_method=init_embedding_method)

        self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_time_features +
                                                             n_edge_features, embedding_dimension)
                                             for _ in range(n_layers)])
        self.linear_2 = torch.nn.ModuleList(
            [torch.nn.Linear(embedding_dimension + n_node_features + n_time_features,
                             embedding_dimension) for _ in range(n_layers)])

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings,
                  edge_time_embeddings, edge_features, mask):
        neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features],
                                       dim=2)
        neighbor_embeddings = self.linear_1[n_layer - 1](neighbors_features)
        neighbors_sum = torch.nn.functional.relu(torch.sum(neighbor_embeddings, dim=1))

        source_features = torch.cat([source_node_features,
                                     source_nodes_time_embedding.squeeze()], dim=1)
        source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
        source_embedding = self.linear_2[n_layer - 1](source_embedding)

        return source_embedding


class GraphAttentionEmbedding(GraphEmbedding):
    def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
                 n_node_features, n_edge_features, n_time_features, embedding_dimension, device, use_edge_type_feats,
                 n_heads=2, dropout=0.1, use_memory=True, init_embedding_method="merge", attn_mode="n_loop"):
        super(GraphAttentionEmbedding, self).__init__(node_features, edge_features, memory,
                                                      neighbor_finder, time_encoder, n_layers,
                                                      n_node_features, n_edge_features,
                                                      n_time_features,
                                                      embedding_dimension,
                                                      device, use_edge_type_feats,
                                                      n_heads, dropout,
                                                      use_memory,
                                                      init_embedding_method=init_embedding_method)
        if attn_mode == "n_loop":
            self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
                node_hidden_emb_dim=self.node_hidden_emb_dim,
                n_edge_features=n_edge_features,
                time_dim=n_time_features,
                n_head=n_heads,
                dropout=dropout,
                output_dimension=self.embedding_dimension)
                for _ in range(n_layers)])
        elif attn_mode == "w_loop":
            self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer_EX(
                node_hidden_emb_dim=self.node_hidden_emb_dim,
                n_edge_features=n_edge_features,
                time_dim=n_time_features,
                n_head=n_heads,
                dropout=dropout,
                output_dimension=self.embedding_dimension)
                for _ in range(n_layers)])

    def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                  neighbor_embeddings,
                  edge_time_embeddings, edge_features, mask):
        attention_model = self.attention_models[n_layer - 1]

        source_embedding, _ = attention_model(source_node_features,
                                              source_nodes_time_embedding,
                                              neighbor_embeddings,
                                              edge_time_embeddings,
                                              edge_features,
                                              mask)

        return source_embedding


def get_embedding_module(module_type, node_features, relation_embedding, memory, neighbor_finder,
                         time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device, use_edge_type_feats,
                         n_heads=2, dropout=0.1, n_neighbors=None,
                         use_memory=True,
                         init_embedding_method="merge", *,
                         mean_time_diff=0,
                         std_time_diff=1, attn_mode="n_loop"):
    if module_type == "graph_attention":
        return GraphAttentionEmbedding(node_features=node_features,
                                       edge_features=relation_embedding,
                                       memory=memory,
                                       neighbor_finder=neighbor_finder,
                                       time_encoder=time_encoder,
                                       n_layers=n_layers,
                                       n_node_features=n_node_features,
                                       n_edge_features=n_edge_features,
                                       n_time_features=n_time_features,
                                       embedding_dimension=embedding_dimension,
                                       device=device,
                                       use_edge_type_feats=use_edge_type_feats,
                                       n_heads=n_heads, dropout=dropout, use_memory=use_memory,
                                       init_embedding_method=init_embedding_method,
                                       attn_mode=attn_mode)
    elif module_type == "graph_sum":
        return GraphSumEmbedding(node_features=node_features,
                                 edge_features=relation_embedding,
                                 memory=memory,
                                 neighbor_finder=neighbor_finder,
                                 time_encoder=time_encoder,
                                 n_layers=n_layers,
                                 n_node_features=n_node_features,
                                 n_edge_features=n_edge_features,
                                 n_time_features=n_time_features,
                                 embedding_dimension=embedding_dimension,
                                 device=device,
                                 use_edge_type_feats=use_edge_type_feats,
                                 n_heads=n_heads, dropout=dropout, use_memory=use_memory,
                                 init_embedding_method=init_embedding_method)
    # elif module_type == "identity":
    #     assert memory.memory_dimension == embedding_dimension, \
    #         "embedding module type is identity, but memory_dimension not euqal embedding_dimension"
    #     return IdentityEmbedding(node_features=node_features,
    #                              edge_features=edge_features,
    #                              memory=memory,
    #                              neighbor_finder=neighbor_finder,
    #                              time_encoder=time_encoder,
    #                              n_layers=n_layers,
    #                              n_node_features=n_node_features,
    #                              n_edge_features=n_edge_features,
    #                              n_time_features=n_time_features,
    #                              embedding_dimension=embedding_dimension,
    #                              device=device,
    #                              use_edge_type_feats=use_edge_type_feats,
    #                              dropout=dropout)
    elif module_type == "time":
        print("time embedding module is unsafe")
        return TimeEmbedding(node_features=node_features,
                             edge_features=relation_embedding,
                             memory=memory,
                             neighbor_finder=neighbor_finder,
                             time_encoder=time_encoder,
                             n_layers=n_layers,
                             n_node_features=n_node_features,
                             n_edge_features=n_edge_features,
                             n_time_features=n_time_features,
                             embedding_dimension=embedding_dimension,
                             device=device,
                             use_edge_type_feats=use_edge_type_feats,
                             dropout=dropout,
                             n_neighbors=n_neighbors,
                             mean_time_diff=mean_time_diff,
                             std_time_diff=std_time_diff)
    elif module_type == "time_merge":
        return TimeMergeEmbedding(node_features=node_features,
                                  edge_features=relation_embedding,
                                  memory=memory,
                                  neighbor_finder=neighbor_finder,
                                  time_encoder=time_encoder,
                                  n_layers=n_layers,
                                  n_node_features=n_node_features,
                                  n_edge_features=n_edge_features,
                                  n_time_features=n_time_features,
                                  embedding_dimension=embedding_dimension,
                                  device=device,
                                  use_edge_type_feats=use_edge_type_feats,
                                  dropout=dropout,
                                  n_neighbors=n_neighbors,
                                  mean_time_diff=mean_time_diff,
                                  std_time_diff=std_time_diff)
    else:
        raise ValueError("Embedding Module {} not supported".format(module_type))
