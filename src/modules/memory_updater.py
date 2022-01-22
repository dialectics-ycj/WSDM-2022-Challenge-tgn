import torch
from torch import nn

from utils.utils import MergeLayer


class MemoryUpdater(nn.Module):
    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        pass


class SequenceMemoryUpdater(MemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(SequenceMemoryUpdater, self).__init__()
        self.memory = memory
        self.layer_norm = torch.nn.LayerNorm(memory_dimension)
        self.message_dimension = message_dimension
        self.device = device

    def update_memory(self, unique_node_ids, unique_messages, timestamps):
        if len(unique_node_ids) <= 0:
            return

        assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                          "update memory to time in the past"
        # update last_update with memory
        self.memory.set_last_update(unique_node_ids, timestamps)
        assert 1 == 0, "error set_last_update"
        # update memory
        memory = self.memory.get_memory(unique_node_ids)  # get old memory
        updated_memory = self.memory_updater(unique_messages, memory)  # get updated memory by GRU or RNN
        self.memory.set_memory(unique_node_ids, updated_memory)  # 持久化

    def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
        '''

        :param unique_node_ids: list[num_node_ids]
        :param unique_messages: tensor [num_node_ids,message_dim],requires_grad=False
        :param timestamps:
        :return:all memory updated by given messages
        updated_memory:tensor[n_nodes,memory_dim],requires_grad=True
        updated_last_update:tensor[n_nodes],requires_grad=False
        '''
        if len(unique_node_ids) <= 0:
            return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

        assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                          "update memory to time in the past"
        # get updated memory
        if self.memory.updated_last_update is None:
            updated_memory = self.memory.memory.data.clone()  # 获取全部memory,memory模块在初始化时初始化所有memory，last_update; now requires_grad=False
        updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[
            unique_node_ids])  # 同样只更新指定node的memory但是不进行存储  requires_grad=True ;tensor中requires_grad属性保持全局一致
        # get updated timestamp
        updated_last_update = self.memory.last_update.data.clone()  # 获取全部last_update
        updated_last_update[unique_node_ids] = timestamps  # requires_grad=False

        return updated_memory, updated_last_update


class GRUCellMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(GRUCellMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

        self.memory_updater = nn.GRUCell(input_size=message_dimension,
                                         hidden_size=memory_dimension)


class GRUMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(GRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

        self.memory_updater = nn.GRU(input_size=message_dimension,
                                     hidden_size=memory_dimension, num_layers=1)

    def get_updated_memory(self, unique_node_ids, unique_messages_tensor, lens):
        updated_memory = self.memory.memory[unique_node_ids]
        pack_messages = torch.nn.utils.rnn.pack_padded_sequence(unique_messages_tensor, lens, enforce_sorted=False)
        _, h_n = self.memory_updater(pack_messages, updated_memory.view(1, len(unique_node_ids),
                                                                        self.memory.memory_dimension))
        updated_memory = h_n.view(-1, self.memory.memory_dimension)
        return updated_memory


class AttentionMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(AttentionMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)
        self.device = device
        self.multi_head_target = nn.MultiheadAttention(embed_dim=memory_dimension,
                                                       kdim=message_dimension,
                                                       vdim=message_dimension,
                                                       num_heads=4,
                                                       dropout=0.1)
        self.memory_updater = nn.GRUCell(input_size=memory_dimension, hidden_size=memory_dimension)
        self.attention_layer_norm = nn.LayerNorm(memory_dimension)
        self.merge_layer = MergeLayer(memory_dimension, memory_dimension, memory_dimension, memory_dimension)
        self.act = nn.ReLU()

    def get_updated_memory(self, unique_node_ids, unique_messages_tensor, lens):
        updated_memory = self.memory.memory[unique_node_ids]
        key = unique_messages_tensor  # S,N,E
        neighbors_padding_mask = torch.tensor(lens, device=self.device).unsqueeze(dim=1) <= \
                                 torch.arange(0, key.shape[0], device=self.device)  # N,1 ,L
        attn_output, _ = self.multi_head_target(query=updated_memory.unsqueeze(dim=0), key=key, value=key,
                                                key_padding_mask=neighbors_padding_mask)  # [n_neighbors, batch_size, num_of_features]
        attn_output = attn_output.squeeze(dim=0)
        # attn_output = self.layer_norm(attn_output)
        updated_memory = self.memory_updater(attn_output, updated_memory)
        return updated_memory


class AttentionMemoryUpdater_LSTM(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(AttentionMemoryUpdater_LSTM, self).__init__(memory, message_dimension, memory_dimension, device)
        self.device = device
        self.multi_head_target = nn.MultiheadAttention(embed_dim=memory_dimension,
                                                       kdim=message_dimension,
                                                       vdim=message_dimension,
                                                       num_heads=4,
                                                       dropout=0.1)
        self.memory_updater = nn.LSTMCell(input_size=memory_dimension, hidden_size=memory_dimension)
        self.attention_layer_norm = nn.LayerNorm(memory_dimension)
        self.merge_layer = MergeLayer(memory_dimension, memory_dimension, memory_dimension, memory_dimension)
        self.act = nn.ReLU()

    def get_updated_memory(self, unique_node_ids, unique_messages_tensor, lens):
        long_memory = self.memory.memory[unique_node_ids]
        short_memory = self.memory.short_memory[unique_node_ids]
        key = unique_messages_tensor  # S,N,E
        neighbors_padding_mask = torch.tensor(lens, device=self.device).unsqueeze(dim=1) <= \
                                 torch.arange(0, key.shape[0], device=self.device)  # N,1 ,L
        attn_output, _ = self.multi_head_target(query=long_memory.unsqueeze(dim=0), key=key, value=key,
                                                key_padding_mask=neighbors_padding_mask)  # [n_neighbors, batch_size, num_of_features]
        attn_output = self.attention_layer_norm(attn_output.squeeze(dim=0))
        long_memory = self.memory_updater(attn_output, short_memory, long_memory)
        return short_memory, long_memory


class RNNCellMemoryUpdater(SequenceMemoryUpdater):
    def __init__(self, memory, message_dimension, memory_dimension, device):
        super(RNNCellMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

        self.memory_updater = nn.RNNCell(input_size=message_dimension,
                                         hidden_size=memory_dimension)


def get_memory_updater(module_type, memory, message_dimension, memory_dimension, device):
    if module_type == "gru":
        return GRUCellMemoryUpdater(memory, message_dimension, memory_dimension, device)
    elif module_type == "rnn":
        return RNNCellMemoryUpdater(memory, message_dimension, memory_dimension, device)
    elif module_type == "gru_v2":
        return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
    elif module_type == "attention":
        return AttentionMemoryUpdater(memory, message_dimension, memory_dimension, device)
