from collections import defaultdict

import torch
from torch import nn


class Memory(nn.Module):
    def __init__(self, n_nodes, memory_dimension, message_dimension=None,
                 device="cpu", combination_method='sum', *, method_init_memory):
        super(Memory, self).__init__()
        self.n_nodes = n_nodes  # num of node
        self.memory_dimension = memory_dimension
        self.message_dimension = message_dimension
        self.device = device
        self.combination_method = combination_method
        self.method_init_memory = method_init_memory
        self.init_memory = None
        if self.init_memory is None:
            self.init_memory = torch.zeros(self.n_nodes, self.memory_dimension).detach()
            if self.method_init_memory == "random":
                nn.init.xavier_uniform_(self.init_memory)
        self.memory_embedding = torch.nn.Embedding(self.n_nodes, self.memory_dimension, device=self.device)
        self.__init_memory__()

    def __init_memory__(self):
        """
        Initializes the memory to all zeros. It should be called at the start of each epoch.
        """
        # Treat memory as parameter so that it is saved and loaded together with the model

        torch.nn.init.zeros_(self.memory_embedding.weight)
        self.memory = nn.Parameter(self.init_memory.clone().to(self.device), requires_grad=False)
        self.updated_memory = None
        self.last_update = nn.Parameter(torch.zeros(self.n_nodes, dtype=torch.int32).to(self.device),
                                        requires_grad=False)
        self.updated_last_update = None
        self.messages = defaultdict(list)  # key:node; value:list

    def store_raw_messages(self, nodes, node_id_to_messages):
        for node in nodes:
            self.messages[node].extend(node_id_to_messages[node])

    def get_memory(self, node_idxs):
        return self.memory[node_idxs, :]

    def set_memory(self, node_idxs, values):
        self.memory[node_idxs, :] = values.detach()

    def set_last_update(self, node_idxs, timestamps):
        self.last_update[node_idxs] = timestamps

    def get_last_update(self, node_idxs):
        return self.last_update[node_idxs]

    def backup_memory(self):
        messages_clone = {}
        for k, v in self.messages.items():
            messages_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]

        return self.memory.data.clone(), self.last_update.data.clone(), messages_clone

    def restore_memory(self, memory_backup):
        self.memory.data, self.last_update.data = memory_backup[0].clone(), memory_backup[1].clone()

        self.messages = defaultdict(list)
        for k, v in memory_backup[2].items():
            self.messages[k] = [(x[0].clone(), x[1].clone()) for x in v]

    def detach_memory(self):
        self.memory.detach_()
        # Detach all stored messages
        for k, v in self.messages.items():
            new_node_messages = []
            for message in v:
                new_node_messages.append((message[0].detach(), message[1]))

            self.messages[k] = new_node_messages

    def clear_messages(self, nodes):
        for node in nodes:
            self.messages[node] = []
