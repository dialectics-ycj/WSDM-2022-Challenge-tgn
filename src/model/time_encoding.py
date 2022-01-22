import numpy as np
import torch

class TimeEncode(torch.nn.Module):
    def __init__(self, dimension, *, mean_time_diff, std_time_diff):
        super(TimeEncode, self).__init__()

        self.dimension = dimension
        self.mean_time_diff = mean_time_diff
        self.std_time_diff = std_time_diff
        self.mods = torch.nn.Parameter(
            torch.tensor([24, 24 * 7, 24 * 28, 24 * 30, 24 * 365], dtype=torch.int32), requires_grad=False)
        self.divs = torch.nn.Parameter(
            torch.tensor([1, 24, 24 * 7, 24, 24 * 30], dtype=torch.int32), requires_grad=False)
        self.means = torch.nn.Parameter(
            torch.tensor([np.mean(np.arange(0, 24)), np.mean(np.arange(0, 7)),
                          np.mean(np.arange(0, 4)), np.mean(np.arange(0, 30)), np.mean(np.arange(0, 12))],
                         dtype=torch.float), requires_grad=False)
        self.stds = torch.nn.Parameter(
            torch.tensor([np.std(np.arange(0, 24)), np.std(np.arange(0, 7)),
                          np.std(np.arange(0, 4)), np.std(np.arange(0, 30)), np.std(np.arange(0, 12))],
                         dtype=torch.float), requires_grad=False)
        self.w = torch.nn.Linear(self.mods.shape[0], dimension)
        self.w1 = torch.nn.Linear(dimension, dimension)
        torch.nn.init.kaiming_normal_(self.w.weight)
        torch.nn.init.kaiming_normal_(self.w1.weight)
        self.act = torch.nn.ReLU()

    def forward(self, t):
        # t has shape [batch_size, L]   all item: time
        t = t.int().unsqueeze(dim=2)
        t = t % self.mods
        t = torch.div(t, self.divs, rounding_mode='trunc')
        t = t - self.means
        t = t / self.stds
        # t has shape [batch_size, self.input]
        t = self.act(self.w(t))
        t = self.act(self.w1(t))
        return t


class TimeEncode_Abs(torch.nn.Module):
    def __init__(self, dimension):
        super(TimeEncode_Abs, self).__init__()

        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)

        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                           .float().reshape(dimension, -1))
        self.w.bias = torch.nn.Parameter(torch.rand(dimension))

    def forward(self, t):
        # t has shape [batch_size, seq_len]   all item: time
        # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
        t = t.unsqueeze(dim=2)

        # output has shape [batch_size, seq_len, dimension]
        output = torch.cos(self.w(t))

        return output
