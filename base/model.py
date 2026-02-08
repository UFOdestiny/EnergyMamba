import torch.nn as nn
from abc import abstractmethod


class BaseModel(nn.Module):
    def __init__(self, node_num, input_dim, output_dim, seq_len, horizon):
        super().__init__()
        self.node_num = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon

    @abstractmethod
    def forward(self, x, A_hat):
        raise NotImplementedError

    def param_num(self):
        return sum(p.nelement() for p in self.parameters())
