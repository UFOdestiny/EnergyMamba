import torch
import numpy as np


class LogScaler:
    def __init__(self):
        self.base = 1

    def transform(self, data):
        return np.log(data + 1)

    def inverse_transform(self, data, device=None):
        if isinstance(data, np.ndarray):
            return np.exp(data) - 1
        return torch.exp(data) - 1


class LogMinMaxScaler:
    def __init__(self, data_min=0, data_max=0):
        self.data_min_ = torch.tensor(data_min)
        self.data_max_ = torch.tensor(data_max)

    def fit(self, data):
        log_data = np.log1p(data)
        self.data_min_ = log_data.min()
        self.data_max_ = log_data.max()
        return self

    def transform(self, data):
        log_data = np.log1p(data)
        return (log_data - self.data_min_) / (self.data_max_ - self.data_min_)

    def inverse_transform(self, data, device=None):
        span = self.data_max_ - self.data_min_
        if isinstance(data, torch.Tensor):
            log_data = data * span + self.data_min_
            return torch.expm1(log_data)
        log_data = data * span.cpu().numpy() + self.data_min_.cpu().numpy()
        return np.expm1(log_data)
