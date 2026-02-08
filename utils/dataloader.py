import math
import numpy as np
from pathlib import Path
from utils.args import get_data_path
from utils.generate import LogMinMaxScaler, LogScaler


class DataLoader:
    def __init__(self, data, idx, seq_len, horizon, bs, logger, name=None, droplast=False):
        self.data = np.asarray(data)
        self.idx = np.asarray(idx)
        self.size = len(self.idx)
        self.bs = bs
        self.droplast = droplast
        self.num_batch = self.size // bs if droplast else math.ceil(self.size / bs) if bs else 0
        self.current_ind = 0
        loader_name = name or "loader"
        logger.info(f"{loader_name:5s} num: {self.idx.shape[0]},\tBatch num: {self.num_batch}")
        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)

    def shuffle(self):
        perm = np.random.permutation(self.size)
        self.idx = self.idx[perm]

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start = self.bs * self.current_ind
                end = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = np.asarray(self.idx[start:end]).reshape(-1)
                if len(idx_ind) == 0:
                    break
                if self.droplast and len(idx_ind) < self.bs:
                    self.current_ind += 1
                    continue
                x = self.data[idx_ind[:, None] + self.x_offsets, ...].astype(np.float32, copy=False)
                y = self.data[idx_ind[:, None] + self.y_offsets, ...].astype(np.float32, copy=False)
                yield x, y
                self.current_ind += 1

        return _wrapper()


def load_dataset(data_path, args, logger):
    data_dir = Path(data_path) / args.years
    ptr = np.load(data_dir / "his.npz")
    logger.info(f"{'Data shape':20s}: {ptr['data'].shape}")
    X = ptr["data"]

    dataloader = {}
    for cat in ["train", "val", "test"]:
        idx = np.load(data_dir / f"idx_{cat}.npy")
        dataloader[f"{cat}_loader"] = DataLoader(
            X, idx, args.seq_len, args.horizon, args.bs, logger, cat
        )

    scaler = LogMinMaxScaler(ptr["min"], ptr["max"]) if "min" in ptr else LogScaler()
    return dataloader, scaler


def load_adj(adj_path):
    return np.load(adj_path)


def get_dataset_info(dataset):
    base_dir = get_data_path()
    d = {
        "NYISO": [base_dir + "", base_dir + "", 11],
        "CAISO": [base_dir + "", base_dir + "", 9],
        "panhandle": [base_dir + "", base_dir + "", 924],
    }
    return d[dataset]
