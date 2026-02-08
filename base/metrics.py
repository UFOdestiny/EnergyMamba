import numpy as np
import torch
import torch.nn.functional as F


class Metrics:
    def __init__(self, loss_func, metric_lst, horizon=1):
        self.dic = {
            "MAE": masked_mae,
            "MSE": masked_mse,
            "MAPE": masked_mape,
            "RMSE": masked_rmse,
            "MPIW": masked_mpiw,
            "COV": masked_coverage,
            "Quantile": masked_quantile,
            "IS": masked_IS,
        }
        self.horizon = horizon

        cleaned_metrics = []
        seen = set()
        for m in metric_lst:
            if m not in seen:
                cleaned_metrics.append(m)
                seen.add(m)
        if loss_func not in seen:
            cleaned_metrics.insert(0, loss_func)
            seen.add(loss_func)

        self.loss_name = loss_func
        self.metric_lst = cleaned_metrics
        self.metric_func = [self.dic[i] for i in self.metric_lst]
        self.early_stop_method_index = self.metric_lst.index(self.loss_name)

        self._basic_metrics = {"MAE", "MSE", "MAPE", "RMSE"}
        self._interval_metrics = {"MPIW"}
        self._interval_with_target = {"COV", "IS"}
        self._quantile_metric = {"Quantile"}

        self.N = len(self.metric_lst)
        self.train_res = [[] for _ in range(self.N)]
        self.valid_res = [[] for _ in range(self.N)]
        self.test_res = [[] for _ in range(self.N)]
        self.train_msg = None
        self.formatter()

    def formatter(self):
        def _section(prefix):
            return [f"{prefix} {m}: {{:.3f}}, " for m in self.metric_lst]
        msg_parts = ["Epoch: {:d}, "]
        msg_parts.extend(_section("Tr"))
        msg_parts.extend(_section("V"))
        msg_parts.extend(_section("Te"))
        msg_parts.append("LR: {:.4e}, Tr: {:.1f}s, V: {:.1f}s, Te: {:.1f}s")
        self.train_msg = "".join(msg_parts)

    def compute_one_batch(self, preds, labels, null_val, mode="train", **kwargs):
        grad_res = None
        storage = {"train": self.train_res, "valid": self.valid_res, "test": self.test_res}
        cur = storage.get(mode, self.test_res)
        null_tensor = self._align(null_val, preds)

        for i, fname in enumerate(self.metric_lst):
            func = self.metric_func[i]
            if fname in self._basic_metrics:
                res = func(preds, labels, null_tensor)
            elif fname in self._interval_metrics:
                res = func(kwargs["lower"], kwargs["upper"])
            elif fname in self._interval_with_target:
                res = func(kwargs["lower"], kwargs["upper"], labels, alpha=kwargs.get("alpha", 0.1))
            elif fname in self._quantile_metric:
                res = func(kwargs["lower"], preds, kwargs["upper"], labels)
            else:
                raise ValueError(f"Invalid metric: {fname}")
            if fname == self.loss_name and mode == "train":
                grad_res = res
            cur[i].append(self._scalar(res))
        return grad_res

    @staticmethod
    def _align(value, ref):
        if torch.is_tensor(value):
            return value.to(device=ref.device, dtype=ref.dtype)
        return torch.tensor(value, device=ref.device, dtype=ref.dtype)

    @staticmethod
    def _scalar(value):
        if torch.is_tensor(value):
            return value.detach().item()
        return float(value)

    def get_valid_loss(self):
        return np.mean(self.valid_res[self.early_stop_method_index])

    def get_test_loss(self):
        return np.mean(self.test_res[self.early_stop_method_index])

    def get_epoch_msg(self, epoch, lr, t_train, t_val, t_test):
        tr = [np.mean(i) for i in self.train_res]
        va = [np.mean(i) for i in self.valid_res]
        te = [np.mean(i) for i in self.test_res]
        msg = self.train_msg.format(epoch, *tr, *va, *te, lr, t_train, t_val, t_test)
        self._reset()
        return msg

    def _reset(self):
        self.train_res = [[] for _ in range(self.N)]
        self.valid_res = [[] for _ in range(self.N)]
        self.test_res = [[] for _ in range(self.N)]

    def get_test_msg(self):
        fmt = ", ".join([f"{m}: {{:.3f}}" for m in self.metric_lst])
        msgs = []
        for i in range(self.horizon):
            vals = [k[i] for k in self.test_res]
            msgs.append(f"Horizon {i + 1}: " + fmt.format(*vals))
        avg = [np.mean(i) for i in self.test_res]
        msgs.append("Average: " + fmt.format(*avg))
        self.test_res = [[] for _ in range(self.N)]
        return msgs


def get_mask(labels, null_val):
    if not torch.is_tensor(null_val):
        null_val = torch.tensor(null_val, device=labels.device, dtype=labels.dtype)
    else:
        null_val = null_val.to(device=labels.device, dtype=labels.dtype)
    if torch.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels != null_val
    return mask.float()


def _masked_mean(loss, labels, null_val):
    mask = get_mask(labels, null_val)
    n = torch.sum(mask)
    if n == 0:
        return torch.tensor(0.0, device=loss.device)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.sum(loss) / n


def masked_mae(preds, labels, null_val):
    return _masked_mean(torch.abs(preds - labels), labels, null_val)


def masked_mse(preds, labels, null_val):
    return _masked_mean((preds - labels) ** 2, labels, null_val)


def masked_rmse(preds, labels, null_val):
    return torch.sqrt(masked_mse(preds, labels, null_val))


def masked_mape(preds, labels, null_val):
    loss = torch.abs(preds - labels) / torch.abs(labels)
    return _masked_mean(loss, labels, labels.new_tensor(0.0)) * 100


def masked_mpiw(lower, upper):
    return torch.mean(upper - lower)


def masked_coverage(lower, upper, labels, alpha=None):
    in_range = torch.sum((labels >= lower) & (labels <= upper))
    return in_range / labels.numel() * 100


def masked_IS(lower, upper, labels, alpha=0.1):
    lo = lower.reshape(-1)
    up = upper.reshape(-1)
    y = labels.reshape(-1)
    width = up - lo
    below = (y < lo).float()
    above = (y > up).float()
    penalty = (lo - y) * below + (y - up) * above
    return (width + (2.0 / alpha) * penalty).mean()


def masked_quantile(y_lo, y_mid, y_up, y_true, q_lo=0.05, q_up=0.95, q_mid=0.5):
    mask = get_mask(y_true, torch.tensor(float("nan"), device=y_true.device, dtype=y_true.dtype))
    valid = mask.sum()
    if valid.item() == 0:
        return y_true.new_tensor(0.0)
    quantiles = y_true.new_tensor([q_lo, q_mid, q_up]).view(-1, *[1] * y_true.ndim)
    preds = torch.stack([y_lo, y_mid, y_up], dim=0)
    errors = y_true.unsqueeze(0) - preds
    pinball = torch.where(errors >= 0, quantiles * errors, (quantiles - 1) * errors)
    pinball = pinball * mask.unsqueeze(0)
    q_loss = pinball.sum() / valid * pinball.shape[0]
    mono = F.relu(y_lo - y_mid) + F.relu(y_mid - y_up)
    mono = (mono * mask).sum() / valid
    return q_loss + mono
