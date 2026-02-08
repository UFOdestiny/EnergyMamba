import os
import time
import numpy as np
import torch
from base.metrics import Metrics


class BaseEngine:
    def __init__(self, device, model, dataloader, scaler, A_hat, loss_fn, lrate,
                 optimizer, scheduler, clip_grad_value, max_epochs, patience,
                 log_dir, logger, seed, args, alpha=0.1, normalize=True,
                 metric_list=None):
        self._normalize = normalize
        self._device = device
        self._dataloader = dataloader
        self._scaler = scaler
        self._loss_fn = loss_fn
        self._lrate = lrate
        self._optimizer = optimizer
        self._lr_scheduler = scheduler
        self._clip_grad_value = clip_grad_value
        self._max_epochs = max_epochs
        self._patience = patience
        self._iter_cnt = 0
        self._save_path = log_dir
        self._logger = logger
        self._seed = seed
        self.args = args
        self._mask_value = torch.tensor(float("nan"))
        self.alpha = alpha

        self.model = model
        self.model.to(self._device)
        self.A_hat = A_hat.to(self._device) if A_hat is not None else None

        if metric_list is None:
            metric_list = ["Quantile", "MAE", "MAPE", "RMSE", "MPIW", "COV", "IS"]
        self.metric = Metrics(self._loss_fn, metric_list, self.model.horizon)

        self._logger.info(f"{'Loss Function':20s}: {self._loss_fn}")
        self._logger.info(f"{'Parameters':20s}: {self.model.param_num()}")

        self._time_model = "{}_{}.pt".format(
            self.args.model_name, time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        )
        self._logger.info(f"Model Path: {os.path.join(self._save_path, self._time_model)}")

    def _to_device(self, tensors):
        if isinstance(tensors, list):
            return [t.to(self._device) for t in tensors]
        return tensors.to(self._device)

    def _to_tensor(self, nparray):
        if isinstance(nparray, list):
            return [torch.tensor(arr, dtype=torch.float32) for arr in nparray]
        return torch.tensor(nparray, dtype=torch.float32)

    def _prepare_batch(self, batch):
        return self._to_device(self._to_tensor(batch))

    def _inverse_transform(self, tensors, device="cuda"):
        def inv(t):
            return self._scaler.inverse_transform(t, device=device)
        if isinstance(tensors, list):
            return [inv(t) for t in tensors]
        return inv(tensors)

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_path, self._time_model))

    def load_model(self, save_path):
        f = os.path.join(save_path, self._time_model)
        if not os.path.exists(f):
            models = [i for i in os.listdir(save_path) if i.endswith(".pt")]
            if not models:
                self._logger.info(f"No model found in {save_path}")
                return
            models.sort(key=lambda fn: os.path.getmtime(os.path.join(save_path, fn)))
            f = os.path.join(save_path, models[-1])
        self.model.load_state_dict(torch.load(f, weights_only=False))

    def load_exact_model(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=False))

    def train_batch(self):
        self.model.train()
        self._dataloader["train_loader"].shuffle()
        mask_value = self._mask_value.to(self._device)

        for X, label in self._dataloader["train_loader"].get_iterator():
            self._optimizer.zero_grad()
            X, label = self._prepare_batch([X, label])
            label = label[..., :1]
            lo, mid, up = self.model(X, self.A_hat)

            if self._normalize:
                lo, mid, up, label = self._inverse_transform(
                    [lo, mid, up, label], device=self._device.type
                )

            res = self.metric.compute_one_batch(
                mid, label, mask_value, "train", lower=lo, upper=up
            )
            res.backward()

            if self._clip_grad_value != 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self._clip_grad_value
                )
            self._optimizer.step()
            self._iter_cnt += 1

    def train(self):
        wait = 0
        min_loss_val = np.inf
        for epoch in range(self._max_epochs):
            t1 = time.time()
            self.train_batch()
            t2 = time.time()

            v1 = time.time()
            self.evaluate("val")
            v2 = time.time()

            te1 = time.time()
            self.evaluate("test", train_test=True)
            te2 = time.time()

            valid_loss = self.metric.get_valid_loss()
            cur_lr = self._lr_scheduler.get_last_lr()[0] if self._lr_scheduler else self._lrate
            if self._lr_scheduler:
                self._lr_scheduler.step()

            msg = self.metric.get_epoch_msg(epoch + 1, cur_lr, t2 - t1, v2 - v1, te2 - te1)
            self._logger.info(msg)

            if valid_loss < min_loss_val:
                if valid_loss == 0:
                    break
                self.save_model(self._save_path)
                self._logger.info(f"Val loss: {min_loss_val:.3f} -> {valid_loss:.3f}")
                min_loss_val = valid_loss
                wait = 0
            else:
                wait += 1
                if wait == self._patience:
                    self._logger.info(f"Early stop at epoch {epoch + 1}")
                    break

        self.evaluate("test")

    def evaluate(self, mode, model_path=None, train_test=False):
        if mode == "test" and not train_test:
            if model_path:
                self.load_exact_model(model_path)
            else:
                self.load_model(self._save_path)

        self.model.eval()
        mids, los, ups, labels = [], [], [], []

        with torch.no_grad():
            for X, label in self._dataloader[f"{mode}_loader"].get_iterator():
                X, label = self._prepare_batch([X, label])
                label = label[..., :1]
                lo, mid, up = self.model(X, self.A_hat)

                if self._normalize:
                    lo, mid, up, label = self._inverse_transform(
                        [lo, mid, up, label], device=self._device.type
                    )

                mids.append(mid.cpu())
                los.append(lo.cpu())
                ups.append(up.cpu())
                labels.append(label.cpu())

        mids = torch.cat(mids, dim=0)
        los = torch.cat(los, dim=0)
        ups = torch.cat(ups, dim=0)
        labels = torch.cat(labels, dim=0)
        mask_value = torch.tensor(float("nan"))

        def _compute(mode_name):
            for h in range(self.model.horizon):
                self.metric.compute_one_batch(
                    mids[:, h:h + 1], labels[:, h:h + 1], mask_value, mode_name,
                    lower=los[:, h:h + 1], upper=ups[:, h:h + 1],
                )

        if mode == "val":
            _compute("valid")
            return

        if mode == "test":
            _compute("test")
            if not train_test:
                with self._logger.no_time():
                    self._logger.info("\n" + "=" * 25 + "     Test     " + "=" * 25)
                for msg in self.metric.get_test_msg():
                    self._logger.info(msg)

    def get_predictions(self, mode):
        self.model.eval()
        mids, los, ups, labels = [], [], [], []
        with torch.no_grad():
            for X, label in self._dataloader[f"{mode}_loader"].get_iterator():
                X, label = self._prepare_batch([X, label])
                label = label[..., :1]
                lo, mid, up = self.model(X, self.A_hat)
                if self._normalize:
                    lo, mid, up, label = self._inverse_transform(
                        [lo, mid, up, label], device=self._device.type
                    )
                los.append(lo.cpu())
                mids.append(mid.cpu())
                ups.append(up.cpu())
                labels.append(label.cpu())
        return (
            torch.cat(los, dim=0),
            torch.cat(mids, dim=0),
            torch.cat(ups, dim=0),
            torch.cat(labels, dim=0),
        )
