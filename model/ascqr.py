import torch
import numpy as np


class ASCQR:
    def __init__(self, alpha=0.1, gamma=0.005, window_size=100, delta=1e-6):
        self.alpha = alpha
        self.gamma = gamma
        self.window_size = window_size
        self.delta = delta

    def _nonconformity(self, lo, up, y):
        w = up - lo + self.delta
        return torch.maximum(lo - y, y - up) / w

    def run(self, cal_lo, cal_up, cal_y, test_lo, test_up, test_y):
        cal_scores = self._nonconformity(cal_lo, cal_up, cal_y)
        window = cal_scores.reshape(-1).tolist()[-self.window_size:]

        alpha_tilde = self.alpha
        result_lo = torch.zeros_like(test_lo)
        result_up = torch.zeros_like(test_up)

        for t in range(test_lo.shape[0]):
            w_t = test_up[t] - test_lo[t] + self.delta
            q_level = min(max(1 - alpha_tilde, 0.001), 0.999)
            Q_t = np.quantile(window[-self.window_size:], q_level)

            result_lo[t] = test_lo[t] - Q_t * w_t
            result_up[t] = test_up[t] + Q_t * w_t

            score_t = self._nonconformity(
                test_lo[t:t + 1], test_up[t:t + 1], test_y[t:t + 1]
            )
            window.extend(score_t.reshape(-1).tolist())

            covered = ((test_y[t] >= result_lo[t]) & (test_y[t] <= result_up[t])).float().mean()
            alpha_tilde += self.gamma * (self.alpha - (1.0 - covered.item()))
            alpha_tilde = max(0.001, min(alpha_tilde, 0.999))

        return result_lo, result_up
