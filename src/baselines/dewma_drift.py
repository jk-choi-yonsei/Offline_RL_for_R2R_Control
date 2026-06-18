"""
Canonical drift-tracking double-EWMA controllers (for revision baseline rigor).

Two controllers faithful to the semiconductor R2R literature:

  - DoubleEWMA  : predictor-corrector / double-EWMA of Butler & Stefani
    (1994) in the cleaner form of Chen & Guo (2001), Eq. (12)-(13). Two EWMAs
    track the OFFSET a_t and the DRIFT D_t; the process gain b is FIXED (estimated
    once from data). recipe: b . u = T - a_t - D_t.

  - AgeBasedDoubleEWMA  : Chen & Guo (2001), Eq. (18)-(20). The drift is indexed
    by the process AGE A_t (consumable usage), so wear-paced drift is projected
    by the age increment. recipe: b . u = T - a_t - (A_{t+1}-A_t) D_t.

Both map the scalar RR target to a multi-zone action via least-norm using the
fixed gain vector b (same projection used elsewhere, but with b FIXED).
"""
import numpy as np


def fit_gain(train_data, action_dim, rr_idx, rr_end):
    """Fixed process gain b: OLS regression of next-RR on action (mean over RR sites)."""
    from sklearn.linear_model import LinearRegression
    a = np.asarray(train_data["actions"], dtype=np.float64)
    nxt = np.asarray(train_data["next_observations"], dtype=np.float64)
    y = nxt[:, rr_idx] if rr_end is None else nxt[:, rr_idx:rr_end].mean(axis=1)
    b = LinearRegression().fit(a, y).coef_.astype(np.float64)
    # guard against a near-zero gain (degenerate least-norm)
    if np.dot(b, b) < 1e-8:
        b = np.ones(action_dim, dtype=np.float64) / action_dim
    return b


class DoubleEWMA:
    """Drift-tracking double-EWMA (Butler-Stefani 1994; Chen 2001 Eq. 12-13), fixed gain."""

    def __init__(self, target_rr, action_dim, gain, lambda_0=0.3, lambda_1=0.5,
                 action_bounds=None):
        self.target = float(target_rr)
        self.action_dim = action_dim
        self.b = np.asarray(gain, dtype=np.float64)
        self.l0, self.l1 = lambda_0, lambda_1
        lo, hi = action_bounds
        self.lo, self.hi = np.asarray(lo), np.asarray(hi)
        self.reset()

    def reset(self):
        self.a = 0.0   # offset estimate
        self.D = 0.0   # drift estimate (per run)

    def predict_action(self, state=None, drift=None):
        need = self.target - self.a - self.D          # project one run ahead
        u = need * self.b / (np.dot(self.b, self.b) + 1e-8)
        return np.clip(u, self.lo, self.hi)

    def update(self, rr, action):
        bu = float(np.dot(self.b, np.asarray(action, dtype=np.float64)))
        a_prev = self.a
        # Eq.(12): first EWMA carries the drift forward
        self.a = self.l0 * (rr - bu) + (1 - self.l0) * (a_prev + self.D)
        # Eq.(13): second EWMA tracks the drift (change of offset)
        self.D = self.l1 * (self.a - a_prev) + (1 - self.l1) * self.D


class AgeBasedDoubleEWMA:
    """Age-based double-EWMA (Chen 2001 Eq. 18-20). Drift indexed by consumable age."""

    def __init__(self, target_rr, action_dim, gain, wear_idx, lambda_0=0.3, lambda_1=0.5,
                 action_bounds=None):
        self.target = float(target_rr)
        self.action_dim = action_dim
        self.b = np.asarray(gain, dtype=np.float64)
        self.wear_idx = list(wear_idx)
        self.l0, self.l1 = lambda_0, lambda_1
        lo, hi = action_bounds
        self.lo, self.hi = np.asarray(lo), np.asarray(hi)
        self.reset()

    def reset(self):
        self.a = 0.0          # offset estimate
        self.D = 0.0          # drift size PER UNIT AGE
        self.A_prev = None    # age at previous run
        self.A_curr = None    # age at current run
        self.dA_hist = []     # observed age increments (for projecting next)

    def _age(self, drift):
        return float(np.sum(np.asarray(drift, dtype=np.float64)[self.wear_idx]))

    def predict_action(self, state=None, drift=None):
        self.A_curr = self._age(drift) if drift is not None else (self.A_prev or 0.0)
        dA = float(np.mean(self.dA_hist)) if self.dA_hist else 1.0   # projected next increment
        need = self.target - self.a - dA * self.D
        u = need * self.b / (np.dot(self.b, self.b) + 1e-8)
        return np.clip(u, self.lo, self.hi)

    def update(self, rr, action):
        bu = float(np.dot(self.b, np.asarray(action, dtype=np.float64)))
        a_prev = self.a
        if self.A_prev is None:
            dA = 1.0
        else:
            dA = max(self.A_curr - self.A_prev, 1e-6)
            self.dA_hist.append(dA)
        # Eq.(18): offset carries age-scaled drift forward
        self.a = self.l0 * (rr - bu) + (1 - self.l0) * (a_prev + dA * self.D)
        # Eq.(19): drift speed per unit age
        self.D = self.l1 * ((self.a - a_prev) / dA) + (1 - self.l1) * self.D
        self.A_prev = self.A_curr
