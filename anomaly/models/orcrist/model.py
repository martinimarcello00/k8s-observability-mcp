"""Orcrist: one conditional VAE for all the services of an app, the service being its condition.

Versions are named by two letters: the macro area first (A = metrics; B, C, ... will add traces and logs), then the
version inside it. A version never changes once its results are written: a new idea is a new version, and a new
component is a switch, off in the versions before it.

A sample is one service at one 30 s step: x, its B0 features from inputs.prepare (restarts stays a rule of the
protocol). Each feature is standardized over all services together, so the level of a service is carried by its
condition instead of being removed by the scaling. The decoder gives a mean and a deviation per feature, the deviation
at least SIGMA_MIN in log1p units: a cell almost flat in training (threads 33 vs 34 of an idle mongo) cannot make the
score explode. A cell flat or mostly missing in training is left out, as in the PCA.

Score of a feature: its squared residual in decoder deviations, ((x - mu) / sigma)^2, with z at the encoder's mean, so
the score is deterministic and is a sum over features like the PCA's SPE. spread() also gives the deviation (for the
normal range) and the KL of the step's posterior from the prior: the "state" signal, high when the service is where
training never saw it, even with features that agree with each other.
"""
import copy
import math
import time
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from anomaly import inputs
from anomaly.models.orcrist.versions import VERSIONS

TYPES = ("service", "mongodb", "memcached", "redis")  # the technology, read from the name; the rest are app services
D, H, K = 4, 32, 4  # embedding size, hidden units (two layers), latent size
SIGMA_MIN = 0.1  # floor of the decoder deviation, in log1p units (the z floor of the feature analysis)
INSTANCE_DROPOUT = 0.2
BATCH, LR, MAX_EPOCHS, PATIENCE, KL_WARMUP = 512, 1e-3, 300, 20, 20
MIN_FINITE, MIN_SD = 0.95, 1e-6  # a cell enters as in the PCA: finite in this share of training, not flat
SEED, THREADS = 0, 8


def service_type(name):
    return next((t for t in TYPES[1:] if t in name), TYPES[0])


def mlp(i, o):
    return nn.Sequential(nn.Linear(i, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, o))


class Net(nn.Module):
    def __init__(self, types, n_features, condition, prior, sigma_min):
        super().__init__()
        self.kind, self.prior_kind = condition, prior
        self.register_buffer("types", torch.as_tensor(types))
        self.register_buffer("sigma_min", torch.as_tensor(sigma_min, dtype=torch.float32))
        if condition == "onehot":
            self.register_buffer("onehot", torch.eye(len(types)))
            d = len(types)
        else:
            self.instance = nn.Embedding(len(types), D)
            if condition == "type+instance":
                self.tech = nn.Embedding(len(TYPES), D)
            d = D
        self.enc, self.dec = mlp(n_features + d, 2 * K), mlp(K + d, 2 * n_features)
        if prior == "service":
            self.prior = nn.Linear(d, 2 * K)

    def condition(self, s):
        if self.kind == "onehot":
            return self.onehot[s]
        e = self.instance(s)
        if self.kind == "type+instance":
            if self.training:
                e = e * (torch.rand(len(s), 1) >= INSTANCE_DROPOUT)
            e = e + self.tech(self.types[s])
        return e

    def forward(self, x, s):
        """(mean, deviation) of x, the KL per latent dimension and the posterior mean of z; z is sampled in training,
        the mean otherwise."""
        e = self.condition(s)
        mq, lq = self.enc(torch.cat([x, e], 1)).chunk(2, 1)
        mp, lp = self.prior(e).chunk(2, 1) if self.prior_kind == "service" else (torch.zeros_like(mq), torch.zeros_like(lq))
        lq, lp = lq.clamp(-8, 8), lp.clamp(-8, 8)  # log-variances
        z = mq + torch.randn_like(mq) * (0.5 * lq).exp() if self.training else mq
        mx, sx = self.dec(torch.cat([z, e], 1)).chunk(2, 1)
        kl = 0.5 * (lp - lq + (lq.exp() + (mq - mp) ** 2) / lp.exp() - 1)
        return mx, self.sigma_min + F.softplus(sx), kl, mq


def loss(net, x, m, s, beta):
    """Gaussian negative log-likelihood of the cells in the model, plus beta times the KL; mean over samples."""
    mx, sx, kl, _ = net(x, s)
    return ((m * (sx.log() + 0.5 * ((x - mx) / sx) ** 2)).sum(1) + beta * kl.sum(1)).mean()


class Orcrist:
    features = tuple(f for f in inputs.B0 if f != "restarts")

    def __init__(self, version):
        self.version = version
        self.params = {"version": version, **VERSIONS[version], "D": D, "H": H, "K": K, "SIGMA_MIN": SIGMA_MIN,
                       "INSTANCE_DROPOUT": INSTANCE_DROPOUT, "BATCH": BATCH, "LR": LR, "MAX_EPOCHS": MAX_EPOCHS,
                       "PATIENCE": PATIENCE, "KL_WARMUP": KL_WARMUP, "MIN_FINITE": MIN_FINITE, "MIN_SD": MIN_SD, "SEED": SEED}

    def _samples(self, A, drop_empty=False):
        """[T, S, F] -> (x, cells in the model and finite, service) of the T*S samples, x standardized, 0 where not.
        drop_empty leaves out the samples with no such cell (training)."""
        Z = (A - self.mu) / self.sd
        m = self.cols[None] & np.isfinite(Z)
        T, S, Fn = A.shape
        t = lambda a: torch.as_tensor(a.reshape(T * S, Fn), dtype=torch.float32)
        x, m, s = t(np.where(m, Z, 0)), t(m), torch.arange(S).repeat(T)
        keep = m.any(1) if drop_empty else slice(None)
        return x[keep], m[keep], s[keep]

    def fit(self, X, val, services):
        """X, val: lists of [T, S, len(features)] arrays from inputs.prepare, healthy loaded steps; val stops training."""
        torch.manual_seed(SEED)
        torch.set_num_threads(THREADS)
        rng = np.random.default_rng(SEED)
        A = np.concatenate(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells: features a service does not have
            self.mu, sd = np.nanmean(A, (0, 1)), np.nanstd(A, (0, 1))
            self.cols = (np.isfinite(A).mean(0) >= MIN_FINITE) & (np.nan_to_num(np.nanstd(A, 0)) > MIN_SD)
        self.sd = np.where(sd > MIN_SD, sd, 1.0)
        self.net = Net([TYPES.index(service_type(s)) for s in services], len(self.features),
                       VERSIONS[self.version]["condition"], VERSIONS[self.version]["prior"], SIGMA_MIN / self.sd)
        tx, tm, ts = self._samples(A, drop_empty=True)
        vx, vm, vs = self._samples(np.concatenate(val), drop_empty=True)
        opt = torch.optim.Adam(self.net.parameters(), lr=LR)
        best, history, start = (math.inf, 0, None), [], time.time()
        for epoch in range(MAX_EPOCHS):
            self.net.train()
            beta = min(1.0, (epoch + 1) / KL_WARMUP)
            for i in np.array_split(rng.permutation(len(tx)), math.ceil(len(tx) / BATCH)):
                opt.zero_grad()
                loss(self.net, tx[i], tm[i], ts[i], beta).backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                v = float(loss(self.net, vx, vm, vs, 1.0))
            history.append(v)
            if v < best[0]:
                best = (v, epoch, copy.deepcopy(self.net.state_dict()))
            elif epoch - best[1] >= PATIENCE and epoch >= KL_WARMUP:
                break
        self.net.load_state_dict(best[2])
        self.net.eval()
        with torch.no_grad():
            _, sx, kl, _ = self.net(tx, ts)
            train_loss = float(loss(self.net, tx, tm, ts, 1.0))
        S = len(services)
        self.sigma = np.array([(sx[ts == i] * torch.as_tensor(self.sd, dtype=torch.float32)).mean(0).numpy() for i in range(S)])
        self.kl_dims = np.array([kl[ts == i].mean(0).numpy() for i in range(S)])
        self.params.update(epochs=len(history), best_epoch=best[1] + 1, val_loss=round(best[0], 4),
                           train_loss=round(train_loss, 4), seconds=round(time.time() - start, 1),
                           parameters=sum(p.numel() for p in self.net.parameters()),
                           active_dims=int((kl.mean(0) > 0.01).sum()), val_curve=[round(h, 4) for h in history])
        return self

    def _forward(self, X):
        """(x, m, mean, deviation) [T, S, F] and (KL per dimension, posterior mean) [T, S, K] of one run."""
        x, m, s = self._samples(X)
        with torch.no_grad():
            mx, sx, kl, mq = self.net(x, s)
        T, S, Fn = X.shape
        return (x.numpy().reshape(T, S, Fn), m.numpy().reshape(T, S, Fn).astype(bool), mx.numpy().reshape(T, S, Fn),
                sx.numpy().reshape(T, S, Fn), kl.numpy().reshape(T, S, K), mq.numpy().reshape(T, S, K))

    def score(self, X):
        """One run [T, S, F] -> (r2, expected), both [T, S, F]: each feature's squared residual in decoder deviations,
        0 where the cell is out of the model or missing (a missing value counts as expected), and the decoder's mean in
        prepare units, NaN out of the model."""
        x, m, mx, sx, _, _ = self._forward(X)
        r2 = np.where(m, ((x - mx) / sx) ** 2, 0.0)
        return r2, np.where(self.cols[None], mx * self.sd + self.mu, np.nan)

    def spread(self, X):
        """One run [T, S, F] -> (sd [T, S, F], the decoder deviation in prepare units, NaN out of the model;
        kl [T, S], the KL of each step's posterior from the prior)."""
        _, _, _, sx, kl, _ = self._forward(X)
        return np.where(self.cols[None], sx * self.sd, np.nan), kl.sum(2)

    def latent(self, X):
        """One run [T, S, F] -> (z, kl), both [T, S, K]: the posterior mean of the latent state and its KL from the
        prior per dimension. For the notebooks: what the latent state encodes."""
        _, _, _, _, kl, mq = self._forward(X)
        return mq, kl

    def card(self, services):
        """What the model learned, per service: its type, columns left out, condition vector, mean deviation per
        feature (prepare units) and mean KL per latent dimension on training."""
        e = None if self.net.kind == "onehot" else self.net.condition(torch.arange(len(services))).detach().numpy()
        return {s: {"type": service_type(s), "columns": int(self.cols[i].sum()),
                    "left_out": [f for f, c in zip(self.features, self.cols[i]) if not c],
                    "condition": None if e is None else [round(float(v), 3) for v in e[i]],
                    "sigma": {f: round(float(v), 4) for f, v, c in zip(self.features, self.sigma[i], self.cols[i]) if c},
                    "kl_per_dim": [round(float(v), 3) for v in self.kl_dims[i]]}
                for i, s in enumerate(services)}
