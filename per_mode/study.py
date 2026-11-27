import itertools
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from prep import gcm, T, MODES, build, DATA, RES
import models as M


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RES.mkdir(parents=True, exist_ok=True)
ALL = MODES + ["global"]

GRID = dict(max_depth=[2, 4, 6, 8], min_child_weight=[10, 50], reg_lambda=[1.0, 5.0, 10.0])
MASK_TOL = 0.05   # keep the physically clean masked model unless unmasked is >5% better on validation

# methods compared: switched, global xgboost, and persistence
KINDS = ["global_xgb", "switched_xgb_causal"]
STEP = pd.Timedelta("30min")
H_FREE = 48   # 24 h free-run
H_REC = 6   # 3 h recursive forecast
N_BOOT = 2000   # bootstrap resamples of the free-run days

# model selection by the 24 h free-run error on the validation days
DEFAULT = dict(max_depth=4, min_child_weight=50, reg_lambda=5.0)   # start of the coordinate search
TUNE_PATIENCE = 300   # one-step early stopping only caps the trees; the extra trees are kept for the free-run choice
TREE_EVERY = 50     # tree counts compared by free-run: every TREE_EVERY trees + the one-step optimum


def rmse(e): 
    return float(np.sqrt(np.mean(np.square(e))))


def mae(e): 
    return float(np.mean(np.abs(e)))


def load():
    G, feats, y, X, info = build(DATA)
    mono = dict(zip(X.columns, gcm.monotone_constraints(G, T, list(X.columns))))
    groups = G.nodes[T]["groups"]
    return G, feats, y, X, info, mono, groups


def rows(info, mode, split):
    """Steps of one mode (or all steps for 'global') in the given split(s)."""
    s = info.split.isin(split if isinstance(split, list) else [split])
    return s if mode == "global" else s & (info.mode_used == mode)


def kind_of(mode):
    """The method whose free-run a model is judged by."""
    return "global_xgb" if mode == "global" else "switched_xgb_causal"


def tree_counts(n_available, n_one_step):
    """Tree counts compared by free-run error."""
    return sorted(set(range(TREE_EVERY, n_available + 1, TREE_EVERY)) | {int(n_one_step)})


# tune hyperparameters, number of trees and masking by the 24 h free-run error on the validation days
def tune(G, feats, y, X, info, mono, groups):
    """
    Selection by 24 h free-run RMSE on validation: the models act together in the switched free-run, so the
    search is coordinate-wise. All models start from DEFAULT; then, global first (the switched model falls back
    on it where there is no valid mode) and each mode in turn, every grid point x mask x tree count is scored by
    the validation free-run of its method with the other models at their current choice; the best is kept.
    Trees are fitted on train with one-step early stopping (patience TUNE_PATIENCE) to cap the tree count.
    """
    ss_val = SimSet(G, feats, X.index, info, day_starts(info, ["val"]), H_FREE)
    print(f"  validation free-runs: {len(ss_val.starts)} days", flush=True)

    # start point
    cur, best = {}, {}
    for mode in ALL:
        tr, va = rows(info, mode, "train"), rows(info, mode, "val")
        cols = M.mode_columns(list(X.columns), mode, True)
        cur[mode] = M.XGBGraph(mono, groups, **DEFAULT).fit(X.loc[tr, cols], y[tr], X.loc[va, cols], y[va])
        best[mode] = dict(**DEFAULT, n_trees=int(cur[mode].best_n), mask=True)
    # print("  start: " + "  ".join(f"{k}={freerun_rmse(ss_val, cur, k):.4f}" for k in KINDS), flush=True)

    # coordinate search
    recs = []
    for mode in ["global"] + MODES:
        kind = kind_of(mode)
        tr, va = rows(info, mode, "train"), rows(info, mode, "val")
        top = {}    # mask
        for mask in ([True, False] if M.INACTIVE.get(mode) else [True]):
            cols = M.mode_columns(list(X.columns), mode, mask)
            for md, mcw, lam in itertools.product(*GRID.values()):
                m = M.XGBGraph(mono, groups, max_depth=md, min_child_weight=mcw, reg_lambda=lam,
                               early_stopping_rounds=TUNE_PATIENCE)
                m.fit(X.loc[tr, cols], y[tr], X.loc[va, cols], y[va])
                for n in tree_counts(m.n_available, m.best_n):
                    m.best_n = n
                    err = freerun_rmse(ss_val, {**cur, mode: m}, kind)
                    s = dict(max_depth=md, min_child_weight=mcw, reg_lambda=lam, n_trees=n, mask=mask)
                    recs.append(dict(mode=mode, **s, val_fr24_rmse=err,
                                     val_rmse=rmse(y[va] - m.predict(X.loc[va, cols]))))
                    if mask not in top or err < top[mask][0]:
                        top[mask] = (err, s, m)
        err, s, m = top[True]
        if False in top and top[False][0] < (1 - MASK_TOL) * err:
            err, s, m = top[False]
        m.best_n = s["n_trees"]
        cur[mode], best[mode] = m, s
        print(f"  tuned {mode:15s} {s}  val_fr24_rmse={err:.4f}", flush=True)
    pd.DataFrame(recs).to_csv(RES / "tuning.csv", index=False)
    json.dump(best, open(RES / "best_params.json", "w"), indent=1)
    return best


# fit
def fit_models(y, X, info, mono, groups, best, splits):
    """XGB per mode (+ 'other' + global) fitted on splits"""
    xg = {}
    for mode in ALL:
        b = best[mode]
        cols = M.mode_columns(list(X.columns), mode, b["mask"])
        fr = rows(info, mode, splits)
        n_scale = fr.sum() / rows(info, mode, "train").sum()
        xg[mode] = M.XGBGraph(mono, groups, max_depth=b["max_depth"], min_child_weight=b["min_child_weight"],
                              reg_lambda=b["reg_lambda"]).fit(X.loc[fr, cols], y[fr],
                                                               n_estimators=int(round(b["n_trees"] * n_scale)))
    return xg


def predict_by_mode(models, X, modes):
    """Each row predicted by the model of its mode; rows without a model of their own by the global model."""
    p = np.full(len(X), np.nan)
    for mode, mdl in models.items():
        if mode == "global":
            continue
        s = (modes == mode).values
        if s.any():
            p[s] = mdl.predict(X.loc[s, mdl.cols])
    s = np.isnan(p)
    if s.any():
        p[s] = models["global"].predict(X.loc[s, models["global"].cols])
    return p


def switch_modes(info):
    """
    Model that the switch uses for each step, from past data only: the model of the step's mode (at t-1) from
    the first step of the mode; 'other' where there is no valid mode.
    """
    return info["mode"].where(info["mode"].isin(MODES), "other")


def assemble(xg, X, rows, kind):
    """Prediction of one method; 'rows' = the info rows of the steps in X."""
    if kind == "global_xgb":
        return xg["global"].predict(X[xg["global"].cols])
    if kind == "switched_xgb_causal":
        return predict_by_mode(xg, X, switch_modes(rows))
    raise ValueError(kind)


# recursive simulation
def method_fn(xg, kind):
    return lambda Xk, rows: assemble(xg, Xk, rows, kind)


def day_starts(info, splits, horizon=H_FREE):
    """Midnights whose whole free-run (steps 1..horizon) lies in the given split(s)."""
    days = pd.date_range(info.index.min().normalize(), info.index.max().normalize(), freq="D")
    return [d for d in days
            if info["split"].reindex(pd.date_range(d + STEP, periods=horizon, freq=STEP)).isin(splits).all()]


class SimSet:
    """
    Recursive simulations from fixed starts: from the measured temperature at each start, roll forward 'horizon'
    steps feeding the predicted temperature back into indoor_temperature_lag1 and the graph's derived terms
    (recomputed with gcm.design_matrix). Valves, dampers, supply temperatures, weather, CO2 and calendar stay
    measured. The measured inputs are gathered once, so many models can be simulated cheaply.
    """

    def __init__(self, G, feats, X_index, info, starts, horizon):
        idx = pd.DatetimeIndex(X_index)
        T0 = feats[T]
        self.G, self.horizon = G, horizon
        self.starts = pd.DatetimeIndex(
            [s for s in starts
             if pd.notna(T0.get(s, np.nan)) and all((s + k * STEP) in idx for k in range(1, horizon + 1))])
        self.T0 = T0.reindex(self.starts).values.astype(float)          # T at the start
        cols = gcm.mechanism_inputs(G, T)
        self.inp, self.rows = [], []
        self.obs = np.full((len(self.starts), horizon), np.nan)
        for k in range(1, horizon + 1):
            tk = self.starts + k * STEP
            self.inp.append(feats.loc[tk, cols])
            self.rows.append(info.reindex(tk))
            self.obs[:, k - 1] = T0.reindex(tk).values

    def run(self, models_fn):
        """Simulated T (starts x horizon)."""
        Tsim = self.T0.copy()
        out = np.full_like(self.obs, np.nan)
        for k in range(self.horizon):
            Xk = gcm.design_matrix(self.G, T, self.inp[k].assign(**{f"{T}_lag1": Tsim}))
            Tsim = Tsim + models_fn(Xk, self.rows[k])
            out[:, k] = Tsim
        return out


def freerun_rmse(ss, xg, kind):
    out = ss.run(method_fn(xg, kind))
    v = ~np.isnan(ss.obs)
    return rmse((out - ss.obs)[v])


def simulate(G, feats, X_index, models_fn, info, starts, horizon):
    ss = SimSet(G, feats, X_index, info, starts, horizon)
    return ss.starts, ss.run(models_fn), ss.obs


def run_sims(G, feats, X, info, xg, starts, horizon):
    """Simulate every method + persistence from the same starts."""
    ss = SimSet(G, feats, X.index, info, starts, horizon)
    sims = {k: ss.run(method_fn(xg, k)) for k in KINDS}
    sims["persistence"] = np.repeat(ss.T0[:, None], horizon, axis=1)
    return ss.starts, sims, ss.obs


def sim_metrics(sims, obs):
    """Overall RMSE and MAE of the simulated indoor temperature, pooled over all starts and all steps."""
    v = ~np.isnan(obs)
    return pd.DataFrame({k: {"rmse_K": rmse((p - obs)[v]), "mae_K": mae((p - obs)[v])}
                         for k, p in sims.items()}).T


def paired_bootstrap(sims, obs, pairs, n_boot=N_BOOT, seed=0):
    """
    Difference in pooled RMSE, a - b (negative = a better), with a 95% interval from resampling whole
    simulations (days) with replacement; both methods get the same resampled days.
    """
    v = ~np.isnan(obs)
    n = v.sum(axis=1)
    sse = {k: np.where(v, (p - obs) ** 2, 0.0).sum(axis=1) for k, p in sims.items()}
    idx = np.random.default_rng(seed).integers(0, len(obs), (n_boot, len(obs)))
    out = []
    for a, b in pairs:
        d = np.sqrt(sse[a][idx].sum(1) / n[idx].sum(1)) - np.sqrt(sse[b][idx].sum(1) / n[idx].sum(1))
        out.append(dict(a=a, b=b, rmse_a_K=np.sqrt(sse[a].sum() / n.sum()), rmse_b_K=np.sqrt(sse[b].sum() / n.sum()),
                        diff_K=np.sqrt(sse[a].sum() / n.sum()) - np.sqrt(sse[b].sum() / n.sum()),
                        ci95_lo=np.percentile(d, 2.5), ci95_hi=np.percentile(d, 97.5),
                        share_a_better=float((d < 0).mean())))
    return pd.DataFrame(out)


def main(stage="all"):
    t0 = time.time()
    G, feats, y, X, info, mono, groups = load()
    print(f"data ready: {len(y)} steps  ({time.time()-t0:.0f}s)", flush=True)

    if stage in ("all", "tune") or not (RES / "best_params.json").exists():
        best = tune(G, feats, y, X, info, mono, groups)
    else:
        best = json.load(open(RES / "best_params.json"))

    # final models on train + val
    xg = fit_models(y, X, info, mono, groups, best, ["train", "val"])
    pd.to_pickle(dict(xg=xg, best=best), RES / "final_models.pkl")

    # 3 h recursive forecast on 2024, starting at every 30-min step
    starts = list(X.index[info.split == "test"])
    ok, sims, obs = run_sims(G, feats, X, info, xg, starts, H_REC)
    pd.to_pickle(dict(starts=ok, obs=obs, **sims), RES / "recursive3h_test.pkl")
    summ = sim_metrics(sims, obs)
    summ.to_csv(RES / "recursive3h_test.csv")
    print(f"\n3 h recursive over {len(ok)} starts in 2024:\n", summ.round(3).to_string(), flush=True)
    print(f"done in {time.time()-t0:.0f}s")

    # 24 h free-run in 2024
    starts = list(pd.date_range("2024-01-01", "2024-09-29", freq="D"))
    ok, sims, obs = run_sims(G, feats, X, info, xg, starts, H_FREE)
    pd.to_pickle(dict(starts=ok, obs=obs, **sims), RES / "freerun_test.pkl")
    summ = sim_metrics(sims, obs)
    summ.to_csv(RES / "freerun_test.csv")
    print(f"\n24 h free-run over {len(ok)} days of 2024:\n", summ.round(3).to_string(), flush=True)
    # bs = paired_bootstrap(sims, obs, [("switched_xgb_causal", "global_xgb"),
    #                                   ("switched_xgb_causal", "persistence")])
    # bs.to_csv(RES / "freerun_test_bootstrap.csv", index=False)
    # print("\npaired bootstrap over days (RMSE a - b, K):\n", bs.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
