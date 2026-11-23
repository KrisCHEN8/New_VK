"""
Per-mode modelling study of the indoor_temperature mechanism.

Stages (results cached in results/):
  1. tune     per mode + global: XGBoost (graph constraints) grid with early stopping on 2023 H2,
              linear (sign-constrained), structural-zero masking on/off, unconstrained XGB as reference
  2. final    refit chosen models on 2022-03..2023-12, evaluate one-step on 2024 (test)
  3. assemble global vs switched vs blended; one-step and 24 h free-run simulation on 2024
  4. attribute exact group decomposition of the assembled model, per mode and component
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prep import gcm, T, MODES, build, DATA, RES   # noqa: E402
import models as M                                  # noqa: E402

RES.mkdir(parents=True, exist_ok=True)
ALL = MODES + ["other", "global"]      # "other" = steps outside the saved mode runs (transitions)

GRID = dict(max_depth=[2, 3, 4, 6, 8], min_child_weight=[10, 50], reg_lambda=[1.0, 10.0])
MASK_TOL = 0.02   # keep the physically clean masked model unless unmasked is >2% better on validation


def rmse(e): return float(np.sqrt(np.mean(np.square(e))))
def mae(e): return float(np.mean(np.abs(e)))
def r2(y, p): return float(1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2))


def load():
    G, feats, y, X, info = build(DATA)
    mono = dict(zip(X.columns, gcm.monotone_constraints(G, T, list(X.columns))))
    groups = G.nodes[T]["groups"]
    return G, feats, y, X, info, mono, groups


def rows(info, mode, split):
    """Steps of one mode (or all steps for 'global') in the given split(s)."""
    s = info.split.isin(split if isinstance(split, list) else [split])
    return s if mode == "global" else s & (info.mode_used == mode)


# ------------------------------------------------------------------ 1. tune
def tune(y, X, info, mono, groups):
    recs, best = [], {}
    for mode in ALL:
        tr, va = rows(info, mode, "train"), rows(info, mode, "val")
        for mask in ([True, False] if M.INACTIVE.get(mode) else [True]):
            cols = M.mode_columns(list(X.columns), mode, mask)
            # linear
            lin = M.LinearPhys(mono, groups).fit(X.loc[tr, cols], y[tr])
            recs.append(dict(mode=mode, model="linear", mask=mask, val_rmse=rmse(y[va] - lin.predict(X.loc[va, cols]))))
            # xgb grid
            for md, mcw, lam in itertools.product(*GRID.values()):
                m = M.XGBGraph(mono, groups, max_depth=md, min_child_weight=mcw, reg_lambda=lam)
                m.fit(X.loc[tr, cols], y[tr], X.loc[va, cols], y[va])
                recs.append(dict(mode=mode, model="xgb", mask=mask, max_depth=md, min_child_weight=mcw,
                                 reg_lambda=lam, n_trees=m.best_n, val_rmse=rmse(y[va] - m.predict(X.loc[va, cols]))))
        cand = pd.DataFrame([x for x in recs if x["mode"] == mode and x["model"] == "xgb"]).sort_values("val_rmse")
        r = cand[cand["mask"]].iloc[0]
        if (~cand["mask"]).any() and cand[~cand["mask"]].val_rmse.iloc[0] < (1 - MASK_TOL) * r.val_rmse:
            r = cand[~cand["mask"]].iloc[0]
        best[mode] = dict(max_depth=int(r.max_depth), min_child_weight=int(r.min_child_weight),
                          reg_lambda=float(r.reg_lambda), n_trees=int(r.n_trees), mask=bool(r["mask"]))
        # unconstrained reference with the same settings
        cols = M.mode_columns(list(X.columns), mode, best[mode]["mask"])
        u = M.XGBGraph(mono, groups, constrained=False, max_depth=best[mode]["max_depth"],
                       min_child_weight=best[mode]["min_child_weight"], reg_lambda=best[mode]["reg_lambda"])
        u.fit(X.loc[tr, cols], y[tr], X.loc[va, cols], y[va])
        recs.append(dict(mode=mode, model="xgb_unconstrained", mask=best[mode]["mask"],
                         val_rmse=rmse(y[va] - u.predict(X.loc[va, cols]))))
        print(f"  tuned {mode:15s} {best[mode]}  val_rmse={r.val_rmse:.4f}", flush=True)
    pd.DataFrame(recs).to_csv(RES / "tuning.csv", index=False)
    json.dump(best, open(RES / "best_params.json", "w"), indent=1)
    return best


# ------------------------------------------------------------------ 2. fit
def fit_models(y, X, info, mono, groups, best, splits):
    """XGB and linear per mode (+ global) fitted on 'splits'. Trees scaled to the larger data when refitting."""
    xg, li = {}, {}
    for mode in ALL:
        b = best[mode]
        cols = M.mode_columns(list(X.columns), mode, b["mask"])
        fr = rows(info, mode, splits)
        n_scale = fr.sum() / rows(info, mode, "train").sum()
        xg[mode] = M.XGBGraph(mono, groups, max_depth=b["max_depth"], min_child_weight=b["min_child_weight"],
                              reg_lambda=b["reg_lambda"]).fit(X.loc[fr, cols], y[fr],
                                                               n_estimators=int(round(b["n_trees"] * n_scale)))
        li[mode] = M.LinearPhys(mono, groups).fit(X.loc[fr, cols], y[fr])
    return xg, li


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


def assemble(xg, li, X, modes, weights=None, kind="switched_xgb"):
    g_x = xg["global"].predict(X[xg["global"].cols])
    if kind == "global_xgb":
        return g_x
    if kind == "global_linear":
        return li["global"].predict(X[li["global"].cols])
    no_other = lambda d: {k: v for k, v in d.items() if k != "other"}
    if kind == "switched_linear":
        return predict_by_mode(no_other(li), X, modes)
    if kind == "switched_linear_other":
        return predict_by_mode(li, X, modes)
    if kind == "switched_xgb":
        return predict_by_mode(no_other(xg), X, modes)
    s_x = predict_by_mode(xg, X, modes)
    if kind == "switched_xgb_other":
        return s_x
    if kind == "blend_xgb":                      # per-mode weight on the mode/other model, rest on the global model
        w = modes.map(weights).fillna(0.0).values
        return w * s_x + (1 - w) * g_x
    raise ValueError(kind)


def blend_weights(xg, y, X, info):
    """Weight per mode chosen on the validation half-year (models fitted on train only)."""
    w = {}
    va = info.split == "val"
    g = xg["global"].predict(X.loc[va, xg["global"].cols])
    s = predict_by_mode(xg, X.loc[va], info.mode_used[va])
    for mode in MODES + ["other"]:
        k = (info.mode_used[va] == mode).values
        grid = np.linspace(0, 1, 21)
        errs = [rmse(y[va].values[k] - (a * s[k] + (1 - a) * g[k])) for a in grid]
        w[mode] = float(grid[int(np.argmin(errs))])
    return w


# ------------------------------------------------------------------ 3. free-run simulation
def simulate(G, feats, X_index, models_fn, info, starts, horizon=48):
    """
    Free-run: from the measured temperature at each start, roll forward 'horizon' steps feeding the predicted
    temperature back into indoor_temperature_lag1 and the graph's derived terms (recomputed with
    gcm.design_matrix). Valves, dampers, supply temperatures, weather, CO2 and calendar stay measured.
    """
    inputs_cols = gcm.mechanism_inputs(G, T)
    step = pd.Timedelta("30min")
    idx = pd.DatetimeIndex(X_index)
    ok = [s for s in starts if all((s + k * step) in idx for k in range(1, horizon + 1))]
    Tsim = feats[T].reindex(ok).values.astype(float)            # T at the start
    out = np.full((len(ok), horizon), np.nan)
    obs = np.full((len(ok), horizon), np.nan)
    for k in range(1, horizon + 1):
        tk = pd.DatetimeIndex([s + k * step for s in ok])
        inp = feats.loc[tk, inputs_cols].copy()
        inp[f"{T}_lag1"] = Tsim
        Xk = gcm.design_matrix(G, T, inp)
        dT = models_fn(Xk, info.mode_used.reindex(tk))
        Tsim = Tsim + dT
        out[:, k - 1] = Tsim
        obs[:, k - 1] = feats[T].reindex(tk).values
    return pd.DatetimeIndex(ok), out, obs


def main(stage="all"):
    t0 = time.time()
    G, feats, y, X, info, mono, groups = load()
    print(f"data ready: {len(y)} steps  ({time.time()-t0:.0f}s)", flush=True)

    if stage in ("all", "tune") or not (RES / "best_params.json").exists():
        best = tune(y, X, info, mono, groups)
    else:
        best = json.load(open(RES / "best_params.json"))

    # models fitted on train only -> blend weights on validation
    xg_tr, li_tr = fit_models(y, X, info, mono, groups, best, ["train"])
    w = blend_weights(xg_tr, y, X, info)
    print("blend weights (on mode model):", w, flush=True)
    # final models on train + val
    xg, li = fit_models(y, X, info, mono, groups, best, ["train", "val"])
    pd.to_pickle(dict(xg=xg, li=li, best=best, weights=w), RES / "final_models.pkl")

    # ---- one-step evaluation on 2024
    te = info.split == "test"
    yt, Xt, mt = y[te], X[te], info.mode_used[te]
    train_mean = y[info.split.isin(["train", "val"])].groupby(info.mode_used).mean()
    kinds = ["global_linear", "switched_linear", "switched_linear_other", "global_xgb", "switched_xgb",
             "switched_xgb_other", "blend_xgb"]
    preds = {k: assemble(xg, li, Xt, mt, w, k) for k in kinds}
    preds["zero_change"] = np.zeros(len(yt))
    preds["mode_mean"] = mt.map(train_mean).values
    recs = []
    for k, p in preds.items():
        for mode in MODES + ["other", "ALL"]:
            s = np.ones(len(yt), bool) if mode == "ALL" else (mt == mode).values
            recs.append(dict(method=k, mode=mode, n=int(s.sum()), rmse=rmse(yt.values[s] - p[s]),
                             mae=mae(yt.values[s] - p[s]), r2=r2(yt.values[s], p[s])))
    onestep = pd.DataFrame(recs)
    onestep.to_csv(RES / "onestep_test.csv", index=False)
    print(onestep.pivot(index="method", columns="mode", values="r2").round(3).to_string(), flush=True)

    # per-mode model on its own mode vs global model on the same steps (does specialising help?)
    # ---- 24 h free-run on 2024, starting every midnight
    starts = [d for d in pd.date_range("2024-01-01", "2024-09-29", freq="D")]
    sims = {}
    for k in kinds:
        fn = (lambda kk: (lambda Xk, modes: assemble(xg, li, Xk, modes, w, kk)))(k)
        ok, sim, obs = simulate(G, feats, X.index, fn, info, starts)
        sims[k] = sim
    sims["persistence"] = np.repeat(feats[T].reindex(ok).values[:, None], 48, axis=1)
    sims["obs"] = obs
    pd.to_pickle(dict(starts=ok, **sims), RES / "freerun_test.pkl")
    fr = pd.DataFrame({k: [rmse((v - obs)[~np.isnan(obs)]), float(np.nanmean(np.abs(v - obs)[:, -1]))]
                       for k, v in sims.items() if k != "obs"}, index=["rmse_24h_K", "mae_at_24h_K"]).T
    fr.to_csv(RES / "freerun_test.csv")
    print(f"\nfree-run over {len(ok)} days of 2024:\n", fr.round(3).to_string(), flush=True)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
