
import argparse
import pickle
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from GCM_generation import (T, EDGES, COMPONENTS, INTERACTIONS, INTERACTION_INPUTS, GROUPS, MONOTONE,
                            build_gcm, make_features, mechanism_inputs, design_matrix, node_frame,
                            edge_bundles, monotone_constraints, interaction_groups)
from filter_modes import MIN_RUN_H, DT_H


HERE = Path(__file__).resolve().parent
HEAT_ON = 20.0     # heating_valve % above which the radiator counts as on

# components (or single variables) whose edges into the room temperature are cut in each mode
MODES = {
    "off_no_sun": {"inactive": ["chilled_beam", "radiator", "solar_radiation"]},
}


# Graph of the mode
def mode_config(inactive):
    """
    GCM config holding only the mechanism of the room temperature, without the edges of the inactive
    components or variables; interactions, additive groups, monotone signs and components are reduced
    to the remaining columns.
    """
    drop = {v for c in inactive for v in COMPONENTS.get(c, [c])}
    edges = [(p, c, f) for p, c, f in EDGES if c == T and p not in drop]
    inputs = {f for _, _, feats in edges for f in feats}
    names = lambda expr: set(re.findall(r"[A-Za-z_]\w*", expr))
    inter = {k: e for k, e in INTERACTIONS[T].items() if names(e) <= inputs}
    used = set().union(*map(names, inter.values()))
    hidden = [c for c in INTERACTION_INPUTS[T] if c in used]
    columns = (inputs - set(hidden)) | set(inter)
    groups = {g: [c for c in cols if c in columns] for g, cols in GROUPS[T].items() if g not in inactive}
    groups = {g: cols for g, cols in groups.items() if cols}
    monotone = {c: s for c, s in MONOTONE[T].items() if c in columns}
    components = {c: [v for v in vs if v not in drop] for c, vs in COMPONENTS.items() if c not in inactive}
    components = {c: vs for c, vs in components.items() if vs}
    return dict(edges=edges, components=components, interactions={T: inter}, interaction_inputs={T: hidden},
                groups={T: groups}, monotone={T: monotone}, lag_node_sources=[T], make_plots=False)


# Data
def mode_rows(info, mode, min_run_h, skip_first_h):
    """
    Rows t whose step t-1 -> t lies inside one run of the mode: the run lasts at least min_run_h, t-1 is
    in the same run, and the first skip_first_h hours after the switch (rebound of the building mass
    after the change of HVAC operation) are left out.
    """
    sel = ((info["mode"] == mode) & (info["run_len_h"] >= min_run_h)
           & (info["run_id"] == info["run_id"].shift(1))
           & (info["t_in_mode_h"] >= max(skip_first_h, DT_H)))
    if "heating_valve" in info:
        sel &= (info["heating_valve"] <= HEAT_ON) & (info["heating_valve"].shift(1) <= HEAT_ON)
    return sel.fillna(False)


def segments(index, run_id):
    """Id of every contiguous stretch of rows (no gap in time, same run)."""
    t = index.to_series()
    new = (t.diff() != pd.Timedelta(hours=DT_H)) | (run_id != run_id.shift(1))
    return pd.Series(new.cumsum().to_numpy(), index=index, name="segment")


def load_mode_data(room, mode, cfg, min_run_h, skip_first_h):
    src = HERE / f"{room}_data" / "all_with_modes.pkl"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found, run filter_modes.py {room} first")
    df = pd.read_pickle(src).sort_index()
    G = build_gcm(cfg)
    feats = make_features(df, G, cfg)            # lags and means from the full series, on a regular grid
    keep = [c for c in ["mode", "run_id", "run_len_h", "t_in_mode_h", "heating_valve"] if c in df]
    info = df[keep].reindex(feats.index)
    sel = mode_rows(info, mode, min_run_h, skip_first_h)

    cols = list(dict.fromkeys([T] + mechanism_inputs(G, T)))
    data = feats.loc[sel, cols].dropna()
    seg = segments(data.index, info.loc[data.index, "run_id"])
    return G, data, seg, info.loc[data.index, "run_id"]


# Model
def make_model(G, columns, a):
    return HistGradientBoostingRegressor(
        learning_rate=a.lr, max_iter=a.max_iter, max_leaf_nodes=a.max_leaf_nodes,
        min_samples_leaf=a.min_samples_leaf, l2_regularization=a.l2,
        monotonic_cst=monotone_constraints(G, T, columns),
        interaction_cst=interaction_groups(G, T, columns) if a.interaction_cst else None,
        early_stopping=False, random_state=0)


def forecast(model, G, data, seg, steps):
    """
    Multi-step forecast from every origin whose next 'steps' rows lie in one segment: the first step
    uses the observed T(t-1), the later steps the predicted temperature. One row per forecast, indexed
    by its target time: the start temperature, the observed and the predicted temperature 'steps' ahead.
    """
    lag = f"{T}_lag1"
    k = data.groupby(seg).cumcount().to_numpy()
    n = seg.map(seg.value_counts()).to_numpy()
    origins = np.flatnonzero(k + steps <= n)                # rows of a segment are contiguous
    T_start = data[lag].to_numpy()[origins]
    state = T_start.copy()
    for j in range(steps):
        d = data.iloc[origins + j].copy()
        d[lag] = state
        state = state + model.predict(design_matrix(G, T, d))
    end = origins + steps - 1
    return pd.DataFrame({"T_start": T_start, "T_obs": data[T].to_numpy()[end], "T_pred": state},
                        index=data.index[end])


def edge_importance(model, G, data, seg, steps, rng, n_repeats=5):
    """Increase of the forecast MSE when the columns of one incoming edge are permuted together (edge cut)."""
    def mse(d):
        f = forecast(model, G, d, seg, steps)
        return np.mean((f.T_obs - f.T_pred) ** 2)

    base = mse(data)
    out = {}
    for parent, cols in edge_bundles(G, T).items():
        cut = []
        for _ in range(n_repeats):
            d = data.copy()
            d[cols] = d[cols].to_numpy()[rng.permutation(len(d))]
            cut.append(mse(d))
        out[parent] = np.mean(cut) - base
    return pd.Series(out)


def cross_validate(G, data, y, X, seg, groups, steps, a):
    """Out-of-fold multi-step forecasts and edge importances, folds = whole runs."""
    rng = np.random.default_rng(0)
    fc, imp = [], []
    n_splits = min(a.folds, groups.nunique())
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = make_model(G, X.columns, a).fit(X.iloc[tr], y.iloc[tr])
        fc.append(forecast(m, G, data.iloc[te], seg.iloc[te], steps))
        imp.append(edge_importance(m, G, data.iloc[te], seg.iloc[te], steps, rng))
    return pd.concat(fc).sort_index(), pd.concat(imp, axis=1).mean(axis=1).sort_values(ascending=False)


def metrics(fc, horizon_h):
    """Errors of the forecasts, next to persistence (T stays at the start temperature)."""
    rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
    change = fc.T_obs - fc.T_start
    err = {"model": fc.T_obs - fc.T_pred, "persistence": change}
    rows = {
        f"RMSE of T at {horizon_h:g} h [K]": {k: rmse(e) for k, e in err.items()},
        "MAE [K]": {k: float(e.abs().mean()) for k, e in err.items()},
        # "Mean errors (obs - pred) [K]": {k: float(e.mean()) for k, e in err.items()},
        f"R2 of the {horizon_h:g}-h change": {k: 1 - np.mean(e ** 2) / np.var(change) for k, e in err.items()},
        "forecasts": {k: len(e) for k, e in err.items()},
    }
    return pd.DataFrame(rows).T


# Plot
def plot_fit(room, mode, data, fc, seg, horizon_h):
    """Out-of-fold forecast of the temperature change, and the forecasts in the 4 longest segments."""
    fig = plt.figure(figsize=(14, 8), layout="constrained")
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[:, 0])
    obs, pred = fc.T_obs - fc.T_start, fc.T_pred - fc.T_start
    ax.scatter(obs, pred, s=4, alpha=0.4, color="#1f3b73")
    lim = np.nanpercentile(np.abs(np.r_[obs, pred]), 99.5)
    ax.plot([-lim, lim], [-lim, lim], color="#ff9f1c", lw=1)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel=f"observed {horizon_h:g}-h change of T [K]",
           ylabel="out-of-fold predicted change [K]", title=f"{room} {mode}: {horizon_h:g} h ahead")

    fc_seg = seg.loc[fc.index]
    for i, s in enumerate(seg.value_counts().index[:4]):
        ax = fig.add_subplot(gs[i // 2, 1 + i % 2])
        rows = (seg == s).to_numpy()
        f = fc[(fc_seg == s).to_numpy()]
        ax.plot(data.index[rows], data.loc[rows, T], color="#222222", lw=1.4, label="observed")
        ax.plot(f.index, f.T_pred, color="#1f3b73", ls="--", marker=".", lw=1.2,
                label=f"forecast made {horizon_h:g} h earlier")
        ax.set_title(f"{data.index[rows][0]:%Y-%m-%d %H:%M}", fontsize=9, loc="left")
        ax.tick_params(axis="x", labelsize=7, rotation=30)
        ax.set_ylabel("°C")
        if i == 0:
            ax.legend(fontsize=8)
    return fig


# Main
def main(argv=None):
    p = argparse.ArgumentParser(description="Fit the room-temperature mechanism of one mode with gradient boosting.")
    p.add_argument("room", nargs="?", default="07-D4-01", help="room id")
    p.add_argument("--mode", choices=list(MODES), default="off_no_sun")
    p.add_argument("--min-run-h", type=float, default=MIN_RUN_H, help="shortest run of the mode kept [h]")
    p.add_argument("--skip-first-h", type=float, default=1.0,
                   help="hours left out after the switch into the mode (rebound) [h]")
    p.add_argument("--horizon-h", type=float, default=3.0, help="forecast horizon of the evaluation [h]")
    p.add_argument("--folds", type=int, default=5, help="cross-validation folds, grouped by run")
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--max-leaf-nodes", type=int, default=15)
    p.add_argument("--min-samples-leaf", type=int, default=20)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--no-interaction-cst", dest="interaction_cst", action="store_false",
                   help="let all columns interact instead of the additive groups of the graph")
    p.add_argument("--show", action="store_true", help="show the figure")
    a = p.parse_args(argv)

    cfg = mode_config(MODES[a.mode]["inactive"])
    G, data, seg, run_id = load_mode_data(a.room, a.mode, cfg, a.min_run_h, a.skip_first_h)
    y, X = node_frame(G, T, data)
    # print(f"{a.room} {a.mode}: {len(y)} steps ({len(y) * DT_H:.0f} h) in {run_id.nunique()} runs, "
    #       f"{seg.nunique()} contiguous segments")
    # print(f"mechanism of {T}: dT <- {', '.join(X.columns)}")
    # print("additive groups:", {g: c for g, c in G.nodes[T]["groups"].items()} if a.interaction_cst else "none")
    # print("monotone:", G.nodes[T]["monotone"])

    steps = int(round(a.horizon_h / DT_H))
    fc, imp = cross_validate(G, data, y, X, seg, run_id, steps, a)
    res = metrics(fc, a.horizon_h)
    pd.set_option("display.width", 200)
    print(f"\n{a.horizon_h:g}-h forecast errors:\n" + res.round(4).to_string())
    print(f"\n{a.horizon_h:g}-h forecast MSE increase when an edge is cut (out of fold):\n"
          + imp.to_string(float_format="%.2e"))

    model = make_model(G, X.columns, a).fit(X, y)

    out = HERE / f"{a.room}_data" / f"fit_{a.mode}"
    out.mkdir(parents=True, exist_ok=True)
    # with open(out / "model.pkl", "wb") as f:
    #     pickle.dump({"model": model, "columns": list(X.columns), "config": cfg, "args": vars(a)}, f)
    res.to_csv(out / "metrics.csv")
    # imp.rename("mse_increase").to_csv(out / "edge_importance.csv")
    # fc.to_pickle(out / "oof_forecasts.pkl")

    fig_dir = HERE / "figure"
    fig_dir.mkdir(exist_ok=True)
    fig = plot_fit(a.room, a.mode, data, fc, seg, a.horizon_h)
    fig.savefig(fig_dir / f"{a.room}_fit_{a.mode}.png", dpi=300)
    print(f"\nResults to {fig_dir / f'{a.room}_fit_{a.mode}.png'}")
    if a.show:
        plt.show()


if __name__ == "__main__":
    main()
