import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence
from sklearn.metrics import mean_pinball_loss

# Step 0, preprocessing, event grouping, hints and plot panels are shared with the
# binning approach (Option 1); its CONFIG holds the shared parameters
import conditional_dist as cd
from conditional_dist import REGIMES, DIRECTION, _localize


# tunable parameters: conditional_dist.CONFIG plus the overrides/additions below
CONFIG = cd._merge_config({
    "room": "07-D2-02",
    "resample_freq": "15min",
    "occupied_start": "07:00",

    # features (trailing, time-based windows)
    "rolling_features": {
        "T_out_3h": ("T_out", "3h"),
        "T_out_24h": ("T_out", "24h"),
        "solar_2h": ("solar", "2h"),
        "solar_6h": ("solar", "6h"),
    },
    "features": ["T_out_3h", "T_out_24h", "solar_2h", "solar_6h",
                 "hour_sin", "hour_cos", "setpoint"],
    "forbidden_features": ["heating", "cooling", "vav"],  # control responses would mask faults

    # training
    "separate_regime_models": True,  # False: one model with 'occupied' as a feature
    "train_end": None,               # e.g. "2025-03-01" (local time); overrides train_frac
    "train_frac": 0.7,               # share of the overlap period (by time) used for training
    "min_train_samples": 500,
    "quantiles": [0.01, 0.05, 0.5, 0.95, 0.99],
    "band": (0.01, 0.99),            # flagging band
    "inner_band": (0.05, 0.95),      # plotted only
    "hgb_params": {
        "max_depth": 4,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 100,
        "learning_rate": 0.05,
        "max_iter": 500,
        "l2_regularization": 1.0,
        "early_stopping": True,
        "validation_fraction": 0.2,
        "n_iter_no_change": 20,
        "random_state": 0,
    },
    "temporal_validation": True,     # early stopping on the last part of the training period
    "robust_refit": True,

    # validation: warn if |observed - nominal| > min(rel * min(q, 1 - q), max)
    "coverage_tol_rel": 0.5,
    "coverage_tol_max": 0.05,

    # flagging
    "min_exceedance": 0.5,           # °C beyond the band
    "min_band_width": 0.1,           # °C, floor for the score denominator

    # output
    "pd_features": ["T_out_3h", "solar_2h"],
})

REGIME_COLORS = {"all": "#1f3b73", "occupied": "#1f3b73", "unoccupied": "#808080"}


# Helpers
def _merge_config(config):
    """Return a copy of CONFIG updated with the user overrides in `config`."""
    return cd._merge_config(config, base=CONFIG)


def _qcol(q):
    """Column name of a predicted quantile, e.g. 0.05 -> 'q5'."""
    return f"q{100 * q:g}"


def _quantiles(cfg):
    qs = sorted(cfg["quantiles"])
    for q in list(cfg["band"]) + list(cfg["inner_band"]) + [0.5]:
        if q not in qs:
            raise ValueError(f"quantile {q} (band/median) is missing from CONFIG['quantiles']")
    return qs


def model_features(config=None):
    """Feature list used by the models; adds 'occupied' for a pooled model."""
    cfg = _merge_config(config)
    feats = list(cfg["features"])
    bad = sorted(set(feats) & set(cfg["forbidden_features"]))
    if bad:
        raise ValueError(f"{bad} are control responses and must not be model features")
    if not cfg["separate_regime_models"] and "occupied" not in feats:
        feats.append("occupied")
    return feats


def _X(data, feats):
    return data[feats].astype(float)


def _model_groups(data, cfg):
    """Return {group: row mask}; one group per regime or a single 'all' group."""
    if cfg["separate_regime_models"]:
        return {reg: data["regime"] == reg for reg in REGIMES}
    return {"all": pd.Series(True, index=data.index)}


def _scored(data):
    """Points that can be judged: valid dev, a prediction, not in recovery."""
    return data["dev"].notna() & data[_qcol(0.5)].notna() & ~data["recovery"]


# Step 0: inspection and combination (shared with conditional_dist)
def inspect_inputs(dfs, config=None):
    """Per-key type/index/range/interval/NaN/stat report; see conditional_dist.inspect_inputs."""
    return cd.inspect_inputs(dfs, _merge_config(config))


def combine_data(dfs, config=None):
    """Resample to `resample_freq`, rename and restrict to the core overlap; see conditional_dist."""
    return cd.combine_data(dfs, _merge_config(config))


# Step 1: preprocessing (shared with conditional_dist)
def preprocess(data, config=None):
    """
    Gap filling (<= max_interp_gap), dev = T_in - setpoint, occupied/regime
    from the schedule, CO2 agreement report and setpoint-change recovery
    flags; see conditional_dist.preprocess.
    """
    return cd.preprocess(data, _merge_config(config))


# Step 2: features
def build_features(data, config=None):
    """
    Add the trailing rolling means in CONFIG['rolling_features'] and
    hour_sin/hour_cos (local time) to a copy of `data`.
    """
    cfg = _merge_config(config)
    data = data.copy()
    freq = cfg["resample_freq"]
    for name, (src, win) in cfg["rolling_features"].items():
        n_win = pd.Timedelta(win) / pd.Timedelta(freq)
        min_p = max(1, int(np.ceil(cfg["rolling_min_frac"] * n_win)))
        data[name] = data[src].rolling(win, min_periods=min_p).mean()

    hour = data.index.hour + data.index.minute / 60
    data["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    data["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    feats = model_features(cfg)
    missing = [f for f in feats if f not in data]
    if missing:
        raise ValueError(f"features not in data: {missing}")
    print("\nModel features (share NaN):")
    for f in feats:
        print(f"  {f:10s}: {data[f].isna().mean():.1%}")
    return data


# Step 3: training
def assign_split(data, config=None):
    """Add split = 'train' | 'test' by time (train_end, or the first train_frac of the period)."""
    cfg = _merge_config(config)
    data = data.copy()
    idx = data.index
    if cfg["train_end"] is not None:
        t_split = _localize(cfg["train_end"], idx.tz)
    else:
        t_split = idx[0] + cfg["train_frac"] * (idx[-1] - idx[0])
    data["split"] = np.where(idx < t_split, "train", "test")
    usable = data["dev"].notna() & ~data["recovery"]
    print(f"\nTrain: {idx[0]} -> {t_split}")
    print(f"Test : {t_split} -> {idx[-1]}")
    print("Usable samples (valid dev, not recovery) per split and regime:")
    print(pd.crosstab(data.loc[usable, "split"], data.loc[usable, "regime"], margins=True).to_string())
    return data


def _fit_quantile_set(X, y, cfg):
    """Fit one HistGradientBoostingRegressor per quantile; returns {q: model}."""
    params = dict(cfg["hgb_params"])
    fit_kw = {}
    supports_val = "X_val" in inspect.signature(HistGradientBoostingRegressor.fit).parameters
    if cfg["temporal_validation"] and params.get("early_stopping"):
        if supports_val:
            # hold out the last part of the (time-ordered) training data instead of a random split
            n_val = max(1, int(len(X) * params.get("validation_fraction", 0.1)))
            fit_kw = {"X_val": X.iloc[-n_val:], "y_val": y.iloc[-n_val:]}
            X, y = X.iloc[:-n_val], y.iloc[:-n_val]
        else:
            print("  note: scikit-learn < 1.6 has no X_val; early stopping uses a random split")
    return {q: HistGradientBoostingRegressor(loss="quantile", quantile=q, **params)
            .fit(X, y, **fit_kw) for q in _quantiles(cfg)}


def _predict_quantiles(qmodels, X, cfg):
    """Predicted quantiles as an (n, n_q) array, sorted row-wise (non-crossing)."""
    P = np.column_stack([qmodels[q].predict(X) for q in _quantiles(cfg)])
    return np.sort(P, axis=1)


def train_quantile_models(data, config=None):
    """
    Fit quantile models per group (regime, or 'all') on train-split points
    with valid dev that are not in recovery. With robust_refit, points
    outside the predicted band are removed once and the models refitted.

    Returns {(group, quantile): fitted model}.
    """
    cfg = _merge_config(config)
    feats = model_features(cfg)
    qs = _quantiles(cfg)
    i_lo, i_hi = qs.index(cfg["band"][0]), qs.index(cfg["band"][1])
    train = (data["split"] == "train") & data["dev"].notna() & ~data["recovery"]

    models = {}
    print(f"\nTraining quantile models {[_qcol(q) for q in qs]} on {feats}")
    for grp, mask in _model_groups(data, cfg).items():
        d = data.loc[mask & train]
        X, y = _X(d, feats), d["dev"]
        if len(d) < cfg["min_train_samples"]:
            print(f"  {grp:10s}: only {len(d)} training samples (< {cfg['min_train_samples']}); "
                  f"no model, points get status 'no_model'")
            continue
        qmodels = _fit_quantile_set(X, y, cfg)
        n_removed = 0
        if cfg["robust_refit"]:
            P = _predict_quantiles(qmodels, X, cfg)
            keep = (y.to_numpy() >= P[:, i_lo]) & (y.to_numpy() <= P[:, i_hi])
            n_removed = int((~keep).sum())
            qmodels = _fit_quantile_set(X[keep], y[keep], cfg)
        iters = ", ".join(f"{_qcol(q)}={m.n_iter_}" for q, m in qmodels.items())
        print(f"  {grp:10s}: n_train={len(d)}, removed by robust refit={n_removed} "
              f"({n_removed / len(d):.1%}); iterations: {iters}")
        for q, m in qmodels.items():
            models[(grp, q)] = m
    return models


def predict_bands(data, models, config=None):
    """Add the predicted (non-crossing) quantile columns q1, q5, q50, ... to a copy of `data`."""
    cfg = _merge_config(config)
    data = data.copy()
    feats = model_features(cfg)
    qs = _quantiles(cfg)
    cols = [_qcol(q) for q in qs]
    for c in cols:
        data[c] = np.nan
    for grp, mask in _model_groups(data, cfg).items():
        if (grp, qs[0]) not in models or not mask.any():
            continue
        qmodels = {q: models[(grp, q)] for q in qs}
        data.loc[mask, cols] = _predict_quantiles(qmodels, _X(data.loc[mask], feats), cfg)
    return data


# Step 4: validation
def validate(data, config=None):
    """
    Coverage (share of dev below each predicted quantile) and pinball loss
    per split, regime ('all' + each regime) and quantile, on scored points.
    Prints the tables and warns when test coverage is far from nominal.

    Returns a DataFrame indexed by (split, regime, quantile).
    """
    cfg = _merge_config(config)
    qs = _quantiles(cfg)
    sc = _scored(data)
    rows = []
    for split in ["train", "test"]:
        for reg in ["all"] + REGIMES:
            m = sc & (data["split"] == split)
            if reg != "all":
                m &= data["regime"] == reg
            if not m.any():
                continue
            y = data.loc[m, "dev"]
            for q in qs:
                pred = data.loc[m, _qcol(q)]
                obs = (y < pred).mean()
                rows.append({"split": split, "regime": reg, "quantile": q, "n": int(m.sum()),
                             "observed": obs, "error": obs - q,
                             "pinball": mean_pinball_loss(y, pred, alpha=q)})
    val = pd.DataFrame(rows).set_index(["split", "regime", "quantile"])

    lo, hi = cfg["band"]
    print("\nCoverage: share of observations below each predicted quantile [%]")
    print((val["observed"].unstack("quantile") * 100).round(2).to_string())
    print("\nPinball loss [°C]")
    print(val["pinball"].unstack("quantile").round(4).to_string())
    print(f"\nShare inside the {lo:g}–{hi:g} band [%] (nominal {100 * (hi - lo):g}):")
    inside = (val.xs(hi, level="quantile")["observed"] - val.xs(lo, level="quantile")["observed"])
    print((inside * 100).round(2).to_string())

    test = val.xs("test", level="split") if "test" in val.index.get_level_values("split") else None
    if test is None:
        print("WARNING: no test data; coverage cannot be checked out of sample")
        return val
    n_warn = 0
    for (reg, q), r in test.iterrows():
        tol = min(cfg["coverage_tol_rel"] * min(q, 1 - q), cfg["coverage_tol_max"])
        if abs(r["error"]) > tol:
            n_warn += 1
            print(f"WARNING: test coverage {reg}/{_qcol(q)} = {r['observed']:.2%} vs nominal "
                  f"{q:.0%} (tolerance ±{tol:.1%}); the band is unreliable here")
    if n_warn == 0:
        print("Test coverage within tolerance for all quantiles.")
    return val


# Step 5: flagging and events
def flag_anomalies(data, config=None):
    """
    Add exceedance (°C beyond the band), score (exceedance / band width),
    outside_band, anomaly, direction and status columns to a copy of `data`.

    status: 'missing' | 'no_model' | 'recovery' | 'normal' | 'anomaly'
    """
    cfg = _merge_config(config)
    data = data.copy()
    lo, hi = (data[_qcol(q)] for q in cfg["band"])
    above = (data["dev"] - hi).clip(lower=0)
    below = (lo - data["dev"]).clip(lower=0)
    width = (hi - lo).clip(lower=cfg["min_band_width"])
    data["exceedance"] = above + below
    data["score"] = data["exceedance"] / width
    data["outside_band"] = data["exceedance"] > 0
    data["anomaly"] = _scored(data) & (data["exceedance"] > cfg["min_exceedance"])
    data["direction"] = np.where(data["anomaly"], np.where(above > 0, 1, -1), 0).astype(int)

    status = pd.Series("normal", index=data.index, dtype=object)
    has_dev = data["dev"].notna()
    status[has_dev & data[_qcol(0.5)].isna()] = "no_model"
    status[has_dev & data["recovery"]] = "recovery"
    status[~has_dev] = "missing"
    status[data["anomaly"]] = "anomaly"
    data["status"] = status
    return data


def detect_events(data, config=None):
    """
    Events from the band flags (see conditional_dist.collect_events).
    rank_score = max exceedance [°C] × duration [h]. Returns (data, events).
    """
    cfg = _merge_config(config)
    extra = {"max_exceedance": lambda s: s["exceedance"].max(),
             "max_score": lambda s: s["score"].max(),
             "q50_mean": lambda s: s[_qcol(0.5)].mean()}
    return cd.collect_events(data, cfg, extra, "rank_score",
                             lambda ev: ev["max_exceedance"] * ev["duration"].dt.total_seconds() / 3600)


# Step 6: plots
def plot_period(data, events, start, end, config=None):
    """
    Five stacked panels sharing x for [start, end]:
      (1) dev with predicted median, inner and outer bands, anomalies and events,
      (2) T_in & setpoint, (3) T_out & solar (twin axis), (4) heating, cooling, vav, (5) co2.
    `start`/`end` may be strings; they are interpreted in the data's time zone.
    """
    cfg = _merge_config(config)
    tz = data.index.tz
    start, end = _localize(start, tz), _localize(end, tz)
    d = data.loc[start:end]
    step = d.index.to_series().diff().median() if len(d) > 1 else pd.Timedelta(cfg["resample_freq"])
    fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1.3, 1.3, 1.3, 1]})
    (b_lo, b_hi), (i_lo, i_hi) = cfg["band"], cfg["inner_band"]

    ax = axes[0]
    ax.fill_between(d.index, d[_qcol(b_lo)], d[_qcol(b_hi)], color="#1f3b73", alpha=0.10, lw=0,
                    label=f"{100 * b_lo:g}–{100 * b_hi:g} %")
    ax.fill_between(d.index, d[_qcol(i_lo)], d[_qcol(i_hi)], color="#1f3b73", alpha=0.22, lw=0,
                    label=f"{100 * i_lo:g}–{100 * i_hi:g} %")
    ax.plot(d.index, d[_qcol(0.5)], color="#1f3b73", lw=1.2, ls="--", label="predicted median")
    ax.plot(d.index, d["dev"], color="#222222", lw=1.2, label="dev")
    cd._shade_recovery(ax, d)
    for sign, color in [(1, "#d7301f"), (-1, "#3182bd")]:
        a = d[d["direction"] == sign]
        if len(a):
            ax.scatter(a.index, a["dev"], s=14, color=color, zorder=3, label=DIRECTION[sign])
    cd._shade_events(ax, events, start, end, step, alpha=0.15, label=False)
    ax.axhline(0, color="#808080", lw=0.8)
    test = d.index[d["split"] == "test"]
    if len(test) and test[0] > d.index[0]:
        for a in axes:
            a.axvline(test[0], color="#555555", lw=1, ls=":")
        ax.text(test[0], 1, " test →", transform=ax.get_xaxis_transform(), va="top", fontsize=8)
    ax.set_ylabel("dev = T_in − setpoint [°C]")
    ax.set_title(f"{cfg['room']}: {start:%Y-%m-%d %H:%M} – {end:%Y-%m-%d %H:%M}")
    ax.legend(loc="upper left", ncol=7, fontsize=8, frameon=False)

    cd._plot_T_in(axes[1], d)
    axes[1].legend(loc="upper left", ncol=2, fontsize=8, frameon=False)
    cd._plot_context_panels(axes[2:], d)
    cd._finish_time_axes(fig, axes, tz)
    return fig


def plot_calibration(validation, config=None):
    """Nominal vs observed coverage (left) and coverage error with tolerance (right), train vs test."""
    cfg = _merge_config(config)
    qs = np.array(_quantiles(cfg))
    x = np.arange(len(qs))  # evenly spaced, the tail quantiles would overlap on a linear axis
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(x, qs, color="#b0b0b0", lw=1, ls="--", label="nominal")
    tol = np.minimum(cfg["coverage_tol_rel"] * np.minimum(qs, 1 - qs), cfg["coverage_tol_max"])
    axes[1].fill_between(x, -tol * 100, tol * 100, color="#e5e5e5", lw=0, label="tolerance")
    axes[1].axhline(0, color="#b0b0b0", lw=1)

    splits = validation.index.get_level_values("split").unique()
    for split, color in [("train", "#808080"), ("test", "#1f3b73")]:
        if split not in splits:
            continue
        v = validation.xs(split, level="split")
        for reg, marker, ls in [("all", "o", "-"), ("occupied", "s", "none"),
                                ("unoccupied", "^", "none")]:
            if reg not in v.index.get_level_values("regime"):
                continue
            r = v.xs(reg, level="regime").reindex(qs)
            kw = dict(color=color, marker=marker, ls=ls, ms=6 if reg == "all" else 5,
                      mfc=color if reg == "all" else "none", label=f"{split} – {reg}")
            axes[0].plot(x, r["observed"], **kw)
            axes[1].plot(x, r["error"] * 100, **kw)

    axes[0].set_xlabel("nominal quantile")
    axes[0].set_ylabel("observed share below predicted quantile")
    axes[0].set_title("Calibration")
    axes[1].set_xlabel("nominal quantile")
    axes[1].set_ylabel("observed − nominal [percentage points]")
    axes[1].set_title("Coverage error")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{q:g}" for q in qs])
        ax.grid(True, color="#e5e5e5", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(qs)
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_partial_dependence(data, models, config=None):
    """
    Partial dependence of the median model on CONFIG['pd_features'] (one line
    per model group, computed on its training data), and print the change
    in predicted dev over the 2–98 % range of each feature.
    """
    cfg = _merge_config(config)
    feats = model_features(cfg)
    pd_feats = cfg["pd_features"]
    fig, axes = plt.subplots(1, len(pd_feats), figsize=(6 * len(pd_feats), 4.5), squeeze=False)
    axes = axes[0]
    train = (data["split"] == "train") & data["dev"].notna() & ~data["recovery"]
    print("\nPartial dependence of the median model (change over the 2–98 % feature range):")
    for grp, mask in _model_groups(data, cfg).items():
        if (grp, 0.5) not in models:
            continue
        X = _X(data.loc[mask & train], feats)
        for ax, f in zip(axes, pd_feats):
            Xf = X.dropna(subset=[f])
            r = partial_dependence(models[(grp, 0.5)], Xf, [f], grid_resolution=50,
                                   percentiles=(0.02, 0.98))
            grid, avg = r["grid_values"][0], r["average"][0]
            ax.plot(grid, avg, color=REGIME_COLORS[grp], lw=1.8, label=grp)
            print(f"  {grp:10s} {f:10s}: {avg[-1] - avg[0]:+.2f} °C "
                  f"({grid[0]:.1f} -> {grid[-1]:.1f})")
            if f.startswith("solar") and avg[-1] < avg[0]:
                print(f"    note: dev decreases with {f}; physically unexpected, check the model")
    for ax, f in zip(axes, pd_feats):
        ax.set_xlabel(f)
        ax.grid(True, color="#e5e5e5", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("partial dependence of median dev [°C]")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Median model: partial dependence (training data)")
    fig.tight_layout()
    return fig


def plot_score_hist(data, config=None):
    """Histogram of the anomaly score for scored points outside the band (flagged vs below the °C floor)."""
    cfg = _merge_config(config)
    out = data[_scored(data) & data["outside_band"]]
    if out.empty:
        print("No points outside the band; score histogram skipped.")
        return None
    bins = np.linspace(0, max(np.nanpercentile(out["score"], 99.5), 1e-3), 40)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist([out.loc[out["anomaly"], "score"].clip(upper=bins[-1]),
             out.loc[~out["anomaly"], "score"].clip(upper=bins[-1])], bins=bins, stacked=True,
            color=["#d7301f", "#b0b0b0"],
            label=[f"flagged (exceedance > {cfg['min_exceedance']} °C)", "below °C floor"])
    ax.set_yscale("log")
    ax.set_xlabel("anomaly score = exceedance / band width (last bin includes the tail)")
    ax.set_ylabel("samples")
    ax.set_title(f"Anomaly score outside the band (n={len(out)})")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# Step 7: summary and comparison with the binning approach
def compare_events(events, events_binning, config=None):
    """
    Match quantile-regression events with binning events (same direction,
    overlapping in time). Returns a table with source 'both' | 'qr only' |
    'binning only' and the event ids from each table.
    """
    cfg = _merge_config(config)
    freq = pd.Timedelta(cfg["resample_freq"])

    def _end(e):
        return e["start"] + e["duration"] if "duration" in e else e["end"] + freq

    rows, matched_b = [], set()
    for qid, e in events.iterrows():
        hits = [bid for bid, b in events_binning.iterrows()
                if b["direction"] == e["direction"] and b["start"] < _end(e) and _end(b) > e["start"]]
        matched_b.update(hits)
        rows.append({"source": "both" if hits else "qr only", "qr_event_id": qid,
                     "binning_event_ids": hits, "start": e["start"], "end": e["end"],
                     "direction": e["direction"]})
    for bid, b in events_binning.iterrows():
        if bid not in matched_b:
            rows.append({"source": "binning only", "qr_event_id": None, "binning_event_ids": [bid],
                         "start": b["start"], "end": b["end"], "direction": b["direction"]})
    comp = pd.DataFrame(rows)
    if comp.empty:
        print("\nNo events in either table.")
        return comp
    print("\nComparison with binning events (same direction, overlapping in time):")
    print(pd.crosstab(comp["source"], comp["direction"], margins=True).to_string())
    return comp


def print_summary(data, events, config=None):
    """Print regime shares, flag rates (train vs test), event counts per direction/hint and top events."""
    cfg = _merge_config(config)
    cd._print_regime_status(data)
    print("\nFlagged share of scored samples [%]:")
    flagged = data[_scored(data)].groupby(["split", "regime"])["anomaly"].agg(["sum", "size", "mean"])
    flagged.columns = ["flagged", "scored", "share_%"]
    flagged["share_%"] = (flagged["share_%"] * 100).round(2)
    print(flagged.to_string())

    show = ["start", "duration", "regime", "direction", "max_abs_dev", "max_exceedance",
            "q50_mean", "T_out_3h_mean", "solar_2h_mean", "heating_mean", "cooling_mean",
            "vav_mean", "co2_mean", "hint"]
    cd._print_events(events, cfg, show, "rank_score", "max exceedance × duration [°C·h]")


# main workflow
def run_analysis(dfs, config=None, strict=True, events_binning=None):
    """
    Run the full pipeline on `dfs` (not modified).

    Returns (data, models, validation, events):
      data       - 15-min combined DataFrame (local time) with features, predicted
                   quantiles (q1, q5, q50, q95, q99), score, anomaly and status
      models     - {(group, quantile): HistGradientBoostingRegressor}
      validation - coverage and pinball loss per (split, regime, quantile)
      events     - anomaly events with context and hints
    events_binning: events table from conditional_dist; if None, a variable of
    that name in the calling notebook/script (__main__) is used when it exists.
    With strict=True the run stops after inspection if any ERROR is found.
    """
    cfg = _merge_config(config)
    issues = inspect_inputs(dfs, cfg)
    errors = [m for lvl, m in issues if lvl == "ERROR"]
    if errors and strict:
        raise ValueError("Input inspection found errors; fix them or pass strict=False:\n  "
                         + "\n  ".join(errors))

    data, _ = combine_data(dfs, cfg)
    data = preprocess(data, cfg)
    data = build_features(data, cfg)
    data = assign_split(data, cfg)
    models = train_quantile_models(data, cfg)
    data = predict_bands(data, models, cfg)
    validation = validate(data, cfg)
    data = flag_anomalies(data, cfg)
    data, events = detect_events(data, cfg)
    print_summary(data, events, cfg)

    if events_binning is None:
        import __main__
        events_binning = getattr(__main__, "events_binning", None)
    if events_binning is not None and not events.empty:
        comp = compare_events(events, events_binning, cfg)
        both = comp.loc[comp["source"] == "both", "qr_event_id"]
        events["found_by_binning"] = events.index.isin(both)

    if cfg["make_plots"]:
        plot_calibration(validation, cfg)
        plot_partial_dependence(data, models, cfg)
        plot_score_hist(data, cfg)
        if not events.empty:
            e = events.sort_values("rank_score", ascending=False).iloc[0]
            plot_period(data, events, e["start"] - pd.Timedelta("2D"), e["end"] + pd.Timedelta("2D"), cfg)
        plt.show()
    return data, models, validation, events


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description="Run the quantile-regression anomaly detection on CSV files.")
    # parser.add_argument("files", nargs="+", help="CSV files to process")
    # parser.add_argument("--strict", action="store_true", help="Stop on input errors")
    # args = parser.parse_args()

    # dfs = {f: pd.read_csv(f, parse_dates=["time"], index_col="time") for f in args.files}
    # run_analysis(dfs, strict=args.strict)
    import pickle

    with open("./07-D2-02_data/dfs.pkl", "rb") as f:
        dfs = pickle.load(f)

    run_analysis(dfs, strict=True)
