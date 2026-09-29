import graphlib
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error

import conditional_dist as cd


_TIME = ["hour_sin", "hour_cos", "workday"]   # time-of-day and day-type features

# tunable parameters (schedule, holidays and time zone come from conditional_dist.CONFIG)
CONFIG = {
    "key_map": {
        "indoor_temperature": "T_in",
        "setpoint": "setpoint",
        "outdoor_temperature": "T_out",
        "solar_radiation": "solar",
        "co2": "co2",
        "heating": "heating",
        "cooling": "cooling",
        "VAV": "vav",
        "VS02_supply": "vs_sup",
        "VS02_return": "vs_ret",
        "KB02_supply": "kb_sup",
        "KB02_return": "kb_ret",
    },
    "context_cols": ["co2", "heating", "cooling", "vav", "vs_sup", "vs_ret", "kb_sup", "kb_ret"],

    # features
    "T_out_short": "3h",
    "T_out_long": "24h",
    "solar_halflife": "4h",
    "lag": "30min",              # one sample; breaks the T_in <-> controller loop

    # causal graph: node -> parent feature columns (edit here)
    "root_nodes": ["T_out", "solar", "co2"],
    "graph": {
        "vs_sup": ["T_out_3h"] + _TIME,
        "kb_sup": ["T_out_3h"] + _TIME,
        "heating": ["T_in_lag", "setpoint", "T_out_3h"] + _TIME,
        "cooling": ["T_in_lag", "setpoint", "T_out_3h"] + _TIME,
        "vav": ["T_in_lag", "setpoint", "co2_lag", "T_out_3h"] + _TIME,
        "vs_ret": ["vs_sup", "heating", "T_out_3h"],
        "kb_ret": ["kb_sup", "cooling", "T_out_3h"],
        "T_in": ["T_out_3h", "T_out_24h", "solar_acc", "co2", "vav_lag",
                 "heat_del_lag", "cool_del_lag"] + _TIME,
    },
    "return_nodes": ["vs_ret", "kb_ret"],
    "include_returns": False,    # circuits likely serve many rooms
    # which graph nodes a derived feature comes from (anything else maps to itself)
    "feature_sources": {
        "T_out_3h": ["T_out"], "T_out_24h": ["T_out"], "solar_acc": ["solar"],
        "co2_lag": ["co2"], "vav_lag": ["vav"], "T_in_lag": ["T_in"],
        "heat_del_lag": ["heating", "vs_sup", "T_in"],
        "cool_del_lag": ["cooling", "kb_sup", "T_in"],
    },
    "rs_drop_features": ["T_in_lag"],   # no T_in -> controller edges in the summary graph

    # mechanism models
    "hgb_params": {"max_iter": 200, "learning_rate": 0.1, "random_state": 0},
    "n_jobs": -1,
    "root_window": "24h",        # rolling median baseline for T_out and solar

    # detection
    "score_threshold": 0.995,    # A(t) = 1 - min_i p_i
    "min_consecutive": 2,        # samples
    "event_max_gap_samples": 1,
    "T_in_dev_min": 1.0,         # °C, mean |T_in - setpoint| for indoor-temperature events
    "alpha": 0.5,                # Eq. (1)
    "top_k": 3,

    # known cases: (label, start, end), local time, end exclusive
    "cases": [
        ("2024-04-01", "2024-04-01 00:00", "2024-04-02 00:00"),
        ("2024-07-22 office hours", "2024-07-22 08:00", "2024-07-22 18:00"),
        ("2024-08-29 evening", "2024-08-29 17:00", "2024-08-30 00:00"),
        ("2024-09-06", "2024-09-06 00:00", "2024-09-07 00:00"),
    ],
    "night_window": ("00:00", "02:00"),
    "night_heating_min": 15.0,   # % heating valve = "spike"

    # output
    "heatmap_period": ("2024-09-02", "2024-09-09"),
    "heatmap_vmin": 0.9,         # S below this is drawn white
    "make_plots": True,
}


# Helpers
def _merge_config(config=None):
    """CONFIG on top of conditional_dist.CONFIG, then the user overrides in 'config'."""
    return cd._merge_config(config, base=cd._merge_config(CONFIG, base=cd.CONFIG))


def _lag(s, lag):
    """Value of s at t - lag (time-based shift)."""
    return s.shift(freq=lag).reindex(s.index)


def _rolling(s, window, cfg, how="mean"):
    """Trailing time-based rolling mean/median with rolling_min_frac valid samples."""
    n_win = pd.Timedelta(window) / pd.Timedelta(cfg["resample_freq"])
    min_p = max(1, int(np.ceil(cfg["rolling_min_frac"] * n_win)))
    return getattr(s.rolling(window, min_periods=min_p), how)()


def _window(obj, start, end):
    """Rows of obj with start <= t < end (strings are local time)."""
    tz = obj.index.tz
    start, end = cd._localize(start, tz), cd._localize(end, tz)
    return obj[(obj.index >= start) & (obj.index < end)]


def model_nodes(cfg):
    """Non-root nodes that get a mechanism model."""
    return [n for n in cfg["graph"]
            if cfg["include_returns"] or n not in cfg["return_nodes"]]


# Step 1: features and causal graph
def build_features(data, cfg):
    """
    Add day type, hour (sin/cos), T_out means, accumulated solar, delivered
    heating/cooling and the lagged parents to a copy of 'data'.
    """
    d = cd.add_schedule(data.copy(), cfg)
    d["workday"] = (d.index.weekday.isin(cfg["occupied_weekdays"]) & ~d["day_off"]).astype(int)
    hour = d.index.hour + d.index.minute / 60
    d["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    d["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    d["T_out_3h"] = _rolling(d["T_out"], cfg["T_out_short"], cfg)
    d["T_out_24h"] = _rolling(d["T_out"], cfg["T_out_long"], cfg)
    d["solar_acc"] = d["solar"].ewm(halflife=cfg["solar_halflife"], times=d.index).mean()

    # approximate delivered heating/cooling: valve position x water-room temperature difference
    d["heat_del"] = d["heating"] * (d["vs_sup"] - d["T_in"]).clip(lower=0)
    d["cool_del"] = d["cooling"] * (d["T_in"] - d["kb_sup"]).clip(lower=0)
    for c in ["T_in", "co2", "vav", "heat_del", "cool_del"]:
        d[f"{c}_lag"] = _lag(d[c], cfg["lag"])
    return d


def summary_graph(cfg):
    """
    Node -> parent nodes of the summary graph used for root cause scores, and
    the topological order (roots first). Lagged T_in -> controller edges and
    non-scored inputs (setpoint, hour, day type) are left out.
    """
    nodes = cfg["root_nodes"] + model_nodes(cfg)
    parents = {n: [] for n in cfg["root_nodes"]}
    for node in model_nodes(cfg):
        src = [s for f in cfg["graph"][node] if f not in cfg["rs_drop_features"]
               for s in cfg["feature_sources"].get(f, [f])]
        parents[node] = sorted({s for s in src if s in nodes and s != node})
    order = list(graphlib.TopologicalSorter(parents).static_order())
    return parents, order


# Step 2: local mechanism models
def fit_mechanism(d, node, parents, cfg):
    """
    Out-of-fold predictions of 'node' from 'parents' with blocked CV: each
    calendar week is predicted by a model trained on all other weeks.
    Rows with a missing parent (or target) are dropped for this node only.
    """
    rows = d[parents + [node]].dropna()
    iso = rows.index.isocalendar()
    week = iso["year"].to_numpy() * 100 + iso["week"].to_numpy()
    model = HistGradientBoostingRegressor(**cfg["hgb_params"])
    pred = cross_val_predict(model, rows[parents], rows[node], groups=week,
                             cv=LeaveOneGroupOut(), n_jobs=cfg["n_jobs"])
    return pd.Series(pred, index=rows.index).reindex(d.index)


def root_baselines(d, cfg):
    """Expected values of the root nodes: co2 by hour and day type, T_out/solar by 24 h rolling median."""
    base = pd.DataFrame(index=d.index)
    base["co2"] = d["co2"].groupby([d.index.hour, d["workday"]]).transform("median")
    for c in ["T_out", "solar"]:
        base[c] = _rolling(d[c], cfg["root_window"], cfg, how="median")
    return base[cfg["root_nodes"]]


def empirical_pvalues(resid):
    """p = share of all |r| of the node that are >= this |r| (NaN stays NaN)."""
    a = resid.abs()
    ref = np.sort(a.dropna().to_numpy())
    p = (len(ref) - np.searchsorted(ref, a.to_numpy(), side="left")) / len(ref)
    return pd.Series(p, index=resid.index).where(a.notna())


def score_nodes(d, cfg):
    """Return pred, resid and p-value DataFrames (one column per scored node)."""
    pred = root_baselines(d, cfg)
    for node in model_nodes(cfg):
        print(f"  fitting {node:8s} <- {', '.join(cfg['graph'][node])}")
        pred[node] = fit_mechanism(d, node, cfg["graph"][node], cfg)
    resid = d[pred.columns] - pred
    pval = resid.apply(empirical_pvalues)
    return pred, resid, pval


def mechanism_metrics(d, pred):
    """Cross-validated R² and MAE per node (out-of-fold for models, baseline for roots)."""
    rows = []
    for node in pred.columns:
        ok = d[node].notna() & pred[node].notna()
        y, yhat = d.loc[ok, node], pred.loc[ok, node]
        rows.append({"node": node, "n": int(ok.sum()), "R2": r2_score(y, yhat),
                     "MAE": mean_absolute_error(y, yhat)})
    return pd.DataFrame(rows).set_index("node")


# Step 3: anomaly score and events
def flag_samples(A, cfg):
    """A(t) > score_threshold for at least min_consecutive samples in a row."""
    mask = A > cfg["score_threshold"]
    _, run_len = cd._run_lengths(mask)
    return mask & (run_len >= cfg["min_consecutive"])


def root_cause_scores(S, parents, order, alpha):
    """Eq. (1): RS_i = S_i + alpha * mean(RS of children of i), children first."""
    children = {n: [c for c in order if n in parents[c]] for n in order}
    RS = pd.DataFrame(index=S.index, columns=order, dtype=float)
    for node in reversed(order):
        RS[node] = S[node]
        if children[node]:
            RS[node] += alpha * RS[children[node]].mean(axis=1).fillna(0)
    return RS


def build_events(d, A, flagged, RS, cfg):
    """
    Group flagged samples into events (gaps up to event_max_gap_samples) and
    rank root causes by mean RS over each event.
    """
    freq = pd.Timedelta(cfg["resample_freq"])
    k = cfg["top_k"]
    pos = np.flatnonzero(flagged.to_numpy())
    grp = np.cumsum(np.diff(pos, prepend=-np.inf) > cfg["event_max_gap_samples"] + 1)
    rows = []
    for g in np.unique(grp):
        start, end = d.index[pos[grp == g][0]], d.index[pos[grp == g][-1]]
        seg = d.loc[start:end]
        rs = RS.loc[start:end].mean().nlargest(k)
        row = {"start": start, "end": end, "duration": end - start + freq,
               "n_flagged": int((grp == g).sum()), "max_A": A.loc[start:end].max()}
        for i, (node, score) in enumerate(rs.items(), 1):
            row[f"cause_{i}"], row[f"rs_{i}"] = node, score
        dev = seg["T_in"] - seg["setpoint"]
        row.update({"dev_mean": dev.mean(), "abs_dev_mean": dev.abs().mean(),
                    "T_out_mean": seg["T_out"].mean(), "solar_mean": seg["solar"].mean(),
                    "cooling_mean": seg["cooling"].mean(), "kb_sup_mean": seg["kb_sup"].mean()})
        rows.append(row)
    events = pd.DataFrame(rows)
    events.index.name = "event_id"
    return events


def indoor_events(events, cfg):
    """Events with mean |T_in - setpoint| > T_in_dev_min or T_in among the top-k causes."""
    if events.empty:
        return events
    causes = events[[f"cause_{i}" for i in range(1, cfg["top_k"] + 1)]]
    return events[(events["abs_dev_mean"] > cfg["T_in_dev_min"]) | (causes == "T_in").any(axis=1)]


# Step 4: plots
def plot_score_heatmap(res, start, end, config=None):
    """Node scores S_i (rows) over time for [start, end), with A(t) and flagged samples on top."""
    cfg = _merge_config(config)
    S = _window(res["S"], start, end)[res["order"]]
    A = _window(res["A"], start, end)
    fl = _window(res["flagged"], start, end)
    tz = S.index.tz
    fig, axes = plt.subplots(2, 1, figsize=(13, 3.2 + 0.4 * S.shape[1]), sharex=True, layout="constrained",
                             gridspec_kw={"height_ratios": [1, 0.4 * S.shape[1]]})

    ax = axes[0]
    ax.plot(A.index, A, color="#1f3b73", lw=1.2, label="A(t)")
    ax.scatter(A.index[fl], A[fl], s=12, c="#d7301f", zorder=3, label=f"flagged (n={int(fl.sum())})")
    ax.axhline(cfg["score_threshold"], color="#808080", lw=0.8, ls="--")
    ax.set_ylim(max(0, cfg["heatmap_vmin"] - 0.05), 1.002)
    ax.set_ylabel("A(t)")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1), ncol=2, fontsize=8, frameon=False)
    ax.set_title(f"Node scores S = 1 − p: {S.index[0]:%Y-%m-%d} – {S.index[-1]:%Y-%m-%d}", loc="left")

    ax = axes[1]
    step = pd.Timedelta(cfg["resample_freq"])
    edges = S.index.append(pd.DatetimeIndex([S.index[-1] + step]))
    mesh = ax.pcolormesh(edges, np.arange(S.shape[1] + 1), S.T.to_numpy(), cmap="Reds",
                         vmin=cfg["heatmap_vmin"], vmax=1, shading="flat")
    ax.set_yticks(np.arange(S.shape[1]) + 0.5)
    ax.set_yticklabels(S.columns)
    ax.invert_yaxis()
    ax.set_facecolor("#d9d9d9")
    fig.colorbar(mesh, ax=axes[1], label=f"S (white < {cfg['heatmap_vmin']:g}, grey = no data)",
                 shrink=0.9)

    axes[0].grid(True, color="#e5e5e5", lw=0.6)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=tz))
    fig.autofmt_xdate()
    return fig


def plot_event(res, start, end, config=None):
    """
    Stacked panels for [start, end): A(t), then actual vs predicted for each
    node (roots: baseline). Samples where the node itself is anomalous
    (S > score_threshold) are marked; flagged events are shaded.
    """
    cfg = _merge_config(config)
    d = _window(res["data"], start, end)
    pred = _window(res["pred"], start, end)
    S = _window(res["S"], start, end)
    A = _window(res["A"], start, end)
    fl = _window(res["flagged"], start, end)
    nodes = res["order"]
    tz = d.index.tz
    step = pd.Timedelta(cfg["resample_freq"])
    fig, axes = plt.subplots(len(nodes) + 1, 1, figsize=(13, 1.6 * (len(nodes) + 1)), sharex=True)

    ax = axes[0]
    ax.plot(A.index, A, color="#1f3b73", lw=1.2)
    ax.axhline(cfg["score_threshold"], color="#808080", lw=0.8, ls="--")
    ax.set_ylabel("A(t)")
    ax.set_title(f"07-D2-02: {d.index[0]:%Y-%m-%d %H:%M} – {d.index[-1] + step:%Y-%m-%d %H:%M}")

    for ax, node in zip(axes[1:], nodes):
        ax.plot(d.index, d[node], color="#1f3b73", lw=1.4, label="actual")
        ax.plot(pred.index, pred[node], color="#ff9f1c", lw=1.4, ls="--",
                label="baseline" if node in cfg["root_nodes"] else "predicted")
        if node == "T_in":
            ax.plot(d.index, d["setpoint"], color="#808080", lw=1.2, ls=":", label="setpoint",
                    drawstyle="steps-post")
        bad = S[node] > cfg["score_threshold"]
        ax.scatter(d.index[bad], d.loc[bad, node], s=24, c="#d7301f", zorder=3,
                   label="node anomalous" if bad.any() else None)
        ax.set_ylabel(node)
        ax.legend(loc="upper left", ncol=4, fontsize=7, frameon=False)

    # shade flagged runs on every panel
    run_id, _ = cd._run_lengths(fl)
    for _, g in fl[fl].groupby(run_id[fl]):
        for ax in axes:
            ax.axvspan(g.index[0], g.index[-1] + step, color="#d7301f", alpha=0.12, lw=0)
    cd._finish_time_axes(fig, axes, tz)
    return fig


# Step 5: known cases
def summarize_window(res, start, end, cfg):
    """
    Flagged samples, max A and top-k root causes in [start, end). RS is
    averaged over flagged samples, or over the whole window if none are flagged.
    """
    fl = _window(res["flagged"], start, end)
    RS = _window(res["RS"], start, end)
    use = RS[fl] if fl.any() else RS
    kb = _window(res["S"]["kb_sup"], start, end) > cfg["score_threshold"]
    kb_res = _window(res["resid"]["kb_sup"], start, end)[kb]
    return {"n_flagged": int(fl.sum()), "max_A": _window(res["A"], start, end).max(),
            "top": use.mean().nlargest(cfg["top_k"]),
            "kb_sup_flagged": int(kb.sum()), "kb_sup_resid": kb_res.mean()}


def _fmt_top(top):
    return ", ".join(f"{n} ({v:.2f})" for n, v in top.items())


def check_known_cases(res, config=None):
    """Print the top root cause for the known cases and for nights with heating spikes."""
    cfg = _merge_config(config)
    print("\n" + "=" * 78 + "\nKNOWN CASES\n" + "=" * 78)
    for label, start, end in cfg["cases"]:
        s = summarize_window(res, start, end, cfg)
        basis = "flagged samples" if s["n_flagged"] else "whole window, nothing flagged"
        print(f"\n{label}: {s['n_flagged']} flagged samples, max A = {s['max_A']:.4f}")
        print(f"  top causes ({basis}): {_fmt_top(s['top'])}")
        if s["kb_sup_flagged"]:
            warmer = "warmer" if s["kb_sup_resid"] > 0 else "colder"
            print(f"  kb_sup flagged in {s['kb_sup_flagged']} samples, mean residual "
                  f"{s['kb_sup_resid']:+.2f} °C ({warmer} chilled water than normal)")
        else:
            print("  kb_sup not flagged")

    # nights with a heating spike inside night_window
    d = res["data"]
    t0, t1 = cfg["night_window"]
    step = pd.Timedelta(cfg["resample_freq"])
    night = d.between_time(t0, t1, inclusive="left")
    peak = night["heating"].groupby(night.index.date).max()
    spikes = peak[peak >= cfg["night_heating_min"]]
    rows = []
    for day, h in spikes.items():
        # window bounds from the tz-aware index (avoids DST-ambiguous local times)
        idx = night.index[night.index.date == day]
        s = summarize_window(res, idx[0], idx[-1] + step, cfg)
        rows.append({"date": day, "heating_max": h, "n_flagged": s["n_flagged"],
                     "max_A": s["max_A"], "top_cause": s["top"].index[0],
                     "top_causes": _fmt_top(s["top"])})
    print(f"\nNights with heating >= {cfg['night_heating_min']:g} % between {t0} and {t1}: "
          f"{len(rows)}")
    if rows:
        nights = pd.DataFrame(rows).set_index("date")
        hit = nights[nights["n_flagged"] > 0]
        print(f"  flagged: {len(hit)} of {len(nights)}")
        print("  top cause, all spike nights:")
        print(nights["top_cause"].value_counts().to_string())
        if len(hit):
            with pd.option_context("display.width", 250, "display.max_colwidth", 80):
                print(hit.round(3).to_string())


# main workflow
def print_events(events, title):
    """Print an events table."""
    print("\n" + "=" * 78 + f"\n{title}: {len(events)}\n" + "=" * 78)
    if events.empty:
        return
    show = events.copy()
    num = show.select_dtypes("number", exclude="timedelta").columns
    show[num] = show[num].round(2)
    show["start"] = show["start"].dt.strftime("%Y-%m-%d %H:%M")
    show["end"] = show["end"].dt.strftime("%Y-%m-%d %H:%M")
    with pd.option_context("display.width", 250, "display.max_rows", 500):
        print(show.to_string())


def run_causal(dfs, config=None):
    """
    Full pipeline. Returns a dict with data (features), pred, resid, pval,
    S, RS, A, flagged, events, events_T_in, metrics, parents and order.
    """
    cfg = _merge_config(config)
    data, _ = cd.combine_data(dfs, cfg)
    d = build_features(data, cfg)
    parents, order = summary_graph(cfg)
    print("Summary graph (node <- parents), topological order:")
    for n in order:
        print(f"  {n:8s} <- {', '.join(parents[n]) or '-'}")

    print("\nMechanism models (blocked CV by calendar week):")
    pred, resid, pval = score_nodes(d, cfg)
    metrics = mechanism_metrics(d, pred)
    print("\nPer-node fit (out-of-fold; roots vs baseline):")
    print(metrics.loc[order].round(3).to_string())

    S = 1 - pval
    A = 1 - pval.min(axis=1)
    flagged = flag_samples(A, cfg)
    RS = root_cause_scores(S, parents, order, cfg["alpha"])
    events = build_events(d, A, flagged, RS, cfg)
    events_T_in = indoor_events(events, cfg)
    print(f"\nFlagged samples: {int(flagged.sum())} of {int(A.notna().sum())} "
          f"({flagged.sum() / A.notna().sum():.2%})")
    print_events(events, "(a) All mechanism events")
    print_events(events_T_in, "(b) Indoor-temperature events")

    res = {"data": d, "pred": pred[order], "resid": resid[order], "pval": pval[order],
           "S": S[order], "RS": RS, "A": A, "flagged": flagged, "events": events,
           "events_T_in": events_T_in, "metrics": metrics, "parents": parents, "order": order}
    check_known_cases(res, cfg)

    if cfg["make_plots"]:
        figure_dir = Path("./figure")
        figure_dir.mkdir(parents=True, exist_ok=True)
        plot_score_heatmap(res, *cfg["heatmap_period"], cfg).savefig(
            figure_dir / "causal_score_heatmap.png", dpi=300, bbox_inches="tight")
        plot_event(res, "2024-09-06 00:00", "2024-09-07 00:00", cfg).savefig(
            figure_dir / "causal_event_2024-09-06.png", dpi=300, bbox_inches="tight")
        plt.show()
    return res


if __name__ == "__main__":
    import pickle

    with open("./07-D2-02_data/dfs.pkl", "rb") as f:
        dfs = pickle.load(f)

    res = run_causal(dfs)
