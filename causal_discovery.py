from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.ensemble import HistGradientBoostingRegressor

import conditional_dist as cd
import causal_anomaly_detection as cad


# Configuration (graph, features and models come from causal_anomaly_detection.CONFIG)
CONFIG = {
    "target": "T_in",

    "n_repeats": 3,
    "permute_within_month": True,       # swap days only within the same calendar month
    "permute_within_day_type": True,
    "seed": 0,
    "report_cv": True,                  # print cross-validated R^2 / MAE of each mechanism

    "units": {
        "T_in": "°C", "setpoint": "°C", "T_out": "°C", "solar": "W/m²", "co2": "ppm",
        "RH": "%", "heating": "%", "cooling": "%", "vav": "%",
        "vs_sup": "°C", "vs_ret": "°C", "kb_sup": "°C", "kb_ret": "°C",
    },

    # output
    "csv_dir": "./analysis csv",
    "make_plots": True,
}


# Helpers
def _merge_config(config=None):
    """CONFIG on top of causal_anomaly_detection's configuration, then the user overrides in 'config'."""
    return cad._merge_config(cd._merge_config(config, base=CONFIG))


def _predict(model, d, feats):
    """Model prediction where all features are present, NaN elsewhere."""
    ok = d[feats].notna().all(axis=1)
    out = pd.Series(np.nan, index=d.index)
    out[ok] = model.predict(d.loc[ok, feats])
    return out


def _children(parents, order):
    return {n: [c for c in order if n in parents[c]] for n in order}


def descendants(node, children, order):
    """All nodes downstream of 'node', in topological order."""
    found, stack = set(), list(children[node])
    while stack:
        c = stack.pop()
        if c not in found:
            found.add(c)
            stack += children[c]
    return [n for n in order if n in found]


# Step 1: mechanism models
def fit_models(d, cfg):
    """One HistGradientBoostingRegressor per non-root node, fitted on all rows with complete parents."""
    models = {}
    for node in cad.model_nodes(cfg):
        feats = cfg["graph"][node]
        rows = d[feats + [node]].dropna()
        models[node] = HistGradientBoostingRegressor(**cfg["hgb_params"]).fit(rows[feats], rows[node])
    return models


def cv_metrics(d, cfg):
    """Cross-validated R^2 and MAE of each mechanism (blocked by calendar week)."""
    pred = pd.DataFrame({n: cad.fit_mechanism(d, n, cfg["graph"][n], cfg) for n in cad.model_nodes(cfg)})
    return cad.mechanism_metrics(d, pred)


# Step 2: interventions
def day_groups(d, cfg):
    """Label per local date; days are only swapped within the same label (month and/or day type)."""
    workday = d["workday"].groupby(pd.Index(d.index.date)).first()
    label = pd.Series("all", index=workday.index)
    if cfg["permute_within_month"]:
        label = label + "_" + [f"{x:%Y-%m}" for x in workday.index]
    if cfg["permute_within_day_type"]:
        label = label + "_" + workday.map({1: "workday", 0: "off"})
    return label


def permute_days(s, groups, rng):
    """
    Swap whole days of 's' at random within each group of 'groups' (date ->
    label). The daily profile is kept, e.g. no sun at night, but the link to
    the other variables on that day is broken.
    """
    date = pd.Index(s.index.date)
    tod = s.index.strftime("%H:%M")
    lookup = s.groupby([date, tod]).first()
    days = pd.Series(groups.index, index=groups.index)
    src = days.groupby(groups).transform(lambda g: rng.permutation(g.to_numpy()))
    idx = pd.MultiIndex.from_arrays([date.map(src), tod])
    return pd.Series(lookup.reindex(idx).to_numpy(), index=s.index)


def intervene(data, node, values, models, base, children, order, cfg):
    """
    Replace 'node' by 'values' and propagate it down the graph. Each
    descendant keeps its observed residual and moves by the change in its
    model prediction. Open loop: the lagged T_in seen by the controllers stays
    as observed, so the controllers do not react to the change they cause.

    Returns (change of each child's prediction, direct and total change of the target).
    """
    tgt = cfg["target"]
    x = data.copy()
    x[node] = values
    d = cad.build_features(x, cfg)
    delta = {c: _predict(models[c], d, cfg["graph"][c]) - base[c] for c in children[node]}
    direct = delta.get(tgt, pd.Series(0.0, index=data.index))
    for c in descendants(node, children, order):
        if c == tgt:
            continue
        change = (_predict(models[c], d, cfg["graph"][c]) - base[c]).fillna(0)
        x[c] = (data[c] + change).clip(data[c].min(), data[c].max())
        d = cad.build_features(x, cfg)
    total = _predict(models[tgt], d, cfg["graph"][tgt]) - base[tgt]
    return delta, direct, total


def quantify_effects(data, models, parents, order, cfg):
    """
    Day-shuffle each node n_repeats times and measure
      impact: mean |ΔT_in| [°C], direct (only T_in's model reacts) and total
              (through all downstream mechanisms), plus the average
              sensitivity ΔT_in / Δnode (sign = direction);
      edges:  mean |Δchild| per parent -> child edge, in child units and in
              child standard deviations (strength, comparable across edges).
    """
    tgt = cfg["target"]
    children = _children(parents, order)
    d0 = cad.build_features(data, cfg)
    base = {c: _predict(m, d0, cfg["graph"][c]) for c, m in models.items()}
    groups = day_groups(d0, cfg)
    rng = np.random.default_rng(cfg["seed"])

    impact, edges = [], []
    for node in order:
        if node == tgt:
            continue
        direct, total, dxs, dts = [], [], [], []
        change = {c: [] for c in children[node]}
        for _ in range(cfg["n_repeats"]):
            values = permute_days(data[node], groups, rng)
            delta, d_dir, d_tot = intervene(data, node, values, models, base, children, order, cfg)
            for c in change:
                change[c].append(delta[c].abs().mean())
            direct.append(d_dir.abs().mean())
            total.append(d_tot.abs().mean())
            dxs.append(values - data[node])
            dts.append(d_tot)
        dx, dt = pd.concat(dxs), pd.concat(dts)
        ok = dx.notna() & dt.notna()
        unit = cfg["units"].get(node, "")
        impact.append({"node": node, "total": np.mean(total), "total_std": np.std(total),
                       "direct": np.mean(direct),
                       "sensitivity": (dx[ok] * dt[ok]).sum() / (dx[ok] ** 2).sum(),
                       "sensitivity_unit": f"{cfg['units'][tgt]}/{unit}"})
        for c, v in change.items():
            edges.append({"parent": node, "child": c, "change": np.mean(v),
                          "change_unit": cfg["units"].get(c, ""), "strength": np.mean(v) / data[c].std()})

    impact = pd.DataFrame(impact).set_index("node").sort_values("total", ascending=False)
    impact.insert(0, "rank", range(1, len(impact) + 1))
    return impact, pd.DataFrame(edges)


# Step 3: plot
def _layout(parents, order):
    """x = longest path from a root, y = evenly spread within each layer."""
    layer = {}
    for n in order:
        layer[n] = 1 + max((layer[p] for p in parents[n]), default=-1)
    pos = {}
    for k in set(layer.values()):
        col = [n for n in order if layer[n] == k]
        for i, n in enumerate(col):
            pos[n] = (k, (len(col) - 1) / 2 - i)
    return pos


def plot_causal_graph(impact, edges, parents, order, config=None):
    """
    Left: causal graph; arrow width and label = edge strength (mean |Δchild|
    in child standard deviations), node fill = total impact on the target.
    Right: impact order on the target, direct vs total, with the average sensitivity.
    """
    cfg = _merge_config(config)
    tgt = cfg["target"]
    pos = _layout(parents, order)
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(16, 7.5), layout="constrained",
                                  gridspec_kw={"width_ratios": [1.5, 1]})

    # edges; edges that skip a layer are curved so they pass around the middle nodes
    s_max = edges["strength"].max()
    rads = []
    for _, e in edges.iterrows():
        rad = 0.2 if pos[e["child"]][0] - pos[e["parent"]][0] > 1 else 0.0
        rads.append(rad)
        ax.annotate("", xy=pos[e["child"]], xytext=pos[e["parent"]],
                    arrowprops=dict(arrowstyle="-|>", color="#1f3b73", alpha=0.75,
                                    lw=0.6 + 5 * e["strength"] / s_max, mutation_scale=12,
                                    shrinkA=28, shrinkB=28, connectionstyle=f"arc3,rad={rad}"))
    # lagged feedback T_in(t-1) -> controllers (not quantified, not part of the DAG)
    for c in cad.model_nodes(cfg):
        if c != tgt and any(tgt in cfg["feature_sources"].get(f, [f]) for f in cfg["graph"][c]):
            ax.annotate("", xy=pos[c], xytext=pos[tgt],
                        arrowprops=dict(arrowstyle="-|>", color="#808080", lw=0.8, ls="--",
                                        connectionstyle="arc3,rad=0.3", shrinkA=28, shrinkB=28))

    # nodes
    cmap = plt.get_cmap("Oranges")
    norm = plt.Normalize(0, impact["total"].max())
    for n, (x, y) in pos.items():
        if n == tgt:
            ax.text(x, y, f"{n}\n(target)", ha="center", va="center", fontsize=10, color="white",
                    bbox=dict(boxstyle="round,pad=0.5", fc="#1f3b73", ec="#1f3b73"))
            continue
        v = impact.loc[n, "total"]
        ax.text(x, y, f"{n}\n{v:.2f} °C", ha="center", va="center", fontsize=9, color="#222222",
                bbox=dict(boxstyle="round,pad=0.45", fc=cmap(0.1 + 0.6 * norm(v)), ec="#1f3b73", lw=0.8))
    xs, ys = zip(*pos.values())
    ax.set_xlim(min(xs) - 0.35, max(xs) + 0.35)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    ax.axis("off")
    ax.legend(handles=[Line2D([], [], color="#1f3b73", lw=2, label="causal edge: width/label = strength "
                                                                   "(mean |Δchild| / std(child))"),
                       Line2D([], [], color="#808080", lw=0.8, ls="--", label=f"{tgt}(t-1) feedback to controllers")],
              loc="lower center", bbox_to_anchor=(0.5, -0.06), fontsize=8, frameon=False)
    ax.set_title(f"Causal graph; node value and fill = total impact on {tgt}, mean |Δ{tgt}| [°C]",
                 loc="left")

    # impact order
    imp = impact.sort_values("total")
    y = np.arange(len(imp))
    axb.barh(y + 0.2, imp["total"], height=0.38, color="#1f3b73", xerr=imp["total_std"],
             error_kw=dict(ecolor="#808080", lw=0.8), label="total (through the graph)")
    axb.barh(y - 0.2, imp["direct"], height=0.38, color="#ff9f1c", label=f"direct ({tgt} model only)")
    for yi, (_, r) in zip(y, imp.iterrows()):
        axb.text(r["total"] + r["total_std"] + 0.005, yi + 0.2,
                 f"{r['sensitivity']:+.4f} {r['sensitivity_unit']}", va="center", fontsize=7, color="#555555")
    axb.set_yticks(y)
    axb.set_yticklabels([f"{r['rank']}. {n}" for n, r in imp.iterrows()])
    axb.set_xlabel(f"mean |Δ{tgt}| [°C] when the node's days are shuffled")
    axb.set_xlim(0, (imp["total"] + imp["total_std"]).max() * 1.35)
    axb.grid(True, axis="x", color="#e5e5e5", lw=0.6)
    axb.spines[["top", "right"]].set_visible(False)
    axb.legend(loc="lower right", fontsize=8, frameon=False)
    axb.set_title(f"Impact order on {tgt} (label: average sensitivity)", loc="left")

    # edge labels on the drawn curves; arc3 is defined in display coordinates, so fix the layout first
    fig.canvas.draw()
    t = 0.35
    for (_, e), rad in zip(edges.iterrows(), rads):
        p0, p2 = ax.transData.transform([pos[e["parent"]], pos[e["child"]]])
        dx, dy = p2 - p0
        ctrl = (p0 + p2) / 2 + rad * np.array([dy, -dx])
        x, y = ax.transData.inverted().transform((1 - t) ** 2 * p0 + 2 * t * (1 - t) * ctrl + t ** 2 * p2)
        ax.text(x, y, f"{e['strength']:.2f}", fontsize=7, ha="center", va="center", color="#222222",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))
    return fig


# main workflow
def run_discovery(dfs, config=None):
    """
    Fit the mechanisms of the causal graph and quantify how each node
    influences the target. Returns a dict with data, models, impact, edges,
    metrics, parents and order.
    """
    cfg = _merge_config(config)
    data, _ = cd.combine_data(dfs, cfg)
    d = cad.build_features(data, cfg)
    parents, order = cad.summary_graph(cfg)
    print("Causal graph, topological order:")
    for n in order:
        print(f"  {n:8s} <- {', '.join(parents[n]) or '-'}")

    metrics = None
    if cfg["report_cv"]:
        metrics = cv_metrics(d, cfg)
        print("\nMechanism fit (blocked CV by calendar week):")
        print(metrics.round(3).to_string())

    models = fit_models(d, cfg)
    impact, edges = quantify_effects(data, models, parents, order, cfg)
    with pd.option_context("display.width", 200):
        print(f"\nImpact on {cfg['target']} (mean |Δ{cfg['target']}| [°C], "
              f"{cfg['n_repeats']} day shuffles per node):")
        print(impact.round(4).to_string())
        print("\nEdge strength (mean |Δchild| in child units and in child std):")
        print(edges.sort_values("strength", ascending=False).round(4).to_string(index=False))

    csv_dir = Path(cfg["csv_dir"])
    csv_dir.mkdir(parents=True, exist_ok=True)
    impact.round(4).to_csv(csv_dir / f"causal_impact_{cfg['target']}.csv")
    edges.round(4).to_csv(csv_dir / "causal_edges.csv", index=False)
    print(f"\nTables -> {csv_dir}")

    if cfg["make_plots"]:
        figure_dir = Path("./figure")
        figure_dir.mkdir(parents=True, exist_ok=True)
        plot_causal_graph(impact, edges, parents, order, cfg).savefig(
            figure_dir / f"causal_graph_{cfg['target']}.png", dpi=300, bbox_inches="tight")
        plt.show()
    return {"data": d, "models": models, "impact": impact, "edges": edges,
            "metrics": metrics, "parents": parents, "order": order}


if __name__ == "__main__":
    import pickle

    with open("./07-D2-02_data/dfs.pkl", "rb") as f:
        dfs = pickle.load(f)

    res = run_discovery(dfs)
