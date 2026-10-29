from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import causal_anomaly_detection as cad


# Configuration
CONFIG = {
    "target": "T_in",
    "figure_name": "causal_dag.png",
}


# Helpers
def _merge_config(config=None):
    """CONFIG on top of causal_anomaly_detection's configuration, then the user overrides in 'config'."""
    cfg = cad._merge_config(config)
    return {**CONFIG, **cfg, **(config or {})}


def _edge_features(cfg):
    """(parent, child) to the feature columns of 'child' that are computed from 'parent'."""
    feats = {}
    for child in cad.model_nodes(cfg):
        for f in cfg["graph"][child]:
            for s in cfg["feature_sources"].get(f, [f]):
                feats.setdefault((s, child), []).append(f)
    return feats


# Step 1: causal graph
def build_gcm(config=None):
    """
    Causal DAG of the GCM as a networkx DiGraph, from the summary graph in
    causal_anomaly_detection.CONFIG. Node attributes: 'root', 'layer' (longest
    path from a root) and 'inputs' (non-node features such as hour and day type).
    Edge attribute 'features': the parent's feature columns used by the child.
    The lagged feedback edges (T_in(t-1) to the controllers) are kept out of the
    DAG and listed in G.graph["feedback"].
    """
    cfg = _merge_config(config)
    parents, order = cad.summary_graph(cfg)
    feats = _edge_features(cfg)

    G = nx.DiGraph(target=cfg["target"])
    for n in order:
        inputs = [f for f in cfg["graph"].get(n, [])
                  if not any(s in parents for s in cfg["feature_sources"].get(f, [f]))]
        G.add_node(n, root=n in cfg["root_nodes"], inputs=inputs)
    for child, ps in parents.items():
        for p in ps:
            G.add_edge(p, child, features=feats[(p, child)])
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError(f"causal graph has a cycle: {nx.find_cycle(G)}")

    for k, layer in enumerate(nx.topological_generations(G)):
        for n in layer:
            G.nodes[n]["layer"] = k
    G.graph["feedback"] = [(p, c, feats[(p, c)]) for (p, c) in feats
                           if p in cfg["feedback_parents"] and p != c and c in G]
    return G


# Step 2: plot
def _layout(G):
    """x = layer; within a layer, nodes sorted by the mean y of their parents to reduce crossings."""
    pos = {}
    for k, layer in enumerate(nx.topological_generations(G)):
        if k > 0:
            layer = sorted(layer, key=lambda n: -sum(pos[p][1] for p in G.predecessors(n)) / G.in_degree(n))
        for i, n in enumerate(layer):
            pos[n] = (k, (len(layer) - 1) / 2 - i)
    return pos


def _draw_edge(ax, boxes, u, v, rad, label=None, **arrow):
    """Arrow from box u to box v, clipped at the box outlines; label at 40% along the curve."""
    ax.annotate("", xy=boxes[v].get_position(), xytext=boxes[u].get_position(),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=14, connectionstyle=f"arc3,rad={rad}",
                                patchA=boxes[u].get_bbox_patch(), patchB=boxes[v].get_bbox_patch(),
                                shrinkA=2, shrinkB=2, **arrow))
    if label:
        # point on the arc3 Bezier curve, which is defined in display coordinates
        p0, p2 = ax.transData.transform([boxes[u].get_position(), boxes[v].get_position()])
        dx, dy = p2 - p0
        ctrl = (p0 + p2) / 2 + rad * np.array([dy, -dx])
        t = 0.4
        x, y = ax.transData.inverted().transform((1 - t) ** 2 * p0 + 2 * t * (1 - t) * ctrl + t ** 2 * p2)
        ax.text(x, y, label, fontsize=7, ha="center", va="center", color="#222222",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))


def plot_dag(G):
    """Layered DAG, roots on the left; edge labels = parent feature columns, when they differ from the parent name."""
    tgt = G.graph["target"]
    pos = _layout(G)
    fig, ax = plt.subplots(figsize=(13, 7.5), layout="constrained")
    xs, ys = zip(*pos.values())
    ax.set_xlim(min(xs) - 0.3, max(xs) + 0.3)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.6)
    ax.axis("off")

    # nodes: roots, mechanisms, target
    boxes = {}
    for n, (x, y) in pos.items():
        if n == tgt:
            fc, ec, color = "#1f3b73", "#1f3b73", "white"
        elif G.nodes[n]["root"]:
            fc, ec, color = "#fde3c4", "#ff9f1c", "#222222"
        else:
            fc, ec, color = "#eef2f9", "#1f3b73", "#222222"
        boxes[n] = ax.text(x, y, n, ha="center", va="center", fontsize=10, color=color, zorder=3,
                           bbox=dict(boxstyle="round,pad=0.5", fc=fc, ec=ec, lw=0.8))
    fig.canvas.draw()   # box sizes are needed to clip the arrows

    # edges; edges that skip a layer are curved so they pass around the middle nodes
    for u, v, f in G.edges(data="features"):
        rad = 0.15 if pos[v][0] - pos[u][0] > 1 else 0.0
        _draw_edge(ax, boxes, u, v, rad, None if f == [u] else ", ".join(f),
                   color="#1f3b73", alpha=0.75, lw=1.2)
    for p, c, _ in G.graph["feedback"]:
        _draw_edge(ax, boxes, p, c, 0.25, color="#808080", lw=0.8, ls="--")

    inputs = sorted({f for n in G for f in G.nodes[n]["inputs"]})
    ax.legend(handles=[Line2D([], [], color="#ff9f1c", marker="s", ls="", ms=10, mfc="#fde3c4",
                              label="root node"),
                       Line2D([], [], color="#1f3b73", lw=1.5, label="causal edge (label: parent features used)"),
                       Line2D([], [], color="#808080", lw=0.8, ls="--",
                              label=f"{tgt}(t-1) feedback to controllers (not in the DAG)")],
              loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=8, frameon=False)
    ax.set_title(f"Causal DAG of the GCM; every non-root mechanism also takes {', '.join(inputs)}",
                 loc="left")
    return fig


# main workflow
def run_gcm(config=None):
    """Build the causal DAG, print it and save the figure. Returns the networkx DiGraph."""
    cfg = _merge_config(config)
    G = build_gcm(cfg)
    print(f"Causal DAG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    for n in nx.topological_sort(G):
        ps = [f"{p} ({', '.join(f)})" for p, _, f in G.in_edges(n, data="features")]
        print(f"  {n:19s} <- {', '.join(ps) or '-'}")
    print("Lagged feedback (left out):", ", ".join(f"{p} -> {c}" for p, c, _ in G.graph["feedback"]))

    if cfg["make_plots"]:
        figure_dir = Path("./figure")
        figure_dir.mkdir(parents=True, exist_ok=True)
        plot_dag(G).savefig(figure_dir / cfg["figure_name"], dpi=300, bbox_inches="tight")
        print(f"Figure -> {figure_dir / cfg['figure_name']}")
        plt.show()
    return G


if __name__ == "__main__":
    G = run_gcm()
