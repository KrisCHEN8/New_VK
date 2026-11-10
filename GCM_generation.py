from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import holidays


T = "indoor_temperature"

# parent, child, feature columns names
EDGES = [
    # the supply temperature in radiator circuit is determined by outsoor temperature
    ("outdoor_temperature", "VS02_supply", ["outdoor_temperature"]),

    # control actuators inside the room
    (T, "cooling_valve", [T]),
    (T, "heating_valve", [T]),
    (T, "VAV_damper", [T]),
    ("co2", "VAV_damper", ["co2"]),

    # # hydraulics, same interval
    # ("cooling_valve", "cooling_flow", ["cooling_valve"]),
    # ("heating_valve", "heating_flow", ["heating_valve"]),

    # physical effects on the room
    (T, T, [f"{T}_lag1"]),                                               # thermal inertia
    ("cooling_valve", T, ["cooling_valve_lag1"]),                          # chilled beam, faster
    ("KB02_supply", T, ["KB02_supply_lag1"]),
    ("relative_humidity", "KB02_supply", ["humidity_lag1"]),
    ("heating_valve", T, ["heating_valve_lag1", "heating_valve_lag2"]),     # radiator, slower
    ("VS02_supply", T, ["VS02_supply_lag1", "VS02_supply_lag2"]),
    ("VAV_damper", T, ["VAV_damper_lag1"]),
    ("co2", T, ["co2", "co2_lag1"]),                                     # occupancy proxy
    ("solar_radiation", T, ["solar_radiation_ma4"]),                     # 2-hour moving average, 4 * 30mins
    ("outdoor_temperature", T, ["outdoor_temperature_ma6"]),            # 3-hour moving average, 6 * 30mins
]

# Measured variables that are irrelevant 
EXCLUDED = {
    "VS02_return": "circuit-level return, mixes all rooms on the circuit",
    "KB02_return": "circuit-level return, mixes all rooms on the circuit",
}


# Configuration
CONFIG = {
    "target": T,
    "edges": EDGES,
    "excluded": EXCLUDED,
    "lag_node_sources": [T],
    "lag_label": "t-1",
    "time_inputs": ["hour_sin", "workday"],   # non-node inputs of every non-root mechanism
    "figure_name": "causal_dag.png",
    "make_plots": True,
}


# Helpers
def _merge_config(config=None):
    """CONFIG, then the user overrides in 'config'."""
    return {**CONFIG, **(config or {})}


# causal graph
def build_gcm(config=None):
    """
    Causal DAG of the GCM as a networkx DiGraph, from the edge list in CONFIG.
    Node attributes: 'root' (no parents), 'layer' (longest path from a root)
    and 'inputs' (non-node features such as hour and day type).
    Edge attribute 'features': the parent's feature columns used by the child.
    Edges into the target are lagged (physical effects); all other edges are
    within the same 30-min interval. Every edge that leaves a node in
    'lag_node_sources' starts from its lag node instead, e.g.
    indoor_temperature(t-1), which has the attribute 'lag_of' = its node family.
    """
    cfg = _merge_config(config)
    lagged = {s: f"{s}({cfg['lag_label']})" for s in cfg["lag_node_sources"]}

    G = nx.DiGraph(target=cfg["target"])
    children = {child for _, child, _ in cfg["edges"]}
    for parent, child, _ in cfg["edges"]:
        for n in (parent, child):
            if n not in G:
                root = n not in children
                G.add_node(n, root=root, inputs=[] if root else list(cfg["time_inputs"]))
    for parent, child, feats in cfg["edges"]:
        if parent not in lagged:
            G.add_edge(parent, child, features=list(feats))
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError(f"causal graph has a cycle: {nx.find_cycle(G)}")
    for k, layer in enumerate(nx.topological_generations(G)):
        for n in layer:
            G.nodes[n]["layer"] = k

    # lag nodes
    for parent, child, feats in cfg["edges"]:
        if parent in lagged:
            G.add_node(lagged[parent], root=False, inputs=[], lag_of=parent, layer=G.nodes[parent]["layer"])
            G.add_edge(lagged[parent], child, features=list(feats))
    return G


# plot
def _layout(G):
    """
    x = layer; within a layer, nodes sorted by the mean y of their parents to reduce crossings.
    A lag node sits directly below its node family (same variable, earlier time).
    """
    lag_of = nx.get_node_attributes(G, "lag_of")
    H = G.subgraph([n for n in G if n not in lag_of])
    pos = {}
    for k, layer in enumerate(nx.topological_generations(H)):
        if k > 0:
            layer = sorted(layer, key=lambda n: -sum(pos[p][1] for p in H.predecessors(n)) / H.in_degree(n))
        layer = [m for n in layer for m in [n] + [lag for lag, fam in lag_of.items() if fam == n]]
        for i, n in enumerate(layer):
            pos[n] = (k, (len(layer) - 1) / 2 - i)
    return pos


def _curve(ax, p_u, p_v, rad, t):
    """Points of the arc3 Bezier curve from p_u to p_v (data coordinates) at the parameters t."""
    p0, p2 = ax.transData.transform([p_u, p_v])   # arc3 is defined in display coordinates
    dx, dy = p2 - p0
    ctrl = (p0 + p2) / 2 + rad * np.array([dy, -dx])
    t = np.atleast_1d(t)[:, None]
    return ax.transData.inverted().transform((1 - t) ** 2 * p0 + 2 * t * (1 - t) * ctrl + t ** 2 * p2)


def _pick_rad(ax, pos, u, v, candidates=(0.0, 0.15, -0.15, 0.3, -0.3)):
    """Curvature from 'candidates' that keeps the edge u to v furthest from the nodes of the layers it skips."""
    lo, hi = sorted((pos[u][0], pos[v][0]))
    between = [(x, y) for n, (x, y) in pos.items() if lo < x < hi]
    if not between:
        return 0.0

    def clearance(rad):
        xy = _curve(ax, pos[u], pos[v], rad, np.linspace(0, 1, 201))
        return min(abs(xy[np.argmin(abs(xy[:, 0] - x)), 1] - y) for x, y in between)

    return max(candidates, key=lambda r: round(clearance(r), 2))


def _draw_edge(ax, boxes, u, v, rad, label=None, **arrow):
    """Arrow from box u to box v, clipped at the box outlines; label at 40% along the curve."""
    ax.annotate("", xy=boxes[v].get_position(), xytext=boxes[u].get_position(),
                arrowprops=dict(arrowstyle="-|>", mutation_scale=14, connectionstyle=f"arc3,rad={rad}",
                                patchA=boxes[u].get_bbox_patch(), patchB=boxes[v].get_bbox_patch(),
                                shrinkA=2, shrinkB=2, **arrow))
    if label:
        x, y = _curve(ax, boxes[u].get_position(), boxes[v].get_position(), rad, 0.4)[0]
        ax.text(x, y, label, fontsize=7, ha="center", va="center", color="#222222",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))


def _edge_label(G, parent, features):
    """Lags / moving averages of the parent used by the child, e.g. 'lag1, lag2' or 'ma12'; None if the node says it."""
    name = G.nodes[parent].get("lag_of", parent)
    if parent != name or features == [name]:
        return None   # edge from a lag node, or same-interval edge
    return ", ".join("lag0" if f == name else f.removeprefix(name + "_") for f in features)


def plot_dag(G):
    """
    Layered DAG, roots on the left; edge labels = lags or moving averages of the parent (none = same interval).
    Lag nodes (dashed outline) sit below their node family.
    """
    tgt = G.graph["target"]
    lag_of = nx.get_node_attributes(G, "lag_of")
    pos = _layout(G)
    fig, ax = plt.subplots(figsize=(13, 7.5), layout="constrained")
    xs, ys = zip(*pos.values())
    ax.set_xlim(min(xs) - 0.3, max(xs) + 0.3)
    ax.set_ylim(min(ys) - 1.1, max(ys) + 0.6)   # room below for the edges from the lag nodes
    ax.axis("off")

    # nodes: roots, mechanisms, target, lag nodes
    boxes = {}
    for n, (x, y) in pos.items():
        ls = "-"
        if n in lag_of:
            fc, ec, color, ls = "white", "#1f3b73", "#222222", "--"
        elif n == tgt:
            fc, ec, color = "#1f3b73", "#1f3b73", "white"
        elif G.nodes[n]["root"]:
            fc, ec, color = "#fde3c4", "#ff9f1c", "#222222"
        else:
            fc, ec, color = "#eef2f9", "#1f3b73", "#222222"
        boxes[n] = ax.text(x, y, n, ha="center", va="center", fontsize=10, color=color, zorder=3,
                           bbox=dict(boxstyle="round,pad=0.5", fc=fc, ec=ec, lw=0.8, ls=ls))
    fig.canvas.draw()   # box sizes are needed to clip the arrows

    # edges; edges that skip a layer take the curvature that passes best between the middle nodes
    for u, v, f in G.edges(data="features"):
        _draw_edge(ax, boxes, u, v, _pick_rad(ax, pos, u, v), _edge_label(G, u, f),
                   color="#1f3b73", alpha=0.75, lw=1.2)

    inputs = sorted({f for n in G for f in G.nodes[n]["inputs"]})
    handles = [Line2D([], [], color="#ff9f1c", marker="s", ls="", ms=10, mfc="#fde3c4", label="root node"),
               Line2D([], [], color="#1f3b73", lw=1.5,
                      label="causal edge (label: lag / moving average; none = same interval)")]
    if lag_of:
        handles.append(Line2D([], [], color="#1f3b73", marker="s", ls="", ms=10, mfc="white", mew=0.8,
                              label="lag node: previous value of the node above it"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=8,
              frameon=False)
    title = "Graphical Causal Model (30-min data)"
    # if inputs:
    #     title += f"; every non-root mechanism also takes {', '.join(inputs)}"
    ax.set_title(title, loc="left")
    return fig


# main workflow
def run_gcm(config=None):
    """Build the causal DAG, print it and save the figure. Returns the networkx DiGraph."""
    cfg = _merge_config(config)
    G = build_gcm(cfg)
    lag_of = nx.get_node_attributes(G, "lag_of")
    print(f"Causal DAG: {G.number_of_nodes()} nodes ({len(lag_of)} lag nodes), {G.number_of_edges()} edges")
    for n in nx.topological_sort(G):
        if n in lag_of:
            continue
        ps = [f"{p} ({', '.join(f)})" for p, _, f in G.in_edges(n, data="features")]
        print(f"  {n:19s} <- {', '.join(ps) or '-'}")
    print("Lag nodes:", ", ".join(f"{n} (of {fam})" for n, fam in lag_of.items()) or "-")
    print("Not in the DAG:", ", ".join(cfg["excluded"]))

    if cfg["make_plots"]:
        figure_dir = Path("./figure")
        figure_dir.mkdir(parents=True, exist_ok=True)
        plot_dag(G).savefig(figure_dir / cfg["figure_name"], dpi=300, bbox_inches="tight")
        print(f"Figure -> {figure_dir / cfg['figure_name']}")
        plt.show()
    return G


if __name__ == "__main__":
    G = run_gcm()
