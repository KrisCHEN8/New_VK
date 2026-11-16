from pathlib import Path
import re

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import holidays


T = "indoor_temperature"
CAL = "calendar"
CAL_FEATURES = ["hour_sin", "hour_cos", "workday"]

# parent, child, feature column names
EDGES = [
    # set-point curves
    ("outdoor_temperature", "VS02_supply", ["outdoor_temperature"]),     # VS02 heating curve
    ("relative_humidity", "KB02_supply", ["relative_humidity_lag1"]),       # chilled water setpoint to avoid condensation
    ("outdoor_temperature", "AHU_supply_temperature", ["outdoor_temperature"]),  # AHU setpoint curve

    # room controllers react to the room with rather fast response
    (T, "cooling_valve", [T]),
    (T, "heating_valve", [T]),
    (T, "VAV_damper", [T]),
    ("co2", "VAV_damper", ["co2"]),    # demand-controlled ventilation
    # (CAL, "cooling_valve", CAL_FEATURES),     # schedules, night setback
    # (CAL, "heating_valve", CAL_FEATURES),
    (CAL, "VAV_damper", CAL_FEATURES),
    # ("setpoint", "cooling_valve", ["setpoint"]),        # the setpoint of roo temeprature
    # ("setpoint", "heating_valve", ["setpoint"]),

    # # hydraulics, same interval
    # ("cooling_valve", "cooling_flow", ["cooling_valve"]),
    # ("heating_valve", "heating_flow", ["heating_valve"]),

    # physical effects on the room (delayed)
    (T, T, [f"{T}_lag1"]),                                          # thermal inertia
    ("cooling_valve", T, ["cooling_valve_lag1"]),                           # chilled beam, faster
    ("KB02_supply", T, ["KB02_supply_lag1"]),
    ("heating_valve", T, ["heating_valve_lag1", "heating_valve_lag2"]),     # radiator, slower
    ("VS02_supply", T, ["VS02_supply_lag1", "VS02_supply_lag2"]),
    ("VAV_damper", T, ["VAV_damper_lag1"]),
    ("AHU_supply_temperature", T, ["AHU_supply_temperature_lag1"]),
    ("solar_radiation", T, ["solar_radiation_ma4"]),                        # 2-hour moving average, 4 * 30 min
    ("outdoor_temperature", T, ["outdoor_temperature_ma6"]),                # 3-hour moving average, 6 * 30 min

    # proxies for unmeasured drivers of the room temperature
    ("co2", T, ["co2", "co2_lag1"]),                                        # occupants
    (CAL, T, CAL_FEATURES),                                                 # scheduled internal gains
]


# Components: the variables that act on the room together through one emitter.
# Their edges into the room are cut together when the strength of the component is quantified.
COMPONENTS = {
    "chilled_beam": ["cooling_valve", "KB02_supply", "VAV_damper", "AHU_supply_temperature"],
    "radiator": ["heating_valve", "VS02_supply"],
}

# Emitters' joint effects, per component
INTERACTIONS = {
    T: {
        # chilled beam, air side: heat removed by the primary air
        "chilled_beam_air": "VAV_damper_lag1 * (indoor_temperature_lag1 - AHU_supply_temperature_lag1)",
        # chilled beam, water side: heat removed by the coil; its capacity grows with the primary air flow
        "chilled_beam_water": ("cooling_valve_lag1 * VAV_damper_lag1"
                               " * (indoor_temperature_lag1 - KB02_supply_lag1)"),
        # radiator: heat supplied
        "radiator_drive": ("(heating_valve_lag1 + heating_valve_lag2) / 2"
                           " * ((VS02_supply_lag1 + VS02_supply_lag2) / 2 - indoor_temperature_lag1)"),
    },
}

# Monotonic effects
MONOTONE = {
    T: {
        "solar_radiation_ma4": +1, "outdoor_temperature_ma6": +1,
        "chilled_beam_air": -1, "chilled_beam_water": -1, "radiator_drive": +1,
    },
    "cooling_valve": {T: +1},
    "heating_valve": {T: -1},
    "VS02_supply": {"outdoor_temperature": -1},
}

# Measured variables that are irrelevant
EXCLUDED = {
    "VS02_return": "circuit-level return, mixes all rooms on the circuit",
    "KB02_return": "circuit-level return, mixes all rooms on the circuit",
}

# Configuration
CONFIG = {
    "target": T,
    "edges": EDGES,
    "components": COMPONENTS,
    "interactions": INTERACTIONS,
    "monotone": MONOTONE,
    "excluded": EXCLUDED,
    "lag_node_sources": [T, "cooling_valve", "heating_valve", "VAV_damper"],   # variables in a loop with the target
    "proxy_nodes": ["co2", CAL],          # edges from these into the target stand for unmeasured drivers
    "calendar_node": CAL,
    "calendar_features": CAL_FEATURES,
    "display_names": {CAL: "calendar\n(hour, workday)"},
    "freq": "30min",
    "country": "SE",                      # public holidays for 'workday'
    "regime_valves": {"heating": "heating_valve", "cooling": "cooling_valve"},
    # "valve_active_threshold": 5.0,        # daily mean opening above which an emitter counts as active (% open; use 0.05 for 0-1 signals)
    "figure_name": "causal_dag.png",
    "make_plots": True,
}


# Helpers
def _merge_config(config=None):
    """CONFIG, then the user overrides in 'config'."""
    return {**CONFIG, **(config or {})}


def _feature_lags(var, feat, cfg):
    """Time lags (in intervals) that the feature column 'feat' of the variable 'var' covers."""
    if var == cfg["calendar_node"]:
        if feat not in cfg["calendar_features"]:
            raise ValueError(f"calendar feature '{feat}' is not in calendar_features")
        return [0]
    if feat == var:
        return [0]
    m = re.fullmatch(rf"{re.escape(var)}_(lag|ma)(\d+)", feat)
    if not m:
        raise ValueError(f"feature '{feat}' of '{var}' must be '{var}', '{var}_lag<k>' or '{var}_ma<w>'")
    k = int(m.group(2))
    return [k] if m.group(1) == "lag" else list(range(k))


def _lag_node_name(var, lags):
    return f"{var}({', '.join(f't-{k}' for k in sorted(lags))})"


# causal graph
def build_gcm(config=None):
    cfg = _merge_config(config)
    tgt, lag_sources = cfg["target"], set(cfg["lag_node_sources"])

    # lag of every feature, and the routing of each edge
    routed, lag_node_lags = [], {}
    for parent, child, feats in cfg["edges"]:
        lags = [min(_feature_lags(parent, f, cfg)) for f in feats]
        if parent in lag_sources and min(lags) >= 1:
            lag_node_lags.setdefault(parent, set()).update(k for f in feats for k in _feature_lags(parent, f, cfg))
            routed.append((parent, child, feats, True))
        elif parent in lag_sources and max(lags) >= 1:
            raise ValueError(f"edge {parent} -> {child} mixes same-interval and lagged columns {feats}; "
                             "split it into two edges")
        else:
            routed.append((parent, child, feats, False))

    G = nx.DiGraph(target=tgt)
    children = {child for _, child, _ in cfg["edges"]}
    names = cfg["display_names"]
    for parent, child, _, _ in routed:
        for n in (parent, child):
            if n not in G:
                G.add_node(n, root=n not in children, inputs=[], label=names.get(n, n))
    for parent, child, feats, via_lag in routed:
        if not via_lag:
            G.add_edge(parent, child, features=list(feats))
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError(f"same-interval edges form a cycle {nx.find_cycle(G)}: add the variable that acts "
                         "back on the loop with a delay to 'lag_node_sources'")
    for k, layer in enumerate(nx.topological_generations(G)):
        for n in layer:
            G.nodes[n]["layer"] = k

    # lag nodes
    for parent, child, feats, via_lag in routed:
        if via_lag:
            lag = _lag_node_name(parent, lag_node_lags[parent])
            if lag not in G:
                G.add_node(lag, root=True, inputs=[], lag_of=parent, layer=G.nodes[parent]["layer"], label=lag)
            G.add_edge(lag, child, features=list(feats))

    # edge kinds and components
    component_of = {var: comp for comp, members in cfg["components"].items() for var in members}
    unknown = set(component_of) - {G.nodes[n].get("lag_of", n) for n in G}
    if unknown:
        raise ValueError(f"components list {sorted(unknown)}, which are not variables of the graph")
    for u, v in G.edges:
        src = G.nodes[u].get("lag_of", u)
        G.edges[u, v]["component"] = component_of.get(src)
        G.edges[u, v]["kind"] = ("control" if v != tgt else "proxy" if src in cfg["proxy_nodes"] else "physical")

    # mechanism settings
    for n in G:
        if G.in_degree(n) == 0:
            continue
        inputs = mechanism_inputs(G, n)
        inter = dict(cfg["interactions"].get(n, {}))
        for name, expr in inter.items():
            missing = set(re.findall(r"[A-Za-z_]\w*", expr)) - set(inputs)
            if missing:
                raise ValueError(f"interaction '{name}' of {n} uses {sorted(missing)}, which are not inputs of {n}")
        mono = dict(cfg["monotone"].get(n, {}))
        unknown = set(mono) - set(inputs) - set(inter)
        if unknown:
            raise ValueError(f"monotone constraint on {sorted(unknown)}, which are not columns of {n}'s mechanism")
        G.nodes[n].update(interactions=inter, monotone=mono)
    return G


# mechanisms
def mechanism_inputs(G, node):
    """Feature columns of the node's mechanism, in edge order, without duplicates."""
    cols = [f for _, _, feats in G.in_edges(node, data="features") for f in feats]
    return list(dict.fromkeys(cols))


def edge_bundles(G, node, by_component=True):
    """
    Feature columns to permute together when an edge is cut. With by_component, the edges of one
    component (chilled beam, radiator) form a single bundle keyed by the component name; the other
    edges, and all edges with by_component=False, are keyed by the parent.
    """
    bundles = {}
    for u, _, d in G.in_edges(node, data=True):
        key = d["component"] if by_component and d["component"] else u
        bundles.setdefault(key, []).extend(f for f in d["features"] if f not in bundles[key])
    return bundles


def design_matrix(G, node, inputs):
    """
    Mechanism inputs plus the node's interaction terms, computed from 'inputs'
    """
    X = inputs[mechanism_inputs(G, node)].copy()
    for name, expr in G.nodes[node]["interactions"].items():
        X[name] = X.eval(expr)
    return X


def monotone_constraints(G, node, columns):
    """Signs for the design-matrix columns."""
    return [G.nodes[node]["monotone"].get(c, 0) for c in columns]


def node_frame(G, node, feats):
    """Rows with complete data for one mechanism: target series y and design matrix X."""
    cols = list(dict.fromkeys([node] + mechanism_inputs(G, node)))
    data = feats[cols].dropna()
    return data[node], design_matrix(G, node, data)


# data
def make_features(df, G, config=None):
    """
    All columns the GCM needs, from raw data with a DatetimeIndex. The index is put on a regular
    grid first, so a gap in the data gives NaN lags instead of shifting values across the gap.
    """
    cfg = _merge_config(config)
    df = df.sort_index().asfreq(cfg["freq"])
    raw = [n for n in G if "lag_of" not in G.nodes[n] and n != cfg["calendar_node"]]
    missing = [c for c in raw if c not in df]
    if missing:
        raise KeyError(f"raw data lacks the columns {missing}")

    out = df[raw].copy()
    for u, _, feats in G.edges(data="features"):
        var = G.nodes[u].get("lag_of", u)
        for f in feats:
            if f in out or var == cfg["calendar_node"]:
                continue
            kind, k = re.fullmatch(rf"{re.escape(var)}_(lag|ma)(\d+)", f).groups()
            k = int(k)
            out[f] = df[var].shift(k) if kind == "lag" else df[var].rolling(k, min_periods=k).mean()

    if cfg["calendar_node"] in G:
        idx = df.index
        hour = idx.hour + idx.minute / 60
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        hol = holidays.country_holidays(cfg["country"], years=range(idx.year.min(), idx.year.max() + 1))
        out["workday"] = ((idx.dayofweek < 5) & ~pd.Index(idx.date).isin(list(hol))).astype(float)
    return out


def label_regimes(df, config=None):
    """
    Operating regime of every row, decided per day from the daily mean valve openings:
    'heating', 'cooling', 'both' or 'neither'. Fit and quantify the mechanisms per regime.
    """
    cfg = _merge_config(config)
    thr = cfg["valve_active_threshold"]
    daily = df[list(cfg["regime_valves"].values())].resample("D").mean()
    heat = daily[cfg["regime_valves"]["heating"]] > thr
    cool = daily[cfg["regime_valves"]["cooling"]] > thr
    day = pd.Series(np.select([heat & cool, heat, cool], ["both", "heating", "cooling"], "neither"),
                    index=daily.index)
    return pd.Series(day.reindex(df.index.normalize()).to_numpy(), index=df.index, name="regime")


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
    """Lags / moving averages of the parent used by the child, e.g. 'lag1, lag2' or 'ma4'; None if the node says it."""
    name = G.nodes[parent].get("lag_of", parent)
    if parent != name or features == [name] or name == CONFIG["calendar_node"]:
        return None   # edge from a lag node, same-interval edge, or calendar inputs
    return ", ".join("lag0" if f == name else f.removeprefix(name + "_") for f in features)


EDGE_STYLE = {"physical": "-", "control": "--", "proxy": ":"}


def plot_dag(G):
    """
    Layered DAG, roots on the left; edge labels = lags or moving averages of the parent (none = same interval).
    Lag nodes (dashed outline) sit below their node family. Line style = edge kind.
    """
    tgt = G.graph["target"]
    lag_of = nx.get_node_attributes(G, "lag_of")
    pos = _layout(G)
    fig, ax = plt.subplots(figsize=(14, 8.5), layout="constrained")
    xs, ys = zip(*pos.values())
    ax.set_xlim(min(xs) - 0.35, max(xs) + 0.35)
    ax.set_ylim(min(ys) - 1.1, max(ys) + 0.6)   # room below for the legend
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
        boxes[n] = ax.text(x, y, G.nodes[n]["label"], ha="center", va="center", fontsize=10, color=color,
                           zorder=3, bbox=dict(boxstyle="round,pad=0.5", fc=fc, ec=ec, lw=0.8, ls=ls))
    fig.canvas.draw()   # box sizes are needed to clip the arrows

    # edges; edges that skip a layer take the curvature that passes best between the middle nodes
    for u, v, d in G.edges(data=True):
        _draw_edge(ax, boxes, u, v, _pick_rad(ax, pos, u, v), _edge_label(G, u, d["features"]),
                   color="#1f3b73", alpha=0.75, lw=1.2, linestyle=EDGE_STYLE[d["kind"]])

    handles = [Line2D([], [], color="#ff9f1c", marker="s", ls="", ms=10, mfc="#fde3c4", label="root node"),
               Line2D([], [], color="#1f3b73", lw=1.5, ls="-", label="physical effect on the room"),
               Line2D([], [], color="#1f3b73", lw=1.5, ls="--", label="control / set-point curve"),
               Line2D([], [], color="#1f3b73", lw=1.5, ls=":", label="proxy for unmeasured drivers")]
    if lag_of:
        handles.append(Line2D([], [], color="#1f3b73", marker="s", ls="", ms=10, mfc="white", mew=0.8,
                              label="lag node: earlier values of the node above it"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=5, fontsize=8,
              frameon=False, title="edge label: lags / moving average of the parent (none = same interval)",
              title_fontsize=8)
    ax.set_title("Graphical Causal Model (30-min data)", loc="left")
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
        ps = [f"{p} ({', '.join(d['features'])}) [{d['component'] or d['kind']}]"
              for p, _, d in G.in_edges(n, data=True)]
        print(f"  {n:22s} <- {', '.join(ps) or '-'}")
        for name, expr in G.nodes[n].get("interactions", {}).items():
            print(f"  {'':22s}    + {name} = {expr}")
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
