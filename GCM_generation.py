from pathlib import Path
import re
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import holidays
import rdflib


T = "indoor_temperature"
CAL = "calendar"
CAL_FEATURES = ["hour_sin", "hour_cos", "workday"]

CONTROLLERS = ["cooling_valve", "heating_valve", "VAV_damper"]
ERR_WINDOWS = [2, 4, 8, 16]                                             # 1, 2, 4 and 8 hours
ROOM_NOW = [T] + [f"{T}_ma{w}" for w in ERR_WINDOWS]                    # room now, and its recent means
SETPOINT_NOW = ["setpoint"] + [f"setpoint_ma{w}" for w in ERR_WINDOWS]
SEASON = ["outdoor_temperature_ma48"]                                   # 24-hour mean outdoor temperature

# parent, child, feature column names
EDGES = [
    # set-point curves
    ("outdoor_temperature", "VS02_supply", ["outdoor_temperature"] + SEASON),     # VS02 heating curve
    (CAL, "VS02_supply", CAL_FEATURES),                                             # night setback
    ("outdoor_temperature", "AHU_supply_temperature", ["outdoor_temperature"] + SEASON),     # AHU setpoint curve
    (CAL, "AHU_supply_temperature", CAL_FEATURES),
    ("VAV_damper", "AHU_supply_temperature", ["VAV_damper"]),       # sensor in the room's duct drifts at low air flow
    ("relative_humidity", "KB02_supply", ["relative_humidity_lag1"]),       # chilled water setpoint to avoid condensation
    ("outdoor_temperature", "KB02_supply", ["outdoor_temperature"] + SEASON),
    (CAL, "KB02_supply", CAL_FEATURES),
    ("cooling_valve", "KB02_supply", ["cooling_valve", "cooling_valve_ma4", "cooling_valve_ma16"]),  # closed valve: stagnant water warms at the sensor

    # room controllers react within the interval to the error against the scheduled setpoint,
    # with memory (errors averaged over 1-8 hours), and change with schedule and season
    (T, "cooling_valve", ROOM_NOW),
    ("setpoint", "cooling_valve", SETPOINT_NOW),
    (CAL, "cooling_valve", CAL_FEATURES),
    ("outdoor_temperature", "cooling_valve", SEASON),
    ("relative_humidity", "cooling_valve", ["relative_humidity"]),   # condensation limit
    (T, "heating_valve", ROOM_NOW),
    ("setpoint", "heating_valve", SETPOINT_NOW),
    (CAL, "heating_valve", CAL_FEATURES),
    ("outdoor_temperature", "heating_valve", SEASON),
    (T, "VAV_damper", ROOM_NOW),
    ("setpoint", "VAV_damper", SETPOINT_NOW),
    (CAL, "VAV_damper", CAL_FEATURES),
    ("outdoor_temperature", "VAV_damper", SEASON),
    ("co2", "VAV_damper", ["co2"]),    # demand-controlled ventilation

    # # hydraulics, same interval
    # ("cooling_valve", "cooling_flow", ["cooling_valve"]),
    # ("heating_valve", "heating_flow", ["heating_valve"]),

    # physical effects on the room (delayed: every cause is at t-1 or earlier)
    (T, T, [f"{T}_lag1"]),                                          # thermal inertia, exchange with the building mass
    ("cooling_valve", T, ["cooling_valve_lag1"]),                           # chilled beam, faster
    ("KB02_supply", T, ["KB02_supply_lag1"]),
    ("heating_valve", T, ["heating_valve_lag1", "heating_valve_lag2"]),     # radiator, slower
    ("VS02_supply", T, ["VS02_supply_lag1", "VS02_supply_lag2"]),
    ("VAV_damper", T, ["VAV_damper_lag1"]),
    ("AHU_supply_temperature", T, ["AHU_supply_temperature_lag1"]),
    ("solar_radiation", T, ["solar_radiation_lag1", "solar_radiation_ma4_lag1"]),   # last 30 min, and 2-hour mean up to t-1
    ("outdoor_temperature", T, ["outdoor_temperature_ma6_lag1"]),           # 3-hour mean up to t-1

    # proxies for unmeasured drivers of the room temperature
    ("co2", T, ["co2_lag1"]),      # occupants
    (CAL, T, CAL_FEATURES),         # scheduled internal gains
]


# Components: the variables that act on the room together through one emitter.
COMPONENTS = {
    "chilled_beam": ["cooling_valve", "KB02_supply", "VAV_damper", "AHU_supply_temperature"],
    "radiator": ["heating_valve", "VS02_supply"],
}

# Derived terms, computed from the feature columns
ERR_TERMS = {f"err_{w // 2}h": f"{T}_ma{w} - setpoint_ma{w}" for w in ERR_WINDOWS}   # room minus setpoint
INTERACTIONS = {
    T: {
        # temperature differences that drive the heat flows
        "dT_envelope": "outdoor_temperature_ma6_lag1 - indoor_temperature_lag1",
        "dT_chilled_water": "indoor_temperature_lag1 - KB02_supply_lag1",           # chilled beam, water side
        "dT_supply_air": "indoor_temperature_lag1 - AHU_supply_temperature_lag1",   # chilled beam, air side
        "dT_radiator": "(VS02_supply_lag1 + VS02_supply_lag2) / 2 - indoor_temperature_lag1",
        "heating_valve_1h": "(heating_valve_lag1 + heating_valve_lag2) / 2",
    },
    **{c: ERR_TERMS for c in CONTROLLERS},
}

# Feature columns that enter the mechanism only through its derived terms
INTERACTION_INPUTS = {
    T: ["outdoor_temperature_ma6_lag1", "KB02_supply_lag1", "AHU_supply_temperature_lag1",
        "heating_valve_lag1", "heating_valve_lag2", "VS02_supply_lag1", "VS02_supply_lag2"],
    **{c: ROOM_NOW[1:] + SETPOINT_NOW[1:] for c in CONTROLLERS},
}

GROUPS = {
    T: {
        "envelope": ["dT_envelope"],
        "solar": ["solar_radiation_lag1", "solar_radiation_ma4_lag1"],
        "occupancy": ["co2_lag1"],
        "schedule": CAL_FEATURES,
        "chilled_beam": ["cooling_valve_lag1", "dT_chilled_water", "VAV_damper_lag1", "dT_supply_air", *CAL_FEATURES],
        "radiator": ["heating_valve_1h", "dT_radiator"],
        "building_mass": [f"{T}_lag1", *CAL_FEATURES],
    },
}

# Monotonic effects
MONOTONE = {
    T: {
        "dT_envelope": +1, "solar_radiation_lag1": +1, "solar_radiation_ma4_lag1": +1, "co2_lag1": +1,
        "cooling_valve_lag1": -1, "dT_chilled_water": -1, "dT_supply_air": -1,
        "heating_valve_1h": +1, "dT_radiator": +1,
        f"{T}_lag1": -1,      # the target is the change: a warmer room loses more heat to the building
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
    "interaction_inputs": INTERACTION_INPUTS,
    "groups": GROUPS,
    "monotone": MONOTONE,
    "excluded": EXCLUDED,
    "delta_nodes": [T],                   # mechanisms that predict the change over one interval
    "lag_node_sources": [T, "cooling_valve", "heating_valve", "VAV_damper",
                         "KB02_supply", "AHU_supply_temperature"],   # variables in a loop with the target
    "proxy_nodes": ["co2", CAL],          # edges from these into the target stand for unmeasured drivers
    "calendar_node": CAL,
    "calendar_features": CAL_FEATURES,
    "display_names": {CAL: "calendar\n(hour, workday)"},
    "freq": "30min",
    "country": "SE",  # Filter out the Swedish red days
    "regime_valves": {"heating": "heating_valve", "cooling": "cooling_valve"},
    "valve_active_threshold": 5.0,   # daily mean opening above which an emitter counts as active (% open)
    "figure_name": "causal_dag.png",
    "make_plots": True,
}


# Helpers
def _merge_config(config=None):
    """CONFIG, then the user overrides in 'config'."""
    return {**CONFIG, **(config or {})}


def _window(var, feat):
    """(w, k) of the feature column 'feat' of 'var': the mean of w values, the latest k intervals ago."""
    if feat == var:
        return 1, 0
    m = re.fullmatch(rf"{re.escape(var)}_(?:lag(\d+)|ma(\d+)(?:_lag(\d+))?)", feat)
    if not m:
        raise ValueError(f"feature '{feat}' of '{var}' must be '{var}', '{var}_lag<k>', '{var}_ma<w>' "
                         f"or '{var}_ma<w>_lag<k>'")
    lag, w, k = m.groups()
    return (1, int(lag)) if lag else (int(w), int(k or 0))


def _feature_lags(var, feat, cfg):
    """Time lags (in intervals) that the feature column 'feat' of the variable 'var' covers."""
    if var == cfg["calendar_node"]:
        if feat not in cfg["calendar_features"]:
            raise ValueError(f"calendar feature '{feat}' is not in calendar_features")
        return [0]
    w, k = _window(var, feat)
    return list(range(k, k + w))


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
        used = set()
        for name, expr in inter.items():
            names = set(re.findall(r"[A-Za-z_]\w*", expr))
            missing = names - set(inputs)
            if missing:
                raise ValueError(f"interaction '{name}' of {n} uses {sorted(missing)}, which are not inputs of {n}")
            used |= names
        hidden = list(cfg["interaction_inputs"].get(n, []))
        unused = set(hidden) - used
        if unused:
            raise ValueError(f"interaction_inputs of {n} list {sorted(unused)}, which no interaction of {n} uses")
        columns = [c for c in inputs if c not in hidden] + list(inter)
        mono = dict(cfg["monotone"].get(n, {}))
        unknown = set(mono) - set(columns)
        if unknown:
            raise ValueError(f"monotone constraint on {sorted(unknown)}, which are not columns of {n}'s mechanism")
        groups = {g: list(cols) for g, cols in cfg["groups"].get(n, {}).items()}
        grouped = {c for cols in groups.values() for c in cols}
        if groups and grouped != set(columns):
            raise ValueError(f"groups of {n}: unknown columns {sorted(grouped - set(columns))}, "
                             f"ungrouped columns {sorted(set(columns) - grouped)}")
        delta = n in cfg["delta_nodes"]
        if delta and f"{n}_lag1" not in inputs:
            raise ValueError(f"{n} predicts its change, so '{n}_lag1' must be one of its inputs")
        G.nodes[n].update(interactions=inter, interaction_inputs=hidden, monotone=mono, groups=groups, delta=delta)
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
    Mechanism inputs plus the node's interaction terms, computed from 'inputs'; the columns that
    enter only through the interaction terms are dropped.
    """
    X = inputs[mechanism_inputs(G, node)].copy()
    for name, expr in G.nodes[node]["interactions"].items():
        X[name] = X.eval(expr)
    return X.drop(columns=G.nodes[node]["interaction_inputs"])


def monotone_constraints(G, node, columns):
    """Signs for the design-matrix columns."""
    return [G.nodes[node]["monotone"].get(c, 0) for c in columns]


def interaction_groups(G, node, columns):
    """Index sets of the design-matrix columns per additive group (interaction_cst); None if the node has none."""
    groups = G.nodes[node]["groups"]
    if not groups:
        return None
    idx = {c: i for i, c in enumerate(columns)}
    return [[idx[c] for c in cols] for cols in groups.values()]


def node_frame(G, node, feats):
    """
    Rows with complete data for one mechanism: target series y and design matrix X.
    For delta nodes y is the change over one interval, x_t - x_{t-1}.
    """
    cols = list(dict.fromkeys([node] + mechanism_inputs(G, node)))
    data = feats[cols].dropna()
    y = data[node] - data[f"{node}_lag1"] if G.nodes[node]["delta"] else data[node]
    return y, design_matrix(G, node, data)


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
            w, k = _window(var, f)
            s = df[var] if w == 1 else df[var].rolling(w, min_periods=w).mean()
            out[f] = s.shift(k)

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


def _pick_rad(ax, pos, u, v, candidates=(0.0, 0.1, -0.1, 0.2, -0.2)):
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
    parts = ["lag0" if f == name else f.removeprefix(name + "_") for f in features]
    mas = [q for q in parts if re.fullmatch(r"ma\d+", q)]
    if len(mas) > 1:   # several windows ending now: 'ma2/4/8/16'
        parts = [q for q in parts if q not in mas] + ["ma" + "/".join(q[2:] for q in mas)]
    return ", ".join(parts)


EDGE_STYLE = {"physical": "-", "control": "--", "proxy": ":"}


def plot_dag(G):
    """
    Layered DAG, roots on the left; edge labels = lags or moving averages of the parent (none = same interval).
    Lag nodes (dashed outline) sit below their node family. Line style = edge kind.
    """
    tgt = G.graph["target"]
    lag_of = nx.get_node_attributes(G, "lag_of")
    pos = _layout(G)
    fig, ax = plt.subplots(figsize=(16, 10), layout="constrained")
    xs, ys = zip(*pos.values())
    ax.set_xlim(min(xs) - 0.35, max(xs) + 0.35)
    ax.set_ylim(min(ys) - 1.3, max(ys) + 1.3)   # room for the arcs, and the legend below
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
              frameon=False, # title="edge label: lags / moving average of the parent (none = same interval)",
              title_fontsize=8)
    ax.set_title("Graphical Causal Model (30-min data)", loc="left")
    return fig


# export as .ttl and .txt files
GCM_NS = "https://vk.org/gcm#"

def _flat(attrs):
    """Node/edge attributes as plain strings and numbers (lists and dicts joined), for text formats."""
    out = {}
    for k, v in attrs.items():
        if isinstance(v, dict):
            v = "; ".join(f"{a}={b}" for a, b in v.items())
        elif isinstance(v, (list, tuple)):
            v = ", ".join(map(str, v))
        out[k] = "" if v is None else v if isinstance(v, (int, float, str)) else str(v)
    return out


def _to_rdf(G, ns=GCM_NS):
    """
    The DAG as an RDF graph. Each variable is a gcm:Variable (gcm:LagNode for lag nodes, with gcm:lagOf),
    each edge a direct triple 'parent gcm:causes child' plus a gcm:Edge resource holding its features,
    kind and component. Interaction terms are gcm:Interaction resources of their node, additive groups
    gcm:AdditiveGroup resources; a node that predicts its change has gcm:deltaTarget true.
    """
    from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
    GCM = Namespace(ns)
    uri = lambda name: URIRef(ns + re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_"))
    g = Graph()
    g.bind("gcm", GCM)

    g.add((GCM.Model, RDF.type, GCM.CausalModel))
    g.add((GCM.Model, GCM.target, uri(G.graph["target"])))
    for n, d in G.nodes(data=True):
        u = uri(n)
        g.add((u, RDF.type, GCM.LagNode if "lag_of" in d else GCM.Variable))
        g.add((u, RDFS.label, Literal(n)))
        g.add((u, GCM.isRoot, Literal(G.in_degree(n) == 0)))
        g.add((u, GCM.layer, Literal(d.get("layer", 0))))
        if "lag_of" in d:
            g.add((u, GCM.lagOf, uri(d["lag_of"])))
        for name, expr in d.get("interactions", {}).items():
            i = uri(f"{n}__{name}")
            g.add((i, RDF.type, GCM.Interaction))
            g.add((i, RDFS.label, Literal(name)))
            g.add((i, GCM.expression, Literal(expr)))
            g.add((i, GCM.monotone, Literal(d.get("monotone", {}).get(name, 0))))
            g.add((u, GCM.hasInteraction, i))
        for name, cols in d.get("groups", {}).items():
            a = uri(f"{n}__group__{name}")
            g.add((a, RDF.type, GCM.AdditiveGroup))
            g.add((a, RDFS.label, Literal(name)))
            for c in cols:
                g.add((a, GCM.column, Literal(c)))
            g.add((u, GCM.hasGroup, a))
        if d.get("delta"):
            g.add((u, GCM.deltaTarget, Literal(True)))
    for p, c, d in G.edges(data=True):
        g.add((uri(p), GCM.causes, uri(c)))
        e = uri(f"{p}__to__{c}")
        g.add((e, RDF.type, GCM.Edge))
        g.add((e, GCM.parent, uri(p)))
        g.add((e, GCM.child, uri(c)))
        g.add((e, GCM.kind, Literal(d["kind"])))
        if d.get("component"):
            g.add((e, GCM.component, Literal(d["component"])))
        for f in d["features"]:
            g.add((e, GCM.feature, Literal(f)))
    return g


def export_dag(G, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext in {".ttl", ".rdf", ".nt"}:
        fmt = {".ttl": "turtle", ".rdf": "xml", ".nt": "nt"}[ext]
        _to_rdf(G).serialize(destination=path, format=fmt)
    elif ext in {".txt", ".csv"}:
        rows = [{"parent": p, "child": c, **_flat({k: d.get(k) for k in ("features", "kind", "component")})}
                for p, c, d in G.edges(data=True)]
        pd.DataFrame(rows).to_csv(path, sep="\t" if ext == ".txt" else ",", index=False)
    elif ext == ".dot":
        q = lambda s: '"' + str(s).replace('"', r'\"').replace("\n", r"\n") + '"'
        lines = ["digraph GCM {", "  rankdir=LR;"]
        for n, d in G.nodes(data=True):
            style = "dashed" if "lag_of" in d else "solid"
            lines.append(f"  {q(n)} [label={q(d.get('label', n))}, style={style}];")
        for p, c, d in G.edges(data=True):
            lines.append(f"  {q(p)} -> {q(c)} [label={q(', '.join(d['features']))}, kind={q(d['kind'])}];")
        path.write_text("\n".join(lines + ["}"]) + "\n")
    elif ext in {".graphml", ".gml", ".json"}:
        H = nx.DiGraph(**_flat(G.graph))
        H.add_nodes_from((n, _flat(d)) for n, d in G.nodes(data=True))
        H.add_edges_from((p, c, _flat(d)) for p, c, d in G.edges(data=True))
        if ext == ".json":
            import json
            path.write_text(json.dumps(nx.node_link_data(H, edges="edges"), indent=2))
        else:
            (nx.write_graphml if ext == ".graphml" else nx.write_gml)(H, path)
    else:
        raise ValueError(f"unknown extension '{ext}'")
    print(f"DAG -> {path}")
    return path


# main workflow
def run_gcm(config=None):
    """Build the causal DAG, print it and save the figure."""
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
    export_dag(G, "07-D2-02_data/causal_dag.ttl")
    export_dag(G, "07-D2-02_data/causal_dag.txt")
