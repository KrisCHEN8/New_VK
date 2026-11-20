import copy
from collections import Counter
from dataclasses import dataclass, field
import holidays
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import GCM_generation as gcm


T = gcm.T
FREQ = gcm.CONFIG["freq"]
TZ = "Europe/Stockholm"
STEPS_PER_HOUR = pd.Timedelta("1h") / pd.Timedelta(FREQ)
BOUNDS = {"cooling_valve": (0, 100), "VAV_damper": (0, 100), "heating_valve": (0, 100)}

# Reference state
REFERENCE = {"dT_envelope": 0.0, "solar_radiation_lag1": 0.0, "solar_radiation_ma4_lag1": 0.0, "co2_lag1": 420.0,
             "hour_sin": 0.0, "hour_cos": -1.0, "workday": 0.0,
             "cooling_valve_lag1": 0.0, "VAV_damper_lag1": 0.0, "heating_valve_1h": 0.0}

# Interventions
SHIFTS = {
    "Outdoor +5 K": {"outdoor_temperature": lambda x: x + 5},
    "Solar +50 %": {"solar_radiation": lambda x: x * 1.5},
    "Occupancy +300 ppm CO2": {"co2": lambda x: x + 300},
    "Setpoint +1 K": {"setpoint": lambda x: x + 1},
    "Cooling valve +20 %-pt": {"cooling_valve": lambda x: x + 20},
    "Chilled-water supply +2 K": {"KB02_supply": lambda x: x + 2},
    "VAV damper +20 %-pt": {"VAV_damper": lambda x: x + 20},
    "AHU supply air +2 K": {"AHU_supply_temperature": lambda x: x + 2},
    "Radiator valve +20 %-pt": {"heating_valve": lambda x: x + 20},
    "Radiator supply +5 K": {"VS02_supply": lambda x: x + 5},
}


# Load data
def load_data(path="07-D2-02_data/dfs.pkl"):
    """ turn dict to one big dataframe """
    d = pd.read_pickle(path)
    df = pd.concat({k: v["value"] for k, v in d.items()}, axis=1).sort_index().asfreq(FREQ)
    df.index = df.index.tz_localize("UTC").tz_convert(TZ)
    df = df.interpolate(limit=4)          
    for c, (lo, hi) in BOUNDS.items():    
        df[c] = df[c].clip(lo, hi)
    hour = df.index.hour + df.index.minute / 60
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    hol = holidays.country_holidays(gcm.CONFIG["country"], years=sorted(set(df.index.year)))
    df["workday"] = ((df.index.dayofweek < 5) & ~pd.Index(df.index.date).isin(list(hol))).astype(float)
    return df


def split_blocks(index, block="7D", test_every=3, offset=1):
    """
    Time-block split: the index is cut into consecutive blocks (7 days) and every 'test_every'-th block
    is held out (True = test). The held-out data is spread over the whole period, so every season is represented.
    """
    start = index.min().floor("D")
    block_id = ((index - start) // pd.Timedelta(block)).astype(int)
    return np.asarray(block_id % test_every == offset)


def arrays(df, variables):
    """Series as arrays shaped (1, time): one scenario, shared by all scenarios of a simulation."""
    return {v: df[v].to_numpy(float)[None, :] for v in variables}


# Features: one function serves both the vectorised training path and the step-wise simulation
def _at(a, t, k):
    """Value k steps before t of every scenario; t=None gives the whole shifted series (NaN-padded)."""
    if t is not None:
        return a[:, t - k]
    out = np.full(a.shape, np.nan)
    out[:, k:] = a[:, : a.shape[1] - k] if k else a
    return out


def _window_mean(arrs, var, k, w, t):
    """Mean of the w values of 'var' ending k steps before t."""
    total = _at(arrs[var], t, k)
    for j in range(k + 1, k + w):
        total = total + _at(arrs[var], t, j)
    return total / w


def design(mech, arrs, t=None):
    """
    Design matrix of a mechanism from arrays shaped (scenarios, time): every step of scenario 0 when t is
    None (rows = time), else step t of every scenario (rows = scenarios).
    """
    env = {c: _window_mean(arrs, *spec, t) for c, spec in mech.inputs.items()}
    for name, code in mech.interactions.items():
        env[name] = eval(code, {"__builtins__": {}}, env)
    cols = np.broadcast_arrays(*[np.atleast_1d(env[mech.source.get(c, c)]) for c in mech.columns])
    X = np.stack(cols, axis=-1)
    return X[0] if t is None else X


# Mechanisms
@dataclass
class Mechanism:
    node: str
    inputs: dict                    # input column -> (variable, k, w): mean of w values ending k steps back
    interactions: dict              # derived column -> compiled expression of the inputs
    columns: list                   # design-matrix columns
    delta: bool = False             # target is x_t - x_{t-1}
    groups: dict = None             # additive group -> columns (room)
    context: dict = field(default_factory=dict)   # group -> columns that modulate it but are not the cause
    source: dict = field(default_factory=dict)    # context copy -> column it copies
    monotone: dict = field(default_factory=dict)
    model: object = None
    noise_std: float = None
    fit_rows: np.ndarray = None
    trimmed_days: list = None

    def make_model(self):
        kw = dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                  l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                  n_iter_no_change=40, random_state=0,
                  monotonic_cst=[self.monotone.get(c, 0) for c in self.columns])
        if self.groups:
            kw["interaction_cst"] = [[self.columns.index(c) for c in g] for g in self.groups.values()]
        return HistGradientBoostingRegressor(**kw)

    def predict(self, X, prev=None):
        out = self.model.predict(X)
        return prev + out if self.delta else out


def bound(var, x):
    lo, hi = BOUNDS.get(var, (-np.inf, np.inf))
    return np.clip(x, lo, hi)


def _additive_groups(groups):
    """
    Disjoint additive groups. A column listed in several groups (the calendar) is a cause only in the
    groups made of such shared columns (schedule); in every other group it enters as context, a copy
    'col@group' that modulates that group's effect (e.g. the beam's effect by time of day). Each group
    then has its own columns and the model is exactly additive across groups.
    """
    count = Counter(c for cols in groups.values() for c in cols)
    shared = {c for c, n in count.items() if n > 1}
    out, context, source = {}, {}, {}
    for g, cols in groups.items():
        if set(cols) <= shared:
            out[g] = list(cols)
            continue
        ctx = [f"{c}@{g}" for c in cols if c in shared]
        source.update({f"{c}@{g}": c for c in cols if c in shared})
        out[g] = [c for c in cols if c not in shared] + ctx
        context[g] = ctx
    return out, context, source


def build_mechanisms(G):
    """One mechanism per non-root node of the graph, in topological order (same-interval parents first)."""
    mechs = {}
    for node in nx.topological_sort(G):
        if G.in_degree(node) == 0:
            continue
        d = G.nodes[node]
        inputs = {}
        for u, _, feats in G.in_edges(node, data="features"):
            var = G.nodes[u].get("lag_of", u)
            for f in feats:
                if var == gcm.CAL:
                    inputs[f] = (f, 0, 1)
                else:
                    w, k = gcm._window(var, f)
                    inputs[f] = (var, k, w)
        columns = [c for c in inputs if c not in d["interaction_inputs"]] + list(d["interactions"])
        groups, context, source = _additive_groups(d["groups"]) if d["groups"] else (None, {}, {})
        if groups:
            columns = list(dict.fromkeys(c for cols in groups.values() for c in cols))
        mechs[node] = Mechanism(
            node, inputs=inputs, columns=columns, delta=d["delta"], groups=groups, context=context,
            source=source, monotone=d["monotone"],
            interactions={n: compile(e, n, "eval") for n, e in d["interactions"].items()})
    return mechs


def history(mechs):
    """Steps of history every mechanism needs before the first simulated step."""
    return max(max(k + w - 1 for _, k, w in m.inputs.values()) for m in mechs.values())


def node_data(mech, arrs):
    """Design matrix, target (change for delta nodes) and previous value of the node at every step."""
    X = design(mech, arrs)
    x = arrs[mech.node][0]
    prev = _at(arrs[mech.node], None, 1)[0]
    y = x - prev if mech.delta else x
    return X, y, prev


def fit_mechanisms(G, df, train, trim_quantile=0.97):
    """
    Fit every mechanism on the training rows, then refit once without the days whose mean absolute
    residual is in the top (1 - trim_quantile): a robust substitute for fault labels.
    """
    mechs = build_mechanisms(G)
    arrs = arrays(df, variables(mechs))
    days = df.index.normalize()
    for mech in mechs.values():
        X, y, _ = node_data(mech, arrs)
        ok = train & np.isfinite(y)
        mech.model = mech.make_model().fit(X[ok], y[ok])
        r = pd.Series(np.abs(y - mech.model.predict(X)), index=df.index)[ok]
        score = r.groupby(days[ok]).mean()
        keep_days = score.index[score <= score.quantile(trim_quantile)]
        mech.fit_rows = ok & days.isin(keep_days)
        mech.model = mech.make_model().fit(X[mech.fit_rows], y[mech.fit_rows])
        mech.noise_std = float(np.std(y[mech.fit_rows] - mech.model.predict(X[mech.fit_rows])))
        mech.trimmed_days = sorted(set(score.index) - set(keep_days))
    return mechs


def variables(mechs):
    """Every series the mechanisms read or produce."""
    return list(dict.fromkeys([v for m in mechs.values() for v, _, _ in m.inputs.values()] + list(mechs)))


def mechanism_quality(mechs, arrs, test):
    """One-step fit of every mechanism on the held-out blocks."""
    rows = []
    for node, mech in mechs.items():
        X, y, _ = node_data(mech, arrs)
        ok = test & np.isfinite(y)
        pred = mech.model.predict(X[ok])
        rows.append({"node": node, "target": "change" if mech.delta else "level",
                     "r2": r2_score(y[ok], pred), "rmse": mean_squared_error(y[ok], pred) ** 0.5,
                     "noise_std": mech.noise_std, "n_train": int(mech.fit_rows.sum()), "n_test": int(ok.sum()),
                     "trimmed_days": len(mech.trimmed_days)})
    return pd.DataFrame(rows)


def abduct(mechs, arrs):
    """Noise of every node at every step: observed value minus mechanism prediction."""
    u = {}
    for node, mech in mechs.items():
        X, _, prev = node_data(mech, arrs)
        u[node] = arrs[node][0] - mech.predict(X, prev)
    return u


# Simulation
def simulate(mechs, arrs, noise, t0, t1, active=1.0, interventions=None, nodes=None):
    """
    Simulation of steps t0..t1-1 for S scenarios. Before t0 every variable keeps its observed value.
    arrs           arrays (1, time) shared by all scenarios, or (S, time) with one row per scenario
    noise          abducted noise per node, (time,) or (S, time)
    active         weight on each node's noise, scalar or (S, len(nodes)): 1 = actual behaviour, 0 = expected
    interventions  {variable: f(x) -> x} applied to the variable over t0..t1-1 (after its mechanism if simulated)
    nodes          mechanisms that are simulated (default all: closed loop); every other variable keeps its
                   observed value (nodes=[T]: open loop)
    Returns the simulated arrays (S, t1) of the simulated nodes.
    """
    nodes = [n for n in mechs if nodes is None or n in nodes]
    interventions = interventions or {}
    active = np.atleast_2d(np.asarray(active, float))
    S = max([active.shape[0]] + [a.shape[0] for a in arrs.values()])
    active = np.broadcast_to(active, (S, len(nodes)))
    sim = {}
    for v, a in arrs.items():
        a = a[:, :t1]
        sim[v] = np.array(np.broadcast_to(a, (S, a.shape[1]))) if v in nodes or v in interventions else a
    for v, f in interventions.items():
        if v not in nodes:
            sim[v][:, t0:t1] = bound(v, f(sim[v][:, t0:t1]))
    for t in range(t0, t1):
        for j, node in enumerate(nodes):
            mech = mechs[node]
            x = mech.predict(design(mech, sim, t), sim[node][:, t - 1])
            u = noise[node]
            x = x + active[:, j] * np.nan_to_num(u[:, t] if u.ndim == 2 else u[t])
            if node in interventions:
                x = interventions[node](x)
            sim[node][:, t] = bound(node, x)
    return {v: sim[v] for v in nodes}


def stack_windows(arrs, noise, starts, hist, horizon):
    """Per-scenario windows [start - hist, start + horizon) of every array and noise series."""
    idx = np.asarray(starts)[:, None] + np.arange(-hist, horizon)[None, :]
    return {v: a[0][idx] for v, a in arrs.items()}, {n: u[idx] for n, u in noise.items()}


# ---------------------------------------------------------------------------
# Quantification
# ---------------------------------------------------------------------------
def group_contributions(mech, X, reference=None):
    """
    Contribution of every additive group to the room's temperature change (K per interval) at every row:
    the group at its actual values minus the group at its reference state, with the group's context
    columns at their actual values in both. Exact because the room mechanism is additive across groups
    (interaction constraints). 'Constant gain' = prediction minus the sum of the contributions.
    """
    ref_of = {**REFERENCE, **(reference or {})}
    ref = np.array([ref_of.get(mech.source.get(c, c), np.nanmedian(X[:, i])) for i, c in enumerate(mech.columns)])
    out = {}
    for g, members in mech.groups.items():
        ctx = [mech.columns.index(c) for c in mech.context.get(g, [])]
        idx = [mech.columns.index(c) for c in members]
        X_ref = np.tile(ref, (len(X), 1))
        X_ref[:, ctx] = X[:, ctx]
        X_act = X_ref.copy()
        X_act[:, idx] = X[:, idx]
        out[g] = mech.model.predict(X_act) - mech.model.predict(X_ref)
    out = pd.DataFrame(out)
    out["Constant gain"] = mech.model.predict(X) - out.sum(axis=1).to_numpy()
    return out


def group_strengths(C, y, regimes, min_rows=500):
    """
    Causal strength of every group: variance of its contribution divided by the variance of the room's
    change, overall and per regime (fraction of the room's variability that the group drives).
    """
    groups = [g for g in C.columns if g != "Constant gain"]
    regimes = np.asarray(regimes)
    out = {"all": C[groups].var() / np.var(y)}
    for r in pd.unique(regimes):
        sel = regimes == r
        if sel.sum() >= min_rows:
            out[r] = C.loc[sel, groups].var() / np.var(y[sel])
    return pd.DataFrame(out)


def bootstrap_strengths(mech, X, y, days, rows, regimes, n_boot=20, seed=0):
    """
    Day-block bootstrap of the group strengths: the room mechanism is refitted on training days drawn with
    replacement (as sample weights) and the strengths are recomputed on the evaluation rows.
    Returns mean, 2.5 % and 97.5 % quantiles per group and regime.
    """
    rng = np.random.default_rng(seed)
    train_days = np.unique(days[mech.fit_rows])
    reps = []
    for b in range(n_boot):
        counts = pd.Series(rng.choice(train_days, len(train_days), replace=True)).value_counts()
        weight = pd.Series(days).map(counts).fillna(0).to_numpy()
        sel = mech.fit_rows & (weight > 0)
        mb = copy.copy(mech)
        mb.model = mb.make_model().fit(X[sel], y[sel], sample_weight=weight[sel])
        reps.append(group_strengths(group_contributions(mb, X[rows]), y[rows], regimes[rows]))
    S = pd.concat(reps, keys=range(n_boot))
    by = S.groupby(level=1, sort=False)
    return pd.concat({"mean": by.mean(), "lo": by.quantile(0.025), "hi": by.quantile(0.975)}, axis=1)


def heat_balance(C, regimes, workday):
    """Mean contribution of every group (K per hour) by regime, day type and local time of day."""
    hour = C.index.hour + C.index.minute / 60
    key = [np.asarray(regimes), np.asarray(workday), hour]
    return (C * STEPS_PER_HOUR).groupby(key).mean().rename_axis(["regime", "workday", "hour"])


def intervention_effects(mechs, arrs, noise, starts, interventions, horizon=48, closed_loop=True):
    """
    Effect of shift interventions on the room temperature, by counterfactual simulation from every start
    over 'horizon' steps. Both runs keep the abducted noise of every node (the day as it happened), so the
    difference is the effect of the intervention on that day. Open loop: only the room is simulated, the
    controllers and supply temperatures keep their observed values; closed loop: they respond as well.
    Returns {name: array (len(starts), horizon)} of room-temperature differences (K).
    """
    hist = history(mechs)
    A, U = stack_windows(arrs, noise, starts, hist, horizon)
    nodes = None if closed_loop else [T]
    run = lambda iv: simulate(mechs, A, U, hist, hist + horizon, 1.0, iv, nodes)[T][:, hist:]
    base = run(None)
    return {name: run(iv) - base for name, iv in interventions.items()}


def summarize_effects(effects, regimes):
    """Mean effect over the horizon per regime of the start day (K), and the mean response curve per hour."""
    rows, curves = [], {}
    for (name, loop), e in effects.items():
        for r in ["all"] + list(pd.unique(regimes)):
            sel = np.ones(len(e), bool) if r == "all" else regimes == r
            rows.append({"intervention": name, "loop": loop, "regime": r, "n_days": int(sel.sum()),
                         "mean_effect_K": e[sel].mean(), "end_effect_K": e[sel, -1].mean()})
        curves[(name, loop)] = e.mean(0)
    table = pd.DataFrame(rows).pivot_table(index=["intervention", "regime"], columns="loop",
                                           values=["mean_effect_K", "end_effect_K"], sort=False)
    curves = pd.DataFrame(curves)
    curves.index = (curves.index + 1) / STEPS_PER_HOUR
    curves.index.name = "hours_after_start"
    return table, curves


# Workflow
def run_quantification(df, G, shifts=SHIFTS, n_boot=20, horizon=48):
    """
    Steps 1-4 end to end. Returns a dict of tables:
      quality       held-out one-step fit of every mechanism
      strengths     group strengths of the room per regime, with day-block bootstrap intervals
      heat_balance  mean group contributions (K per hour) by regime, day type and time of day
      effects       mean effect of every shift on the room temperature, open vs closed loop, per regime
      curves        mean response of the room temperature after the shift starts (local midnight)
    """
    test = split_blocks(df.index)
    mechs = fit_mechanisms(G, df, ~test)
    arrs = arrays(df, variables(mechs))
    noise = abduct(mechs, arrs)
    regimes = gcm.label_regimes(df).to_numpy()

    room = mechs[T]
    X, y, _ = node_data(room, arrs)
    ok = np.isfinite(y)
    C = group_contributions(room, X[ok]).set_axis(df.index[ok])
    held = test & ok
    strengths = pd.concat(
        {"point": group_strengths(group_contributions(room, X[held]), y[held], regimes[held]).stack()}, axis=1)
    if n_boot:
        boot = bootstrap_strengths(room, X, y, df.index.normalize(), held, regimes, n_boot)
        strengths = strengths.join(boot.stack(future_stack=True))
    strengths = strengths.rename_axis(["group", "regime"]).sort_index(level="regime", sort_remaining=False)

    hist, n = history(mechs), len(df)
    pos = np.flatnonzero((df.index.hour == 0) & (df.index.minute == 0))
    starts = pos[(pos >= hist) & (pos + horizon <= n)]
    effects = {(name, loop): e
               for loop, closed in [("open", False), ("closed", True)]
               for name, e in intervention_effects(mechs, arrs, noise, starts, shifts, horizon, closed).items()}
    effect_table, curves = summarize_effects(effects, regimes[starts])

    return {"quality": mechanism_quality(mechs, arrs, test), "strengths": strengths,
            "heat_balance": heat_balance(C, regimes[ok], df["workday"].to_numpy()[ok]),
            "effects": effect_table, "curves": curves}


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    G = gcm.build_gcm({"make_plots": False})
    df = load_data()
    res = run_quantification(df, G)
    names = {"quality": "mechanism_quality", "strengths": "causal_strengths", "heat_balance": "heat_balance",
             "effects": "intervention_effects", "curves": "effect_curves"}
    for key, table in res.items():
        table.to_csv(f"07-D2-02_data/{names[key]}.csv")
    print(res["quality"].round(3).to_string(index=False), "\n")
    print(res["strengths"].round(3).to_string(), "\n")
    print(res["effects"].round(3).to_string())
