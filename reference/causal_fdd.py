"""
Causal fault detection and diagnosis of indoor temperature from operational data.

A time-unrolled structural causal model (SCM) of one room: every endogenous variable has a learned
mechanism x_t = f(parents) + u_t with additive noise u_t. The SCM is used to

  1. quantify causal effects (component-wise heat balance of the room, effect curves, strengths,
     open- vs closed-loop responses),
  2. detect temperature anomalies as deviations from the closed-loop behaviour the SCM expects
     under the observed weather, occupancy and schedule,
  3. attribute each anomaly to root causes with exact Shapley values over the nodes' noise terms
     (counterfactual simulation: a node's "abnormal part" is switched on or off),
  4. validate the diagnosis by injecting faults into the closed loop.

No fault labels are needed: mechanisms are fitted on operational data with a robust trimming step.
"""

from dataclasses import dataclass, field
from itertools import combinations
from math import factorial

import holidays
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

TZ = "Europe/Stockholm"
T = "indoor_temperature"
EXO = ["outdoor_temperature", "solar_radiation", "co2", "relative_humidity", "setpoint",
       "hour_sin", "hour_cos", "workday"]
ENDO = [T, "cooling_valve", "VAV_damper", "heating_valve", "KB02_supply", "AHU_supply_temperature", "VS02_supply"]
BOUNDS = {"cooling_valve": (0, 100), "VAV_damper": (0, 100), "heating_valve": (0, 100)}
LABELS = {T: "Room (unmeasured gains)", "cooling_valve": "Cooling valve", "VAV_damper": "VAV damper",
          "heating_valve": "Radiator valve", "KB02_supply": "Chilled-water supply",
          "AHU_supply_temperature": "AHU supply air", "VS02_supply": "Radiator supply"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_data(path):
    """The pickle of 30-min series as one frame, index converted from UTC to local time."""
    d = pd.read_pickle(path)
    df = pd.concat({k: v["value"].rename(k) for k, v in d.items()}, axis=1)
    df = df.rename(columns={"humidity": "relative_humidity"})
    df.index = df.index.tz_localize("UTC").tz_convert(TZ)
    df = df.interpolate(limit=4)      # a handful of single missing values
    for c, (lo, hi) in BOUNDS.items():  # recorded openings slightly outside 0-100 (e.g. 100.00000000000001) would
        df[c] = df[c].clip(lo, hi)      # make a simulation that clips differ from the data at tree splits on 100
    hour = df.index.hour + df.index.minute / 60
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    hol = holidays.country_holidays("SE", years=sorted(set(df.index.year)))
    df["workday"] = ((df.index.dayofweek < 5) & ~pd.Index(df.index.date).isin(list(hol))).astype(float)
    return df


def test_blocks(index, block_days=7, every=3, offset=1):
    """Every 'every'-th block of 'block_days' days is held out (True = test)."""
    day = (index.normalize() - index.normalize()[0]).days
    return np.asarray((day // block_days) % every == offset)


# ---------------------------------------------------------------------------
# Features: one spec drives both the vectorised training path and the step-wise simulation
# ---------------------------------------------------------------------------
def _at(a, t, k):
    """Value k steps before t; t=None gives the whole shifted series (NaN-padded)."""
    if t is not None:
        return a[:, t - k]
    out = np.full(a.shape, np.nan)
    out[:, k:] = a[:, : a.shape[1] - k] if k else a
    return out


def _feature(arrs, spec, t):
    kind = spec[0]
    if kind == "lag":                       # ("lag", var, k)
        return _at(arrs[spec[1]], t, spec[2])
    if kind == "ma":                        # ("ma", var, k, w): mean of lags k .. k+w-1
        # summed in a fixed order so training and step-wise simulation give bit-identical values
        # (tree splits can sit exactly on a training value; a 1e-15 difference would change the branch)
        total = _at(arrs[spec[1]], t, spec[2]).copy()
        for j in range(spec[2] + 1, spec[2] + spec[3]):
            total = total + _at(arrs[spec[1]], t, j)
        return total / spec[3]
    if kind == "diff":                      # ("diff", spec_a, spec_b)
        return _feature(arrs, spec[1], t) - _feature(arrs, spec[2], t)
    raise ValueError(spec)


def feature_matrix(arrs, specs, t=None):
    """Columns of 'specs' from arrays shaped (scenarios, time). t=None: all time steps of scenario 0."""
    S = max(a.shape[0] for a in arrs.values()) if t is not None else 1
    cols = [np.broadcast_to(_feature(arrs, s, t), (S,) if t is not None else (1, next(iter(arrs.values())).shape[1]))
            for s in specs.values()]
    return np.column_stack(cols) if t is not None else np.column_stack([c[0] for c in cols])


lag = lambda v, k=1: ("lag", v, k)
ma = lambda v, k, w: ("ma", v, k, w)
diff = lambda a, b: ("diff", a, b)

CAL_T = {"hour_sin": lag("hour_sin", 0), "hour_cos": lag("hour_cos", 0), "workday": lag("workday", 0)}
ERR = {f"err_{w // 2}h": diff(ma(T, 0, w), ma("setpoint", 0, w)) for w in (2, 4, 8, 16)}
SEASON = {"outdoor_24h": ma("outdoor_temperature", 0, 48)}


# ---------------------------------------------------------------------------
# Mechanisms
# ---------------------------------------------------------------------------
@dataclass
class Mechanism:
    node: str
    specs: dict                               # column name -> feature spec
    delta: bool = False                       # target is x_t - x_{t-1}
    groups: list = None                       # additive groups of column names (room)
    monotone: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)   # group -> columns that modulate it but are not the cause
    model: object = None
    noise_std: float = None

    def make_model(self):
        cols = list(self.specs)
        kw = dict(max_iter=600, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                  l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
                  n_iter_no_change=40, random_state=0,
                  monotonic_cst=[self.monotone.get(c, 0) for c in cols])
        if self.groups:
            kw["interaction_cst"] = [[cols.index(c) for c in g] for g in self.groups.values()]
        return HistGradientBoostingRegressor(**kw)

    def predict(self, X, prev=None):
        out = self.model.predict(X)
        return prev + out if self.delta else out

    def bound(self, x):
        lo, hi = BOUNDS.get(self.node, (-np.inf, np.inf))
        return np.clip(x, lo, hi)


def build_mechanisms():
    """The causal graph: parents (with lags) of every endogenous node."""
    m = {}
    m[T] = Mechanism(T, delta=True, specs={
        # envelope: outdoor (3 h mean) minus room
        "dT_envelope": diff(ma("outdoor_temperature", 1, 6), lag(T)),
        # solar gains
        "solar_30min": lag("solar_radiation"), "solar_2h": ma("solar_radiation", 1, 4),
        # occupants and scheduled internal gains
        "co2": lag("co2"),
        "hour_sin": lag("hour_sin"), "hour_cos": lag("hour_cos"), "workday": lag("workday"),
        # chilled beam: water side (valve, room minus chilled-water supply) and primary air (damper, room minus AHU air)
        "cooling_valve": lag("cooling_valve"), "dT_chilled_water": diff(lag(T), lag("KB02_supply")),
        "VAV_damper": lag("VAV_damper"), "dT_supply_air": diff(lag(T), lag("AHU_supply_temperature")),
        # radiator
        "heating_valve": ma("heating_valve", 1, 2), "dT_radiator": diff(ma("VS02_supply", 1, 2), lag(T)),
        # heat exchange with the building (neighbours, structure)
        "room_lag1": lag(T),
        # schedule context: the beam's and the building's effect vary with time of day and day type
        "beam_hour_sin": lag("hour_sin"), "beam_hour_cos": lag("hour_cos"), "beam_workday": lag("workday"),
        "mass_hour_sin": lag("hour_sin"), "mass_hour_cos": lag("hour_cos"), "mass_workday": lag("workday"),
    }, groups={
        "Envelope (outdoor)": ["dT_envelope"],
        "Solar": ["solar_30min", "solar_2h"],
        "Occupancy (CO2)": ["co2"],
        "Schedule (clock)": ["hour_sin", "hour_cos", "workday"],
        "Chilled beam": ["cooling_valve", "dT_chilled_water", "VAV_damper", "dT_supply_air",
                         "beam_hour_sin", "beam_hour_cos", "beam_workday"],
        "Radiator": ["heating_valve", "dT_radiator"],
        "Building mass": ["room_lag1", "mass_hour_sin", "mass_hour_cos", "mass_workday"],
    }, context={"Chilled beam": ["beam_hour_sin", "beam_hour_cos", "beam_workday"],
                "Building mass": ["mass_hour_sin", "mass_hour_cos", "mass_workday"]}, monotone={"co2": 1, "dT_envelope": 1, "solar_30min": 1, "solar_2h": 1, "cooling_valve": -1,
                 "dT_chilled_water": -1, "dT_supply_air": -1, "heating_valve": 1, "dT_radiator": 1,
                 "room_lag1": -1})
    ctrl = {**{"room": lag(T, 0)}, **ERR, "setpoint": lag("setpoint", 0), **CAL_T, **SEASON}
    m["cooling_valve"] = Mechanism("cooling_valve", specs={**ctrl, "rh": lag("relative_humidity", 0)})
    m["VAV_damper"] = Mechanism("VAV_damper", specs={**ctrl, "co2": lag("co2", 0)})
    m["heating_valve"] = Mechanism("heating_valve", specs=dict(ctrl))
    m["KB02_supply"] = Mechanism("KB02_supply", specs={
        "rh_lag1": lag("relative_humidity"), "outdoor": lag("outdoor_temperature", 0), **SEASON, **CAL_T,
        # no flow in the room's beam -> stagnant water warms up at the sensor
        "valve_now": lag("cooling_valve", 0), "valve_2h": ma("cooling_valve", 0, 4), "valve_8h": ma("cooling_valve", 0, 16)})
    m["AHU_supply_temperature"] = Mechanism("AHU_supply_temperature", specs={
        "outdoor": lag("outdoor_temperature", 0), **SEASON, **CAL_T, "damper": lag("VAV_damper", 0)})
    m["VS02_supply"] = Mechanism("VS02_supply", specs={
        "outdoor": lag("outdoor_temperature", 0), **SEASON, **CAL_T})
    return m


def arrays(df):
    return {c: df[c].to_numpy(float)[None, :] for c in EXO + ENDO}


def node_data(mech, arrs):
    X = feature_matrix(arrs, mech.specs)
    x = arrs[mech.node][0]
    prev = _at(arrs[mech.node], None, 1)[0]
    y = x - prev if mech.delta else x
    return X, y, prev


def fit_mechanisms(df, train, trim_quantile=0.97):
    """
    Fit every mechanism on the training rows, then refit once without the days whose mean absolute
    standardised residual is in the top (1 - trim_quantile): a robust substitute for fault labels.
    """
    mechs, arrs = build_mechanisms(), arrays(df)
    days = df.index.normalize()
    for mech in mechs.values():
        X, y, _ = node_data(mech, arrs)
        ok = train & np.isfinite(X).all(1) & np.isfinite(y)
        mech.model = mech.make_model().fit(X[ok], y[ok])
        r = pd.Series(np.abs(y - mech.model.predict(np.nan_to_num(X))), index=df.index)[ok]
        score = r.groupby(days[ok]).mean()
        keep_days = score.index[score <= score.quantile(trim_quantile)]
        ok2 = ok & days.isin(keep_days)
        mech.model = mech.make_model().fit(X[ok2], y[ok2])
        mech.noise_std = float(np.std(y[ok2] - mech.model.predict(X[ok2])))
        mech.trimmed_days = sorted(set(score.index) - set(keep_days))
    return mechs


def abduct(mechs, arrs):
    """Noise of every node at every step: observed value minus mechanism prediction."""
    u = {}
    for node, mech in mechs.items():
        X, y, prev = node_data(mech, arrs)
        ok = np.isfinite(X).all(1)
        pred = np.full(len(y), np.nan)
        pred[ok] = mech.predict(X[ok], prev[ok] if mech.delta else None)
        u[node] = arrs[node][0] - pred
    return u


# ---------------------------------------------------------------------------
# Simulation (counterfactual, vectorised over scenarios)
# ---------------------------------------------------------------------------
def simulate(mechs, arrs, noise, t0, t1, active, faults=None, order=ENDO):
    """
    Closed-loop simulation of steps t0..t1-1 for S scenarios. Before t0 every variable keeps its
    observed value; exogenous variables keep their observed values throughout.
    active   (S, len(order)) weights in [0, 1] on each node's noise (1 = actual behaviour, 0 = expected)
    faults   optional {node: f(t, x) -> x} applied to the node's value (fault injection)
    Returns the simulated arrays (S, time) of the endogenous nodes.
    """
    active = np.atleast_2d(np.asarray(active, float))
    S = active.shape[0]
    sim = {}
    for v in arrs:   # arrays are (1, time) shared by all scenarios, or (S, time) per scenario (rolling windows)
        a = arrs[v][:, :t1]
        sim[v] = (a.copy() if a.shape[0] == S else np.repeat(a, S, 0)) if v in mechs else a
    for t in range(t0, t1):
        for j, node in enumerate(order):
            mech = mechs[node]
            X = feature_matrix(sim, mech.specs, t)
            x = mech.predict(X, sim[node][:, t - 1] if mech.delta else None)
            u = noise[node]
            u = np.nan_to_num(u[:, t] if u.ndim == 2 else np.full(S, u[t]))
            x = x + active[:, j] * u
            if faults and node in faults:
                x = faults[node](t, x)
            sim[node][:, t] = mech.bound(x)
    return {v: sim[v] for v in order}


# ---------------------------------------------------------------------------
# Quantification
# ---------------------------------------------------------------------------
REFERENCE = {"dT_envelope": 0.0, "solar_30min": 0.0, "solar_2h": 0.0, "co2": 420.0,
             "hour_sin": 0.0, "hour_cos": -1.0, "workday": 0.0, "cooling_valve": 0.0, "VAV_damper": 0.0,
             "heating_valve": 0.0}


def group_contributions(mech, X, reference=None):
    """
    Contribution of every additive group to the room's temperature change (K per 30 min) at every row:
    the group at its actual values minus the group at its reference state (component off, no sun, empty
    room, outdoor = room), with the group's context columns (schedule) at their actual values in both.
    Exact because the room mechanism is additive across groups (interaction constraints). Columns without
    a reference value use their median. 'Other' = prediction minus the sum of the contributions.
    """
    cols = list(mech.specs)
    ref = np.array([(reference or REFERENCE).get(c, np.nanmedian(X[:, i])) for i, c in enumerate(cols)])
    out = {}
    for g, members in mech.groups.items():
        ctx = [cols.index(c) for c in mech.context.get(g, [])]
        idx = [cols.index(c) for c in members]
        X_ref = np.tile(ref, (len(X), 1)); X_ref[:, ctx] = X[:, ctx]
        X_act = X_ref.copy(); X_act[:, idx] = X[:, idx]
        out[g] = mech.model.predict(X_act) - mech.model.predict(X_ref)
    out = pd.DataFrame(out)
    out["Constant gain"] = mech.model.predict(X) - out.sum(1).to_numpy()
    return out


# ---------------------------------------------------------------------------
# Detection and root-cause attribution
# ---------------------------------------------------------------------------
def day_windows(index):
    """(start, end) positions of every complete local day that has a previous day of history."""
    d = pd.Series(np.arange(len(index)), index=index).groupby(index.normalize())
    w = [(g.iloc[0], g.iloc[-1] + 1) for _, g in d if len(g) >= 46]
    return [x for x in w if x[0] >= 48]


def expected_days(mechs, arrs, noise, windows):
    """Closed-loop expected room temperature per day (all node noise off), and observed minus expected."""
    n = arrs[T].shape[1]
    exp = np.full(n, np.nan)
    for t0, t1 in windows:
        exp[t0:t1] = simulate(mechs, arrs, noise, t0, t1, np.zeros((1, len(ENDO))))[T][0, t0:t1]
    return exp, arrs[T][0] - exp


def horizon_thresholds(dev, windows, days_mask, q=0.995):
    """Threshold on |deviation| per 3-hour block of the day, from the given (training, normal) days."""
    vals = {b: [] for b in range(8)}
    for (t0, t1), keep in zip(windows, days_mask):
        if keep:
            for i, t in enumerate(range(t0, t1)):
                vals[min(i // 6, 7)].append(abs(dev[t]))
    return np.array([np.nanquantile(vals[b], q) for b in range(8)])


def find_episodes(dev, windows, thr, min_steps=4):
    """Runs of at least 'min_steps' steps (2 h) with |deviation| above the threshold, within each day."""
    eps = []
    for t0, t1 in windows:
        th = np.array([thr[min((t - t0) // 6, 7)] for t in range(t0, t1)])
        flag = np.abs(dev[t0:t1]) > th
        i = 0
        while i < len(flag):
            if flag[i]:
                j = i
                while j < len(flag) and flag[j] and np.sign(dev[t0 + j]) == np.sign(dev[t0 + i]):
                    j += 1
                if j - i >= min_steps:
                    eps.append({"day_start": t0, "day_end": t1, "start": t0 + i, "end": t0 + j,
                                "sign": float(np.sign(dev[t0 + i]))})
                i = max(j, i + 1)
            else:
                i += 1
    return eps


HIST, HORIZON = 64, 8      # history needed by the features (steps); prediction horizon (8 steps = 4 h)


def stack_windows(arrs, noise, starts, hist=HIST, horizon=HORIZON):
    """Per-scenario windows [start - hist, start + horizon) of every array and noise series."""
    idx = np.asarray(starts)[:, None] + np.arange(-hist, horizon)[None, :]
    A = {v: arrs[v][0][idx] for v in arrs}
    U = {n: noise[n][idx] for n in noise}
    return A, U


def rolling_deviation(mechs, arrs, noise, horizon=HORIZON, batch=4000):
    """
    Closed-loop prediction error of the room temperature 'horizon' steps ahead: at every step t, the SCM
    is restarted from the observed state at t - horizon with all node noise off and run to t under the
    observed exogenous inputs. dev[t] = observed - predicted. Only the last 'horizon' steps can contribute,
    so normal noise does not accumulate over a day.
    """
    n = arrs[T].shape[1]
    dev = np.full(n, np.nan)
    ends = np.arange(HIST + horizon, n)
    for b in range(0, len(ends), batch):
        e = ends[b:b + batch]
        A, U = stack_windows(arrs, noise, e - horizon + 1, horizon=horizon)   # last simulated step = e
        sim = simulate(mechs, A, U, HIST, HIST + horizon, np.zeros((len(e), len(ENDO))))
        dev[e] = arrs[T][0, e] - sim[T][:, -1]
    return dev


def thresholds_by_hour(dev, index, normal_mask, q=0.995):
    """Threshold on |dev| per 3-hour block of the local day, from normal training steps."""
    block = np.asarray(index.hour // 3)
    return np.array([np.nanquantile(np.abs(dev[normal_mask & (block == b)]), q) for b in range(8)])


def episodes_from(dev, index, thr, min_steps=4):
    """Runs of at least 'min_steps' (2 h) beyond the threshold with one sign."""
    block = np.asarray(index.hour // 3)
    flag = np.abs(dev) > thr[block]
    eps, i, n = [], 0, len(dev)
    while i < n:
        if flag[i] and np.isfinite(dev[i]):
            j = i
            while j < n and flag[j] and np.sign(dev[j]) == np.sign(dev[i]):
                j += 1
            if j - i >= min_steps:
                eps.append({"start": i, "end": j, "sign": float(np.sign(dev[i]))})
            i = max(j, i + 1)
        else:
            i += 1
    return eps


def coalitions(n):
    return np.array([[(m >> j) & 1 for j in range(n)] for m in range(2 ** n)], float)


def shapley_attribution(mechs, arrs, noise, ep, players=ENDO, horizon=HORIZON):
    """
    Exact Shapley values of the node noise terms for one episode, on the detection statistic itself:
    for every step t of the episode the SCM is restarted at t - horizon from the observed state, and only
    the coalition's nodes keep their actual (abducted) noise. The value of a coalition is the mean, over
    the episode, of the room temperature at t minus the all-normal prediction, in the episode's direction (K).
    The values sum to the episode's mean detection deviation.
    """
    n = len(players)
    C = coalitions(n)
    steps = np.arange(ep["start"], ep["end"])
    A, U = stack_windows(arrs, noise, steps - horizon + 1, horizon=horizon)
    k = len(steps)
    A = {v: np.tile(a, (len(C), 1)) for v, a in A.items()}
    U = {v: np.tile(a, (len(C), 1)) for v, a in U.items()}
    active = np.zeros((len(C) * k, len(ENDO)))
    for j, p in enumerate(players):
        active[:, ENDO.index(p)] = np.repeat(C[:, j], k)
    end = simulate(mechs, A, U, HIST, HIST + horizon, active)[T][:, -1].reshape(len(C), k)
    v = ep["sign"] * (end.mean(1) - end[0].mean())
    key = {tuple(c): i for i, c in enumerate(C.astype(int))}
    phi = np.zeros(n)
    for j in range(n):
        others = [k for k in range(n) if k != j]
        for r in range(n):
            w = factorial(r) * factorial(n - r - 1) / factorial(n)
            for sub in combinations(others, r):
                c = np.zeros(n, int); c[list(sub)] = 1
                c2 = c.copy(); c2[j] = 1
                phi[j] += w * (v[key[tuple(c2)]] - v[key[tuple(c)]])
    return pd.Series(phi, index=players), v[-1]


def local_scores(mechs, noise, ep, players=ENDO):
    """Baseline 1: mean |standardised noise| of every node over the episode (mechanism anomaly, no propagation)."""
    return pd.Series({p: np.nanmean(np.abs(noise[p][ep["start"]:ep["end"]])) / mechs[p].noise_std for p in players})


def marginal_scores(df, typical, ep, players=ENDO):
    """Baseline 2: mean |deviation from the typical value for that month, hour and day type| / std (no causal model)."""
    sl = slice(ep["start"], ep["end"])
    return pd.Series({p: np.nanmean(np.abs(df[p].to_numpy()[sl] - typical[p].to_numpy()[sl])) / df[p].std() for p in players})


def typical_values(df):
    key = [df.index.month, df.index.hour, df.index.minute, df["workday"]]
    return df[ENDO].groupby(key).transform("median")
