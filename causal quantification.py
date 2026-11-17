import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import GCM_generation as gcm


# Data preparation
def prepare_data(df, G, fault_free=None, valve_active_threshold=5.0, config=None):
    cfg = {**(config or {}), "valve_active_threshold": valve_active_threshold}
    feats = gcm.make_features(df, G, cfg)
    feats["regime"] = gcm.label_regimes(df.sort_index().asfreq(gcm.CONFIG["freq"]), cfg)
    if fault_free is not None:
        keep = fault_free.reindex(feats.index).fillna(False).astype(bool)
        feats = feats[keep]
    return feats


def split_blocks(index, block="7D", test_every=3, offset=1):
    """
    Time-block split: the index is cut into consecutive blocks (7 days) and every
    'test_every'-th block is held out.
    The blocks spread the held-out data over the whole period, so every season is represented
    """
    start = index.min().floor("D")
    block_id = ((index - start) // pd.Timedelta(block)).astype(int)
    return np.asarray(block_id % test_every == offset)


# Mechanism fitting and evaluation
def room_frame(G, feats, delta=True):
    """
    Target and design matrix of the room node. With delta, the target is the change of the room
    temperature over one interval (T - T_lag1), so the strengths describe what drives that change
    instead of being dominated by thermal inertia.
    """
    node = G.graph["target"]
    y, X = gcm.node_frame(G, node, feats)
    if delta:
        y = y - X[f"{node}_lag1"]
    return y.rename("target"), X


def make_model(kind, monotonic_cst=None, random_state=0):
    """'linear' (ordinary least squares) or 'gbm' (gradient boosting, monotone where signs are given)."""
    if kind == "linear":
        return LinearRegression()
    if kind == "gbm":
        return HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
            l2_regularization=1.0, monotonic_cst=monotonic_cst,
            early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=random_state)
    raise ValueError(f"unknown model kind '{kind}'")


def fit_mechanism(G, node, y, X, kind):
    """Fit one mechanism. Monotone signs are applied to the gradient-boosting model."""
    cst = gcm.monotone_constraints(G, node, X.columns) if kind == "gbm" else None
    return make_model(kind, cst).fit(X, y)


def mechanism_quality(model, y, X):
    """
    Held-out fit quality: R2, RMSE and the lag-1 autocorrelation of the residuals. A residual
    autocorrelation far from 0 means slow drivers are missing from the mechanism (the noise then
    acts as a hidden confounder of the lagged inputs).
    """
    resid = y - model.predict(X)
    lag1 = resid.shift(1)[resid.index.to_series().diff() == pd.Timedelta(gcm.CONFIG["freq"])]
    return {"r2": r2_score(y, model.predict(X)),
            "rmse": mean_squared_error(y, model.predict(X)) ** 0.5,
            "resid_autocorr": resid.loc[lag1.index].corr(lag1)}


def controller_gains(G, feats, regime=None):
    """
    Step 0 for the controller and set-point nodes: a linear fit of each non-room node on its inputs.
    These edges are close to deterministic, so they are reported as gains (units of the node per unit
    of the input, e.g. % valve per degC) and R2, not ranked against the physical edges of the room.
    """
    data = feats if regime is None else feats[feats["regime"] == regime]
    rows = []
    for node in G:
        if G.in_degree(node) == 0 or node == G.graph["target"]:
            continue
        y, X = gcm.node_frame(G, node, data)
        if len(y) < 50:
            continue
        model = LinearRegression().fit(X, y)
        for col, coef in zip(X.columns, model.coef_):
            rows.append({"node": node, "input": col, "gain": coef, "r2": model.score(X, y)})
    return pd.DataFrame(rows)


# Causal strength of a group
def _permutation(index, workday, rng, conditional):
    """Row order for one cut. Conditional: rows are only exchanged within the same hour and day type."""
    n = len(index)
    if not conditional:
        return rng.permutation(n)
    order = np.arange(n)
    groups = pd.Series(np.arange(n)).groupby([index.hour, index.minute, np.asarray(workday)]).indices
    for rows in groups.values():
        order[rows] = rng.permutation(rows)
    return order


def cut_predictions(model, G, node, X, bundle, n_perm=50, conditional=False, seed=0):
    """
    Predictions of the mechanism with one bundle cut, for n_perm independent cuts.
    The bundle's columns are permuted together (one row order for the whole bundle), the
    interaction terms are recomputed from the permuted columns, and all other columns stay as
    observed. Returns an array of shape (n_perm, rows).
    """
    rng = np.random.default_rng(seed)
    inputs = X[gcm.mechanism_inputs(G, node)]
    workday = X["workday"] if "workday" in X else np.zeros(len(X))
    preds = np.empty((n_perm, len(X)))
    for k in range(n_perm):
        cut = inputs.copy()
        order = _permutation(X.index, workday, rng, conditional)
        cut[bundle] = inputs[bundle].to_numpy()[order]
        preds[k] = model.predict(gcm.design_matrix(G, node, cut)[X.columns])
    return preds


def bundle_strength(model, G, node, y, X, bundle, n_perm=50, conditional=False, seed=0):
    """
    Causal strength of one bundle on held-out data (y, X), in two measures:

    variance  E[Var(node | other inputs)] added by the cut: the spread the node gains when the bundle
              is replaced by an independent copy while every other input is held at its observed
              value. Units of the node squared (e.g. K^2 per interval for delta targets).
    share     variance divided by the variance of the node: the fraction of its variability that
              depends on the bundle (comparable across nodes and regimes).
    kl        Kullback-Leibler divergence between the observed and the cut conditional distribution,
              the measure of Janzing et al. (2013), in nats. It assumes Gaussian residuals with the
              held-out residual variance: KL = mean_i [log N(y_i; f_i, s) - log mean_k N(y_i; f_ik, s)].
    """
    pred = model.predict(X)
    cut = cut_predictions(model, G, node, X, bundle, n_perm, conditional, seed)
    variance = np.mean(np.var(cut, axis=0))

    s = np.std(y - pred)
    log_obs = -0.5 * ((y.to_numpy() - pred) / s) ** 2
    log_cut = -0.5 * ((y.to_numpy() - cut) / s) ** 2
    log_mix = np.logaddexp.reduce(log_cut, axis=0) - np.log(n_perm)
    return {"variance": variance, "share": variance / np.var(y), "kl": np.mean(log_obs - log_mix)}


def quantify_room(G, feats, kind="linear", delta=True, by_component=True, n_perm=50, conditional=False,
                  min_rows=500, seed=0):
    """
    Steps 2-3 for the room node, separately for every regime: fit on the training blocks, check the
    fit and compute the strength of every bundle on the held-out blocks. Returns one row per
    regime and bundle, with the held-out fit quality of that regime's mechanism.
    """
    node = G.graph["target"]
    bundles = gcm.edge_bundles(G, node, by_component=by_component)
    rows = []
    for regime, data in feats.groupby("regime"):
        y, X = room_frame(G, data, delta)
        if len(y) < min_rows:
            continue
        test = split_blocks(X.index)
        model = fit_mechanism(G, node, y[~test], X[~test], kind)
        quality = mechanism_quality(model, y[test], X[test])
        for name, cols in bundles.items():
            if conditional and set(cols) <= set(gcm.CAL_FEATURES):
                # within one hour and day type the calendar columns are constant: a conditional cut is empty
                strength = {"variance": np.nan, "share": np.nan, "kl": np.nan}
            else:
                strength = bundle_strength(model, G, node, y[test], X[test], cols, n_perm, conditional, seed)
            rows.append({"regime": regime, "bundle": name, "columns": ", ".join(cols), **strength,
                         "model": kind, "n_train": int((~test).sum()), "n_test": int(test.sum()), **quality})
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(["regime", "variance"], ascending=[True, False]).reset_index(drop=True)
    return table


def run_quantification(df, G, fault_free=None, valve_active_threshold=5.0, kinds=("linear", "gbm"),
                       conditional_check=True, **kwargs):
    """
    Steps 0-3 end to end. Returns (strengths, controllers):
      strengths    one row per regime, model kind and bundle; with conditional_check, an extra
                   'strength_conditional' block of rows where the cut only exchanges rows with the same
                   hour and day type (robustness against cuts that create states never observed)
      controllers  gains and R2 of the controller and set-point nodes, per regime
    """
    feats = prepare_data(df, G, fault_free, valve_active_threshold)
    tables = []
    for kind in kinds:
        tables.append(quantify_room(G, feats, kind=kind, **kwargs).assign(cut="plain"))
        if conditional_check:
            tables.append(quantify_room(G, feats, kind=kind, conditional=True, **kwargs).assign(cut="conditional"))
    strengths = pd.concat(tables, ignore_index=True)
    controllers = pd.concat([controller_gains(G, feats, r).assign(regime=r) for r in feats["regime"].unique()],
                            ignore_index=True)
    return strengths, controllers


# Read dict and convert into df
def load_data(path="07-D2-02_data/dfs.pkl"):
    """The pickled dict {variable: DataFrame with a 'value' column} as one wide DataFrame, one column per variable."""
    dfs = pd.read_pickle(path)
    return pd.concat({name: frame["value"] for name, frame in dfs.items()}, axis=1)


if __name__ == "__main__":
    G = gcm.build_gcm({"make_plots": False})
    df = load_data()
    strengths, controllers = run_quantification(df, G)
    strengths.to_csv("07-D2-02_data/causal_strengths.csv", index=False)
    controllers.to_csv("07-D2-02_data/controller_gains.csv", index=False)
    print(strengths[["regime", "model", "cut", "bundle", "share", "kl", "r2"]].to_string(index=False))
