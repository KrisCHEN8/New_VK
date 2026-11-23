"""
Model classes for the indoor_temperature mechanism of the user's causal graph.

All models take the graph's design matrix (node_frame) and respect its settings:
  - monotone signs  (gcm.monotone_constraints)
  - additive groups (gcm interaction_groups): a model may combine columns only inside one group

LinearPhys : sign-constrained least squares on the graph's columns, plus the two products allowed inside the
             chilled_beam group (valve x water dT, damper x supply-air dT) - the physical form of beam heat transfer.
XGBGraph   : XGBoost with the graph's monotone and interaction constraints.
Both can drop the columns of an emitter that is off by definition of the mode ("structural zeros").
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import lsq_linear

BEAM_WATER = ["cooling_valve_lag1", "dT_chilled_water"]
BEAM_AIR = ["VAV_damper_lag1", "dT_supply_air"]

# emitter columns that are fixed at 'off' by the definition of each mode
INACTIVE = {
    "beam_water_air": [],
    "beam_air": BEAM_WATER,
    "off_no_sun": BEAM_WATER + BEAM_AIR,
    "off_sunny": BEAM_WATER + BEAM_AIR,
    "other": [],
    "global": [],
}


def mode_columns(all_cols, mode, mask_inactive=True):
    drop = set(INACTIVE.get(mode, [])) if mask_inactive else set()
    return [c for c in all_cols if c not in drop]


class LinearPhys:
    name = "linear"

    def __init__(self, monotone, groups, products=True):
        self.monotone, self.groups, self.products = dict(monotone), groups, products

    def _design(self, X):
        Z = X.copy()
        if self.products:
            if {"cooling_valve_lag1", "dT_chilled_water"} <= set(X):
                Z["beam_water_x"] = X.cooling_valve_lag1 / 100 * X.dT_chilled_water
            if {"VAV_damper_lag1", "dT_supply_air"} <= set(X):
                Z["beam_air_x"] = X.VAV_damper_lag1 / 100 * X.dT_supply_air
        return Z

    def fit(self, X, y, **_):
        self.cols = list(X.columns)                # input columns (as for XGBGraph)
        Z = self._design(X)
        sign = {**self.monotone, "beam_water_x": -1, "beam_air_x": -1}
        self.zcols = list(Z.columns)
        self.mu, self.sd = Z.mean(), Z.std().replace(0, 1)
        A = np.c_[np.ones(len(Z)), ((Z - self.mu) / self.sd).values]
        lb = [-np.inf] + [0 if sign.get(c, 0) > 0 else -np.inf for c in self.zcols]
        ub = [np.inf] + [0 if sign.get(c, 0) < 0 else np.inf for c in self.zcols]
        r = lsq_linear(A, y.values, bounds=(lb, ub), method="bvls")
        self.b0 = r.x[0]
        self.coef = pd.Series(r.x[1:] / self.sd.values, index=self.zcols)       # per original unit
        self.intercept = self.b0 - (self.coef * self.mu).sum()
        return self

    def predict(self, X):
        return self.intercept + self._design(X[self.cols])[self.zcols].values @ self.coef.values

    def group_contributions(self, X):
        """Additive pieces per graph group (calendar columns go to 'schedule'); the intercept separately."""
        Z = self._design(X[self.cols])[self.zcols]
        parts = Z * self.coef
        out = {}
        for c in self.zcols:
            g = _group_of(c, self.groups)
            out[g] = out.get(g, 0) + parts[c]
        out = pd.DataFrame(out, index=X.index)
        out["_intercept"] = self.intercept
        return out


def _group_of(col, groups):
    if col in ("beam_water_x", "beam_air_x"):
        return "chilled_beam"
    owners = [g for g, cols in groups.items() if col in cols]
    if "schedule" in owners and len(owners) > 1:           # calendar columns belong to several groups
        return "schedule"
    return owners[0]


class XGBGraph:
    name = "xgb"

    def __init__(self, monotone, groups, constrained=True, **params):
        self.monotone, self.groups, self.constrained = dict(monotone), groups, constrained
        self.params = dict(n_estimators=3000, learning_rate=0.03, max_depth=3, min_child_weight=20,
                           subsample=0.8, colsample_bytree=1.0, reg_lambda=1.0, tree_method="hist",
                           early_stopping_rounds=100, n_jobs=2, random_state=0)
        self.params.update(params)

    def fit(self, X, y, X_val=None, y_val=None, n_estimators=None):
        self.cols = list(X.columns)
        p = dict(self.params)
        if self.constrained:
            p["monotone_constraints"] = tuple(self.monotone.get(c, 0) for c in self.cols)
            sets = [[c for c in cols if c in self.cols] for cols in self.groups.values()]
            p["interaction_constraints"] = [s for s in sets if s]
        if X_val is None:
            p.pop("early_stopping_rounds")
            if n_estimators:
                p["n_estimators"] = n_estimators
            self.m = xgb.XGBRegressor(**p).fit(X, y, verbose=False)
            self.best_n = p["n_estimators"]
        else:
            self.m = xgb.XGBRegressor(**p).fit(X, y, eval_set=[(X_val[self.cols], y_val)], verbose=False)
            self.best_n = self.m.best_iteration + 1
        return self

    def predict(self, X):
        return self.m.predict(X[self.cols], iteration_range=(0, self.best_n))

    def group_contributions(self, X):
        """
        Exact additive decomposition: with interaction constraints every tree uses columns of one group only,
        so the prediction = base + sum over trees, and each tree belongs to one group. Trees that use only
        calendar columns are assigned to 'schedule'.
        """
        b = self.m.get_booster()
        td = b.trees_to_dataframe()
        td = td[td.Tree < self.best_n]
        # features on the path from the root to every leaf (constraints apply per branch, not per tree)
        leaf_group, leaf_val = {}, {}
        for t, tt in td.groupby("Tree"):
            nodes = tt.set_index("ID")
            stack = [(f"{t}-0", frozenset())]
            while stack:
                nid, path = stack.pop()
                r = nodes.loc[nid]
                if r.Feature == "Leaf":
                    owners = [g for g, cols in self.groups.items() if path <= set(cols)]
                    leaf_group[(t, r.Node)] = "schedule" if (not path or "schedule" in owners) else owners[0]
                    leaf_val[(t, r.Node)] = r.Gain
                else:
                    p2 = path | {r.Feature}
                    stack += [(r.Yes, p2), (r.No, p2)]
        leaves = b.predict(xgb.DMatrix(X[self.cols]), pred_leaf=True,
                           iteration_range=(0, self.best_n)).astype(int)
        gnames = sorted(set(leaf_group.values()))
        gidx = {g: i for i, g in enumerate(gnames)}
        width = int(td.Node.max()) + 1
        vals = np.zeros((leaves.shape[1], width)); grp = np.zeros((leaves.shape[1], width), int)
        for (t, node), v in leaf_val.items():
            vals[t, node] = v
            grp[t, node] = gidx[leaf_group[(t, node)]]
        tt = np.arange(leaves.shape[1])[None, :]
        V, Gm = vals[tt, leaves], grp[tt, leaves]          # rows x trees
        out = {g: np.where(Gm == i, V, 0.0).sum(axis=1) for g, i in gidx.items()}
        out = pd.DataFrame(out, index=X.index)
        out["_intercept"] = self._base()
        return out

    def _base(self):
        import json
        cfg = json.loads(self.m.get_booster().save_config())
        return float(cfg["learner"]["learner_model_param"]["base_score"].strip("[]"))
