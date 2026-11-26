import numpy as np
import pandas as pd
import xgboost as xgb


BEAM_WATER = ["cooling_valve_lag1", "dT_chilled_water"]
BEAM_AIR = ["VAV_damper_lag1", "dT_supply_air"]


# emitter columns that are fixed at 'off' by the definition of each mode
INACTIVE = {
    "beam_water_air": [],
    "beam_air": BEAM_WATER,
    "off_no_sun": BEAM_WATER + BEAM_AIR,
    "off_sunny": BEAM_WATER + BEAM_AIR,
    "global": [],
}


def mode_columns(all_cols, mode, mask_inactive=True):
    drop = set(INACTIVE.get(mode, [])) if mask_inactive else set()
    return [c for c in all_cols if c not in drop]


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
        Additive decomposition: with interaction constraints every tree uses columns of one group only,
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
