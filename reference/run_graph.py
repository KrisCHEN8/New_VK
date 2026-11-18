import os
import warnings; warnings.filterwarnings("ignore")
import pickle, copy, numpy as np, pandas as pd
import causal_fdd as cf
df = cf.load_data(os.environ.get("FDD_DATA", "dfs.pkl")); mechs, test = pickle.load(open("mechs.pkl", "rb"))
arrs = cf.arrays(df); rng = np.random.default_rng(0)
PARENT = {  # feature column -> parent variable (edge)
    "room": cf.T, "err_1h": cf.T, "err_2h": cf.T, "err_4h": cf.T, "err_8h": cf.T, "setpoint": "setpoint",
    "hour_sin": "calendar", "hour_cos": "calendar", "workday": "calendar", "outdoor_24h": "outdoor_temperature",
    "outdoor": "outdoor_temperature", "rh": "relative_humidity", "rh_lag1": "relative_humidity", "co2": "co2",
    "valve_now": "cooling_valve", "valve_2h": "cooling_valve", "valve_8h": "cooling_valve", "damper": "VAV_damper"}
rows = []
for node, m in mechs.items():
    if node == cf.T: continue
    X, y, _ = cf.node_data(m, arrs); te = test & np.isfinite(X).all(1)
    base = np.mean((y[te] - m.model.predict(X[te])) ** 2)
    cols = list(m.specs); par = pd.Series({c: PARENT[c] for c in cols})
    for p, members in par.groupby(par).groups.items():
        inc = []
        for _ in range(5):
            Xp = X[te].copy(); perm = rng.permutation(te.sum())
            for c in members: Xp[:, cols.index(c)] = Xp[perm, cols.index(c)]
            inc.append(np.mean((y[te] - m.model.predict(Xp)) ** 2) / np.var(y[te]))
        rows.append({"node": node, "parent": p, "importance": np.mean(inc) - base / np.var(y[te])})
imp = pd.DataFrame(rows).pivot(index="parent", columns="node", values="importance")
print("Permutation importance (increase in held-out MSE as share of node variance):")
print(imp.round(3).to_string())
# missing edges: does adding a candidate parent improve held-out R2?
cands = {"cooling_valve": {"solar": cf.lag("solar_radiation", 0), "co2": cf.lag("co2", 0), "outdoor": cf.lag("outdoor_temperature", 0)},
         "VAV_damper": {"solar": cf.lag("solar_radiation", 0), "outdoor": cf.lag("outdoor_temperature", 0), "rh": cf.lag("relative_humidity", 0)},
         "KB02_supply": {"solar": cf.lag("solar_radiation", 0), "room": cf.lag(cf.T, 0)},
         cf.T: {}}
for node, extra in cands.items():
    for name, spec in extra.items():
        m = copy.deepcopy(mechs[node]); m.specs = {**m.specs, f"cand_{name}": spec}
        X, y, _ = cf.node_data(m, arrs); tr = ~test & np.isfinite(X).all(1) & np.isfinite(y); te = test & np.isfinite(X).all(1) & np.isfinite(y)
        Xb, _, _ = cf.node_data(mechs[node], arrs)
        r_new = 1 - np.var(y[te] - m.make_model().fit(X[tr], y[tr]).predict(X[te])) / np.var(y[te])
        r_old = 1 - np.var(y[te] - mechs[node].model.predict(Xb[te])) / np.var(y[te])
        print(f"  candidate edge {name:8s} -> {node:14s}: R2 {r_old:.3f} -> {r_new:.3f}")
imp.to_pickle("graph_importance.pkl")
