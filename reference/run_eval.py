import os
import warnings; warnings.filterwarnings("ignore")
import pickle, time, numpy as np, pandas as pd
import causal_fdd as cf
from sklearn.ensemble import HistGradientBoostingRegressor
df = cf.load_data(os.environ.get("FDD_DATA", "dfs.pkl")); mechs, test = pickle.load(open("mechs.pkl", "rb"))
arrs = cf.arrays(df); u = cf.abduct(mechs, arrs)
win = cf.day_windows(df.index)
tday = [w for w in win if test[w[0]]]
t = time.time()
# closed loop: everything simulated; open loop: only the room simulated, actuators/plant observed
res = {"closed": {n: [] for n in cf.ENDO}, "open": [], "persist": []}
for t0, t1 in tday[::2]:
    s = cf.simulate(mechs, arrs, u, t0, t1, np.zeros((1, 7)))
    for n in cf.ENDO: res["closed"][n].append(s[n][0, t0:t1] - arrs[n][0, t0:t1])
    act = np.ones((1, 7)); act[0, 0] = 0
    # open loop: room noise off, but actuators follow observed (their noise on, and they react to simulated T...)
    so = cf.simulate(mechs, arrs, u, t0, t1, np.zeros((1, 7)), order=[cf.T])
    res["open"].append(so[cf.T][0, t0:t1] - arrs[cf.T][0, t0:t1])
    res["persist"].append(arrs[cf.T][0, t0 - 1] - arrs[cf.T][0, t0:t1])
print("sim time", round(time.time() - t, 1), "s for", len(tday[::2]), "days")
rm = lambda L: float(np.sqrt(np.nanmean(np.concatenate(L) ** 2)))
print("24-h room temperature RMSE: closed loop", round(rm(res["closed"][cf.T]), 3), "| open loop (actuators observed)",
      round(rm(res["open"]), 3), "| persistence", round(rm(res["persist"]), 3))
for n in cf.ENDO[1:]:
    print(f"  closed-loop 24-h RMSE {n}: {rm(res['closed'][n]):.2f}  (std of signal {df[n][test].std():.2f})")
# by season
idx = np.array([df.index[t0].month for t0, _ in tday[::2]])
for name, months in [("cooling season May-Sep", [5,6,7,8,9]), ("Oct-Apr", [10,11,12,1,2,3,4])]:
    sel = np.isin(idx, months)
    print(name, "closed", round(rm([e for e, s in zip(res["closed"][cf.T], sel) if s]), 3),
          "open", round(rm([e for e, s in zip(res["open"], sel) if s]), 3),
          "persist", round(rm([e for e, s in zip(res["persist"], sel) if s]), 3))
# price of additivity: unconstrained model on the same room features
m = mechs[cf.T]; X, y, _ = cf.node_data(m, arrs)
ok_tr = ~test & np.isfinite(X).all(1) & np.isfinite(y); ok_te = test & np.isfinite(X).all(1) & np.isfinite(y)
free = HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                     early_stopping=True, random_state=0).fit(X[ok_tr], y[ok_tr])
r2 = lambda p: 1 - np.var(y[ok_te] - p) / np.var(y[ok_te])
print("room one-step R2: additive", round(r2(m.model.predict(X[ok_te])), 3), "| unconstrained", round(r2(free.predict(X[ok_te])), 3))
summer = ok_te & np.isin(df.index.month, [5,6,7,8,9])
print("cooling season R2 additive", round(1 - np.var(y[summer] - m.model.predict(X[summer])) / np.var(y[summer]), 3))
