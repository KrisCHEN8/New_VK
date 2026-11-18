import os
import warnings; warnings.filterwarnings("ignore")
import pickle, time, numpy as np, pandas as pd
import causal_fdd as cf
t = time.time()
df = cf.load_data(os.environ.get("FDD_DATA", "dfs.pkl"))
test = cf.test_blocks(df.index)
mechs = cf.fit_mechanisms(df, ~test)
arrs = cf.arrays(df)
u = cf.abduct(mechs, arrs)
print("fit", round(time.time() - t, 1), "s")
# one-step quality on held-out blocks
rows = []
for n, m in mechs.items():
    X, y, prev = cf.node_data(m, arrs)
    ok = test & np.isfinite(X).all(1) & np.isfinite(y)
    p = m.model.predict(X[ok]); r = y[ok] - p
    rows.append({"node": n, "target": "change" if m.delta else "level", "R2": 1 - r.var() / y[ok].var(),
                 "RMSE": np.sqrt(np.mean(r**2)), "noise_std_train": m.noise_std, "trimmed_days": len(m.trimmed_days)})
print(pd.DataFrame(rows).round(3).to_string(index=False))
# consistency: full actual noise reproduces observations
t0 = 48 * 200; sim = cf.simulate(mechs, arrs, u, t0, t0 + 48, np.ones((1, len(cf.ENDO))))
print("max |sim - obs| with actual noise:", {n: float(np.nanmax(np.abs(sim[n][0, t0:t0+48] - arrs[n][0, t0:t0+48]))) for n in cf.ENDO})
pickle.dump((mechs, test), open("mechs.pkl", "wb"))
