import os
import warnings; warnings.filterwarnings("ignore")
import pickle, time, numpy as np, pandas as pd
import causal_fdd as cf
df = cf.load_data(os.environ.get("FDD_DATA", "dfs.pkl")); mechs, test = pickle.load(open("mechs.pkl", "rb"))
arrs = cf.arrays(df); u = cf.abduct(mechs, arrs)
cal = pickle.load(open("calibration.pkl", "rb")); thr, rthr = cal["thr"], cal["rthr"]
dev = cf.rolling_deviation(mechs, arrs, u)
eps = cf.episodes_from(dev, df.index, thr)
typ = cf.typical_values(df)
rows = []
for ep in eps:
    phi, total = cf.shapley_attribution(mechs, arrs, u, ep)
    loc = cf.local_scores(mechs, u, ep); mar = cf.marginal_scores(df, typ, ep)
    rows.append({"start": df.index[ep["start"]], "end": df.index[ep["end"] - 1], "hours": (ep["end"] - ep["start"]) / 2,
                 "mean_dev": float(np.mean(dev[ep["start"]:ep["end"]])), "shapley_total": float(total),
                 **{f"phi_{k}": v for k, v in phi.items()}, "top_shapley": phi.idxmax(), "top_local": loc.idxmax(), "top_marginal": mar.idxmax()})
R = pd.DataFrame(rows); R.to_pickle("episodes_year.pkl")
# layer 2 over the year: daily residual alarms per node (09-24)
Z = pd.DataFrame({n: np.abs(u[n]) / mechs[n].noise_std for n in cf.ENDO}, index=df.index)
zd = Z[df.index.hour >= 9]; zd = zd.groupby(zd.index.normalize()).mean()
alarm = zd > rthr
alarm.to_pickle("resid_alarms.pkl"); np.save("dev4h.npy", dev)
pd.set_option("display.width", 250)
ph = R[[c for c in R if c.startswith("phi_")]].round(2); ph.columns = [c[4:12] for c in ph.columns]
print(pd.concat([R[["start", "hours", "mean_dev", "top_local", "top_marginal"]].round(2), ph], axis=1).to_string())
print("\nresidual alarm days per node:", alarm.sum().to_dict())
print("by month:"); print(alarm.groupby(alarm.index.to_period("M")).sum().to_string())
