import os
import warnings; warnings.filterwarnings("ignore")
import pickle, sys, time, numpy as np, pandas as pd
import causal_fdd as cf

DATA = os.environ.get("FDD_DATA", "dfs.pkl")
df = cf.load_data(DATA); mechs, test = pickle.load(open("mechs.pkl", "rb"))
arrs = cf.arrays(df); u_all = cf.abduct(mechs, arrs)
dev_all = cf.rolling_deviation(mechs, arrs, u_all)
typ = cf.typical_values(df)
days = df.index.normalize()
N_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 20

FAULTS = {  # name -> (node it acts on, maker of f(t, x) given the onset index)
    "Cooling valve stuck closed": ("cooling_valve", lambda on: lambda t, x: np.where(t >= on, 0.0, x)),
    "Cooling valve capped at 30 %": ("cooling_valve", lambda on: lambda t, x: np.where(t >= on, np.minimum(x, 30.0), x)),
    "Chilled-water supply +3 K": ("KB02_supply", lambda on: lambda t, x: np.where(t >= on, x + 3.0, x)),
    "VAV damper stuck at 20 %": ("VAV_damper", lambda on: lambda t, x: np.where(t >= on, 20.0, x)),
    "AHU supply air +3 K": ("AHU_supply_temperature", lambda on: lambda t, x: np.where(t >= on, x + 3.0, x)),
    "Extra room heat gain +0.2 K/h": (cf.T, lambda on: lambda t, x: np.where(t >= on, x + 0.1, x)),
    "Radiator valve stuck at 50 %": ("heating_valve", lambda on: lambda t, x: np.where(t >= on, 50.0, x)),
}

trimmed = set().union(*[set(m.trimmed_days) for m in mechs.values()])
wins = cf.day_windows(df.index)
cand = [w for w in wins if test[w[0]] and df.index[w[0]].month in [5, 6, 7, 8, 9] and df.workday.iloc[w[0]] == 1
        and days[w[0]] not in trimmed]
rng = np.random.default_rng(1)
val_days = [cand[i] for i in sorted(rng.choice(len(cand), min(N_DAYS, len(cand)), replace=False))]
val_set = {days[t0] for t0, _ in val_days}
calib = test & ~days.isin(list(trimmed)) & ~days.isin(list(val_set)) & np.isfinite(dev_all)
calib_days = sorted(set(days[calib]))

# layer 1 (room temperature): smallest quantile with <= 5 % of calibration days raising an episode
for q in [0.99, 0.995, 0.998, 0.999]:
    thr = cf.thresholds_by_hour(dev_all, df.index, calib, q)
    eps = cf.episodes_from(np.where(calib, dev_all, np.nan), df.index, thr)
    rate = len({days[e["start"]] for e in eps}) / len(calib_days)
    if rate <= 0.05:
        break
print(f"layer 1: quantile {q}, thresholds {thr.round(2)}, calibration days with a false episode {rate:.3f} of {len(calib_days)}", flush=True)

# layer 2 (node residuals): 99th percentile of the 09-24 daily mean |z| on calibration days
Z = pd.DataFrame({n: np.abs(u_all[n]) / mechs[n].noise_std for n in cf.ENDO}, index=df.index)
zd = Z[df.index.hour >= 9]; zd = zd.groupby(zd.index.normalize()).mean()
rthr = zd[zd.index.isin(calib_days)].quantile(0.99)
fam = (zd[zd.index.isin(calib_days)] > rthr).any(axis=1).mean()
print("layer 2 thresholds:", rthr.round(2).to_dict(), f"| calibration days with any residual alarm {fam:.3f}", flush=True)
pickle.dump({"q": q, "thr": thr, "rthr": rthr, "calib_days": calib_days, "val_days": sorted(val_set), "rate": rate, "fam": fam},
            open("calibration.pkl", "wb"))


def run_case(t0, t1, fault):
    lo = t0 - cf.HIST - cf.HORIZON - 4
    a = {k: v[:, lo:t1].copy() for k, v in arrs.items()}
    s0, s1 = t0 - lo, t1 - lo
    idx = df.index[lo:t1]
    u_real = cf.abduct(mechs, a)
    on = s0 + 18                                               # 09:00 local
    faults = None
    if fault:
        node, make = FAULTS[fault]; faults = {node: make(on)}
    syn = cf.simulate(mechs, a, u_real, s0, s1, np.ones((1, 7)), faults=faults)
    for n in cf.ENDO:
        a[n][:, s0:s1] = syn[n][:, s0:s1]
    u = cf.abduct(mechs, a)
    dev = cf.rolling_deviation(mechs, a, u)
    dev[:on] = np.nan
    eps = cf.episodes_from(dev, idx, thr)
    z = pd.Series({n: np.nanmean(np.abs(u[n][on:s1])) / mechs[n].noise_std for n in cf.ENDO})
    ratio = z / rthr
    out = {"detected": bool(eps), "resid_alarm": bool((ratio > 1).any()), "resid_top": ratio.idxmax()}
    if eps:
        ep = max(eps, key=lambda e: np.abs(dev[e["start"]:e["end"]]).mean())
        phi, total = cf.shapley_attribution(mechs, a, u, ep)
        loc = cf.local_scores(mechs, u, ep)
        sl = slice(ep["start"], ep["end"]); typ_a = typ.iloc[lo:t1]
        mar = pd.Series({p: np.nanmean(np.abs(a[p][0, sl] - typ_a[p].to_numpy()[sl])) / df[p].std() for p in cf.ENDO})
        out.update({"dev_K": float(np.mean(dev[sl])), "shapley_total": float(total), "top_shapley": phi.idxmax(),
                    "top_local": loc.idxmax(), "top_marginal": mar.idxmax(),
                    "share_true": float(phi[FAULTS[fault][0]] / phi.clip(lower=0).sum()) if fault else np.nan,
                    **{f"phi_{k}": v for k, v in phi.items()}})
    return out


rows = []
t = time.time()
for fault in [None] + list(FAULTS):
    for t0, t1 in val_days:
        rows.append({"fault": fault or "No fault", "day": df.index[t0].date(), **run_case(t0, t1, fault)})
    print(fault, round(time.time() - t), "s", flush=True)
R = pd.DataFrame(rows); R.to_pickle("validation.pkl")
true = {f: n for f, (n, _) in FAULTS.items()}
summ = []
for f, g in R.groupby("fault", sort=False):
    det = g[g.detected]
    row = {"fault": f, "cases": len(g), "room_detected": g.detected.mean(), "resid_alarm": g.resid_alarm.mean(),
           "either": (g.detected | g.resid_alarm).mean()}
    if f in true:
        row["resid_top1"] = (g[g.resid_alarm].resid_top == true[f]).mean() if g.resid_alarm.any() else np.nan
        if len(det):
            for m in ["shapley", "local", "marginal"]:
                row[f"top1_{m}"] = (det[f"top_{m}"] == true[f]).mean()
            row["mean_dev_K"] = det.dev_K.mean(); row["share_true"] = det.share_true.mean()
    summ.append(row)
S = pd.DataFrame(summ).set_index("fault"); S.to_pickle("validation_summary.pkl")
pd.set_option("display.width", 220); print(S.round(2).to_string())
