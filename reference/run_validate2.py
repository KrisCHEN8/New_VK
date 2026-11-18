"""Attribution accuracy in kelvin on every fault case, against the true effect of the fault (known in simulation)."""
import warnings; warnings.filterwarnings("ignore")
import pickle, time, numpy as np, pandas as pd
import causal_fdd as cf
exec(open("run_validate.py").read().split("# layer 1")[0])          # data, model, faults, validation days
cal = pickle.load(open("calibration.pkl", "rb")); rthr = cal["rthr"]
COMPONENTS = [n for n in cf.ENDO if n != cf.T]

def synth(t0, t1, fault):
    lo = t0 - cf.HIST - cf.HORIZON - 4
    a = {k: v[:, lo:t1].copy() for k, v in arrs.items()}
    s0, s1 = t0 - lo, t1 - lo
    u_real = cf.abduct(mechs, a); on = s0 + 18
    faults = None
    if fault:
        node, make = FAULTS[fault]; faults = {node: make(on)}
    syn = cf.simulate(mechs, a, u_real, s0, s1, np.ones((1, 7)), faults=faults)
    for n in cf.ENDO: a[n][:, s0:s1] = syn[n][:, s0:s1]
    return a, lo, s0, s1, on

rows = []; t = time.time()
for t0, t1 in val_days:
    a0, lo, s0, s1, on = synth(t0, t1, None)
    u0 = cf.abduct(mechs, a0); dev0 = cf.rolling_deviation(mechs, a0, u0)
    win = slice(s0 + 24, s0 + 36)                                  # 12:00-18:00
    for fault in FAULTS:
        a, *_ = synth(t0, t1, fault)
        u = cf.abduct(mechs, a); dev = cf.rolling_deviation(mechs, a, u)
        true_eff = float(np.mean(dev[win] - dev0[win]))             # what the fault did to the detection statistic
        ep = {"start": win.start, "end": win.stop, "sign": 1.0}
        phi, total = cf.shapley_attribution(mechs, a, u, ep)
        node = FAULTS[fault][0]
        z = pd.Series({n: np.nanmean(np.abs(u[n][on:s1])) / mechs[n].noise_std for n in cf.ENDO}) / rthr
        typ_a = typ.iloc[lo:t1]; sl = slice(on, s1)
        mar = pd.Series({p: np.nanmean(np.abs(a[p][0, sl] - typ_a[p].to_numpy()[sl])) / df[p].std() for p in cf.ENDO})
        rows.append({"fault": fault, "node": node, "day": df.index[t0].date(), "true_effect_K": true_eff,
                     "phi_fault_K": float(phi[node]), "phi_room_K": float(phi[cf.T]), "dev_K": float(np.mean(dev[win])),
                     "top_phi_component": phi[COMPONENTS].abs().idxmax(), "top_phi_all": phi.abs().idxmax(),
                     "top_resid": z.idxmax(), "top_marginal": mar.idxmax()})
    print(df.index[t0].date(), round(time.time() - t), "s", flush=True)
V = pd.DataFrame(rows); V.to_pickle("validation_kelvin.pkl")
out = []
for f, g in V.groupby("fault", sort=False):
    out.append({"fault": f, "true effect (K)": g.true_effect_K.mean(), "Shapley of faulty node (K)": g.phi_fault_K.mean(),
                "abs error (K)": (g.phi_fault_K - g.true_effect_K).abs().mean(),
                "component top-1 (Shapley)": (g.top_phi_component == g.node).mean(),
                "top-1 residual": (g.top_resid == g.node).mean(), "top-1 marginal": (g.top_marginal == g.node).mean()})
O = pd.DataFrame(out).set_index("fault"); O.to_pickle("validation_kelvin_summary.pkl")
pd.set_option("display.width", 220); print(O.round(3).to_string())
print("correlation Shapley vs true effect over all cases:", round(np.corrcoef(V.phi_fault_K, V.true_effect_K)[0, 1], 3))
