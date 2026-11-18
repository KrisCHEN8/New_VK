import os
import warnings; warnings.filterwarnings("ignore")
import pickle, copy, numpy as np, pandas as pd, statsmodels.api as sm
import causal_fdd as cf
from plotstyle import *
pd.set_option("display.width", 220)
df = cf.load_data(os.environ.get("FDD_DATA", "dfs.pkl")); mechs, test = pickle.load(open("mechs.pkl", "rb"))
arrs = cf.arrays(df); m = mechs[cf.T]; cols = list(m.specs)
X, y, _ = cf.node_data(m, arrs); ok = np.isfinite(X).all(1) & np.isfinite(y)
summer = np.isin(df.index.month, [5,6,7,8,9]); winter = np.isin(df.index.month, [12,1,2])
occ = (df.workday.values == 1) & (df.index.hour >= 9) & (df.index.hour < 17)
SHIFTS = {"Occupancy +300 ppm CO2": {"co2": lambda v: v + 300},
          "Solar +500 W/m2": {"solar_30min": lambda v: v + 500, "solar_2h": lambda v: v + 500},
          "Outdoor +5 K": {"dT_envelope": lambda v: v + 5},
          "Cooling valve +20 %-pt": {"cooling_valve": lambda v: np.clip(v + 20, 0, 100)},
          "Chilled-water supply +2 K": {"dT_chilled_water": lambda v: v - 2},
          "AHU supply air +2 K": {"dT_supply_air": lambda v: v - 2},
          "Damper +20 %-pt": {"VAV_damper": lambda v: np.clip(v + 20, 0, 100)}}
def scm_effects(model, sel):
    Xs = X[sel]; out = {}
    for k, ch in SHIFTS.items():
        Xa = Xs.copy()
        for c, f in ch.items(): Xa[:, cols.index(c)] = f(Xa[:, cols.index(c)])
        out[k] = 2 * np.mean(model.predict(Xa) - model.predict(Xs))
    return pd.Series(out)
# 1. SCM effects with day-block bootstrap (refit room mechanism)
days = df.index.normalize(); tr_days = np.unique(days[~test & ok])
rng = np.random.default_rng(0); boot = {"summer": [], "winter": []}; strength_boot = []
C_cols = list(m.groups)
for b in range(20):
    pick = rng.choice(tr_days, len(tr_days), replace=True)
    w = pd.Series(pick).value_counts()
    weight = pd.Series(days).map(w).fillna(0).to_numpy()
    sel_tr = ~test & ok & (weight > 0)
    mb = copy.deepcopy(m); mb.model = mb.make_model().fit(X[sel_tr], y[sel_tr], sample_weight=weight[sel_tr])
    boot["summer"].append(scm_effects(mb.model, ok & summer & occ)); boot["winter"].append(scm_effects(mb.model, ok & winter & occ))
    C = cf.group_contributions(mb, X[ok]); yy = y[ok]
    strength_boot.append(pd.concat({s: C.loc[msk[ok], C_cols].var() / yy[msk[ok]].var() for s, msk in [("summer", summer), ("winter", winter)]}, axis=1))
eff = {}
for s in ["summer", "winter"]:
    B = pd.concat(boot[s], axis=1)
    eff[(s, "SCM")] = scm_effects(m.model, ok & (summer if s == "summer" else winter) & occ)
    eff[(s, "SCM 95% CI")] = B.quantile(0.025, axis=1).round(3).astype(str) + " to " + B.quantile(0.975, axis=1).round(3).astype(str)
# 2. linear heat balance with clock fixed effects (model-free-ish triangulation)
L = lambda c, k=1: df[c].shift(k); Tl = L(cf.T)
d = pd.DataFrame({"dT": (df[cf.T] - Tl) * 2, "co2": L("co2"), "solar": df.solar_radiation.rolling(4).mean().shift(1),
    "env": df.outdoor_temperature.rolling(6).mean().shift(1) - Tl, "cv": L("cooling_valve"), "kb": L("KB02_supply"), "ahu": L("AHU_supply_temperature"),
    "dmp": L("VAV_damper"), "Tl": Tl}, index=df.index)
d["water"] = d.cv / 100 * (d.Tl - d.kb); d["air"] = d.dmp / 100 * (d.Tl - d.ahu)
slot = (df.index.hour * 2 + df.index.minute // 30).astype(str) + "_" + df.workday.astype(int).astype(str)
for s, msk in [("summer", summer), ("winter", winter)]:
    q = d[msk].dropna(); fe = pd.get_dummies(slot[q.index], drop_first=True).astype(float)
    r = sm.OLS(q.dT, sm.add_constant(pd.concat([q[["co2", "solar", "env", "water", "air", "Tl"]], fe.loc[q.index]], axis=1))).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    o = q[(df.workday[q.index] == 1) & (q.index.hour >= 9) & (q.index.hour < 17)]
    lin = {"Occupancy +300 ppm CO2": 300 * r.params.co2, "Solar +500 W/m2": 500 * r.params.solar, "Outdoor +5 K": 5 * r.params.env,
           "Cooling valve +20 %-pt": r.params.water * 0.2 * (o.Tl - o.kb).mean(), "Chilled-water supply +2 K": -r.params.water * 2 * (o.cv / 100).mean(),
           "AHU supply air +2 K": -r.params.air * 2 * (o.dmp / 100).mean(), "Damper +20 %-pt": r.params.air * 0.2 * (o.Tl - o.ahu).mean()}
    se = {"Occupancy +300 ppm CO2": 300 * r.bse.co2, "Solar +500 W/m2": 500 * r.bse.solar, "Outdoor +5 K": 5 * r.bse.env}
    eff[(s, "Linear + clock FE")] = pd.Series(lin)
    eff[(s, "Linear SE")] = pd.Series(se)
E = pd.DataFrame(eff)
print(E.round(3).to_string())
E.to_pickle("effects.pkl")
S = pd.concat(strength_boot, keys=range(20))
strength = pd.concat({"mean": S.groupby(level=1).mean(), "lo": S.groupby(level=1).quantile(0.025), "hi": S.groupby(level=1).quantile(0.975)}, axis=1)
print(strength.round(3).to_string()); strength.to_pickle("strength.pkl")
