import os
import warnings; warnings.filterwarnings("ignore")
import pickle, copy, numpy as np, pandas as pd
import causal_fdd as cf
df = cf.load_data(os.environ.get("FDD_DATA", "dfs.pkl")); mechs, test = pickle.load(open("mechs.pkl", "rb"))
arrs = cf.arrays(df); u = cf.abduct(mechs, arrs)
win = [w for w in cf.day_windows(df.index) if test[w[0]] and df.index[w[0]].month in [6,7,8] and df.workday.iloc[w[0]] == 1]
print(len(win), "held-out summer workdays")
hours = df.index.hour.to_numpy()
def scen(name):
    """returns (modified arrays for exogenous change, fault dict for endogenous change)"""
    a = {k: v.copy() for k, v in arrs.items()}; f = None
    if name == "Occupancy +300 ppm (09-17)":
        m = (hours >= 9) & (hours < 17); a["co2"][0, m] += 300
    elif name == "Solar +300 W/m2 (10-16)":
        m = (hours >= 10) & (hours < 16); a["solar_radiation"][0, m] += 300
    elif name == "Outdoor +5 K (all day)":
        a["outdoor_temperature"][0] += 5
    elif name == "Chilled-water supply +2 K (all day)":
        f = {"KB02_supply": lambda t, x: x + 2}
    elif name == "AHU supply air +2 K (all day)":
        f = {"AHU_supply_temperature": lambda t, x: x + 2}
    return a, f
rows = []
for name in ["Occupancy +300 ppm (09-17)", "Solar +300 W/m2 (10-16)", "Outdoor +5 K (all day)", "Chilled-water supply +2 K (all day)", "AHU supply air +2 K (all day)"]:
    a, f = scen(name)
    dT_open, dT_closed, dV, dD = [], [], [], []
    for t0, t1 in win:
        ones = np.ones((1, 7))
        base = cf.simulate(mechs, arrs, u, t0, t1, ones)                        # factual (reproduces data)
        closed = cf.simulate(mechs, a, u, t0, t1, ones, faults=f)               # everything responds
        # open loop: actuators and plant frozen at factual values (only the disturbance itself changes)
        a_open = {k: v.copy() for k, v in a.items()}
        if f:
            node = list(f)[0]; a_open[node] = a_open[node].copy(); a_open[node][0, t0:t1] = f[node](0, a_open[node][0, t0:t1])
        opn = cf.simulate(mechs, a_open, u, t0, t1, ones[:, :1], order=[cf.T])
        sl = slice(t0 + 24, t0 + 34)   # 12:00-17:00
        dT_open.append(np.mean(opn[cf.T][0, sl] - base[cf.T][0, sl])); dT_closed.append(np.mean(closed[cf.T][0, sl] - base[cf.T][0, sl]))
        dV.append(np.mean(closed["cooling_valve"][0, sl] - base["cooling_valve"][0, sl])); dD.append(np.mean(closed["VAV_damper"][0, sl] - base["VAV_damper"][0, sl]))
    o, c = np.mean(dT_open), np.mean(dT_closed)
    rows.append({"Disturbance": name, "Room, controllers frozen (K)": o, "Room, controllers respond (K)": c,
                 "Hidden by control (%)": 100 * (1 - c / o) if abs(o) > 1e-3 else np.nan,
                 "Cooling valve change (%-pt)": np.mean(dV), "Damper change (%-pt)": np.mean(dD)})
R = pd.DataFrame(rows).set_index("Disturbance"); print(R.round(2).to_string()); R.to_pickle("compensation.pkl")
