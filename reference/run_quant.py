import os
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np, pandas as pd
import causal_fdd as cf
from plotstyle import *
pd.set_option("display.width", 220)
df = cf.load_data(os.environ.get("FDD_DATA", "dfs.pkl")); mechs, test = pickle.load(open("mechs.pkl", "rb"))
arrs = cf.arrays(df); m = mechs[cf.T]
X, y, _ = cf.node_data(m, arrs); ok = np.isfinite(X).all(1)
C = cf.group_contributions(m, X[ok]); C.index = df.index[ok]
C = C * 2   # K per hour
groups = list(m.groups) + ["Constant gain"]
season = pd.Series(np.select([df.index.month.isin([6, 7, 8]), df.index.month.isin([12, 1, 2])], ["summer", "winter"], "other"), index=df.index)
C["season"] = season[ok].values; C["workday"] = df["workday"][ok].values; C["hour"] = C.index.hour + C.index.minute / 60
C.to_pickle("contributions.pkl")
# 1. daily heat balance profiles
panels = [("summer", 1, "Summer workday (Jun-Aug)"), ("summer", 0, "Summer weekend"), ("winter", 1, "Winter workday (Dec-Feb)")]
f, axes = fig(1, 3, figsize=(13, 4.2), sharey=True)
for ax, (s, w, title) in zip(axes, panels):
    sub = C[(C.season == s) & (C.workday == w)].groupby("hour")[groups].mean()
    for g in groups:
        ax.plot(sub.index, sub[g], color=GROUP_COLOR[g], lw=2, label=g)
    ax.axhline(0, color=INK2, lw=0.8)
    style(ax, title, "Local hour", "Effect on room temperature (K per hour)" if ax is axes[0] else None)
    ax.set_xticks(range(0, 25, 6))
h, l = axes[0].get_legend_handles_labels(); f.legend(h, l, frameon=False, fontsize=8, loc="lower center", ncol=8, bbox_to_anchor=(0.5, -0.01))
f.suptitle("Room heat balance by cause (SCM decomposition, mean over days)", x=0.01, ha="left", fontsize=11.5, color=INK)
f.tight_layout(rect=(0, 0.06, 1, 1)); f.savefig("fig_heat_balance.png", dpi=170)
# table: daily totals (K per day) and occupied-hour means
tab = {}
for s, w, title in panels:
    sub = C[(C.season == s) & (C.workday == w)]
    occ = sub[(sub.hour >= 8) & (sub.hour < 18)]
    tab[(title, "occupied mean K/h")] = occ[groups].mean()
print(pd.DataFrame(tab).round(3).to_string())
# 2. strength = variance of each group's contribution, share of variance of the room's change; bootstrap over days
def strengths(Cdf):
    v = Cdf[groups[:-1]].var() / (y[ok][Cdf.index.isin(Cdf.index)].var() if False else 1)
    return v
yy = pd.Series(y[ok] * 2, index=df.index[ok])
res = {}
for s in ["summer", "winter"]:
    sub = C[C.season == s]; tot = yy[sub.index].var()
    res[s] = sub[groups[:-1]].var() / tot
print("Strength (Var of contribution / Var of room change):"); print(pd.DataFrame(res).round(3).to_string())
pickle.dump(res, open("strength_point.pkl", "wb"))
