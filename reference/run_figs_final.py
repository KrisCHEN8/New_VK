import os
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np, pandas as pd
from scipy.stats import binom
import causal_fdd as cf
from plotstyle import *
import matplotlib.dates as mdates
df = cf.load_data(os.environ.get("FDD_DATA", "dfs.pkl")); mechs, test = pickle.load(open("mechs.pkl", "rb"))
R = pd.read_pickle("episodes_year.pkl"); dev = np.load("dev4h.npy"); cal = pickle.load(open("calibration.pkl", "rb"))
u = cf.abduct(mechs, cf.arrays(df))
heat = pd.Series(u[cf.T] * 2, index=df.index)            # unexplained heat, K/h
sep = (R.start >= pd.Timestamp("2024-08-30", tz=cf.TZ)).sum()
print("episodes Aug30-Sep30:", sep, "of", len(R), "| P(X>=k) at 5%/day over 32 days:", binom.sf(sep - 1, 32, 0.05))
# Figure A: year
f, (a1, a2) = fig(2, 1, figsize=(11, 5.4), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
wk = heat.rolling(48 * 7, center=True, min_periods=48 * 3).mean()
a1.axhline(0, color=INK2, lw=0.8)
a1.plot(wk.index, wk, color=PAL[0], lw=2)
for _, r in R.iterrows():
    a1.plot([r.start, r.start], [0.17, 0.19], color=PAL[1], lw=2)
a1.text(R.start.min(), 0.205, "detected temperature episodes", fontsize=8, color=INK2)
a1.axvspan(pd.Timestamp("2024-08-30", tz=cf.TZ), pd.Timestamp("2024-10-01", tz=cf.TZ), color=PAL[1], alpha=0.08, lw=0)
a1.annotate("From 30 Aug: +0.13 K/h of heat\nthe model cannot explain", xy=(pd.Timestamp("2024-09-10", tz=cf.TZ), 0.12),
            xytext=(pd.Timestamp("2024-06-01", tz=cf.TZ), 0.12), fontsize=8.5, color=INK, va="center",
            arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1))
style(a1, "Unexplained room heat (room mechanism residual), 7-day mean", None, "K per hour")
a1.set_ylim(-0.08, 0.23)
occ = df[(df.workday == 1) & (df.index.hour >= 9) & (df.index.hour < 17)]
sat = (occ.cooling_valve > 95).resample("W").mean() * 100
a2.bar(sat.index, sat, width=5, color=PAL[2])
style(a2, "Cooling valve saturated (>95 %), share of occupied hours per week", None, "%")
a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y", tz=cf.TZ))
f.tight_layout(); f.savefig("fig_year.png", dpi=170)
# Figure B: case study, Tue 24 Sep 2024
ep = R[R.start.dt.date == pd.Timestamp("2024-09-24").date()].iloc[0]
day = slice("2024-09-24 00:00", "2024-09-24 23:30")
d = df[day]; pred = pd.Series(df[cf.T].to_numpy() - dev, index=df.index)[day]
f, axes = fig(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [2.1, 1]})
ax = axes[0]
ax2 = ax.twinx(); ax2.fill_between(d.index, d.cooling_valve, color=PAL[2], alpha=0.18, lw=0, step="mid")
ax2.set_ylim(0, 300); ax2.set_yticks([0, 50, 100]); ax2.tick_params(colors=INK2, labelsize=8); ax2.set_ylabel("Cooling valve (%)", fontsize=8.5, color=INK2)
ax2.spines["top"].set_visible(False)
ax.plot(d.index, d[cf.T], color=INK, lw=1.8, label="Observed room")
ax.plot(pred.index, pred, color=PAL[0], lw=1.8, ls="--", label="Predicted 4 h ahead, all components normal")
ax.axvspan(ep.start, ep.end + pd.Timedelta(minutes=30), color=PAL[1], alpha=0.10, lw=0)
ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
style(ax, "Tue 24 Sep 2024: room 1.4 K above prediction while the valve is at 100 %", None, "Room temperature (deg C)")
ax.legend(frameon=False, fontsize=8, loc="upper right"); ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=cf.TZ))
ph = pd.Series({cf.LABELS[c[4:]]: ep[c] for c in R if c.startswith("phi_")}).sort_values()
axes[1].barh(ph.index, ph.values, color=[PAL[1] if v > 0 else PAL[0] for v in ph.values], height=0.6)
for i, v in enumerate(ph.values): axes[1].text(v + (0.03 if v >= 0 else -0.03), i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8, color=INK)
axes[1].axvline(0, color=INK2, lw=0.8); axes[1].set_xlim(-0.3, 1.6)
style(axes[1], f"Who caused the {ep.mean_dev:.2f} K excess ({ep.start:%H:%M}-{ep.end + pd.Timedelta(minutes=30):%H:%M})", "Shapley value (K)", None)
f.tight_layout(); f.savefig("fig_case.png", dpi=170)
# Figure C: validation scatter
V = pd.read_pickle("validation_kelvin.pkl")
f, ax = fig(figsize=(6.4, 5.2))
names = list(dict.fromkeys(V.fault)); cols = PAL + ["#8b5cf6"]
for n, c in zip(names, cols):
    g = V[V.fault == n]; ax.scatter(g.true_effect_K, g.phi_fault_K, s=26, color=c, label=n, edgecolor="white", lw=0.6, zorder=3)
lim = [-0.1, 1.2]; ax.plot(lim, lim, color=INK2, lw=1, ls="--"); ax.set_xlim(lim); ax.set_ylim(lim)
style(ax, "Shapley value of the faulty node vs its true effect", "True effect of the fault on the room, 12:00-18:00 (K)", "Shapley value of the faulty node (K)")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
ax.text(0.65, 0.05, f"140 simulated faulty days\nr = {np.corrcoef(V.phi_fault_K, V.true_effect_K)[0,1]:.2f}", fontsize=8.5, color=INK, transform=ax.transAxes)
f.tight_layout(); f.savefig("fig_validation.png", dpi=170)
print("ok")
