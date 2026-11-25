import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prep import fm, DATA, RES, FIG, FIG_PREFIX


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

FIG.mkdir(parents=True, exist_ok=True)

SURF, T1, T2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
GREY = "#8a8984"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": T2, "xtick.color": T2,
                     "ytick.color": T2, "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False})
MODE_COL = {"beam_water_air": AQUA, "beam_air": VIOLET, "off_no_sun": BLUE, "off_sunny": YELLOW, "other": GREY}
MODE_LAB = {"beam_water_air": "Beam water + air", "beam_air": "Air only", "off_no_sun": "Off, no sun",
            "off_sunny": "Off, sunny", "other": "Transitions (other)"}
METH_LAB = {"switched_xgb_other": "Switched XGBoost", "global_xgb": "Global XGBoost",
            "persistence": "Persistence"}
METH_STYLE = [("persistence", GREY, "--", 1.6), ("global_xgb", BLUE, "-", 1.6), ("switched_xgb_other", AQUA, "-", 2.2)]


def title(ax, t):
    ax.set_title(t, loc="left", color=T1, fontweight="bold", fontsize=12)


# def _rmse_bars(ax, csv):
#     """Overall RMSE per method"""
#     s = pd.read_csv(csv, index_col=0)
#     meths = [k for k, *_ in METH_STYLE]
#     cols = [c for _, c, *_ in METH_STYLE]
#     ax.barh([METH_LAB[m] for m in meths], s.loc[meths, "rmse_K"], color=cols, height=0.6)
#     for yy, v in enumerate(s.loc[meths, "rmse_K"]):
#         ax.text(v, yy, f" {v:.2f}", va="center", color=T1)
#     ax.set_xlabel("Overall RMSE of indoor temperature (K), test year 2024")
#     ax.set_xlim(0, 1.15 * s.loc[meths, "rmse_K"].max())
#     ax.grid(axis="y", visible=False)


# def fig_performance():
#     rec = pd.read_pickle(RES / "recursive3h_test.pkl")
#     fr = pd.read_pickle(RES / "freerun_test.pkl")
#     fig, ax = plt.subplots(1, 2, figsize=(14, 4))
#     _rmse_bars(ax[0], RES / "recursive3h_test.csv")
#     title(ax[0], f"3 h recursive forecast ({len(rec['starts'])} starts, every 30 min)")
#     _rmse_bars(ax[1], RES / "freerun_test.csv")
#     ax[1].set_yticklabels([])
#     title(ax[1], f"24 h free-run simulation ({len(fr['starts'])} days, from midnight)")
#     fig.tight_layout(); fig.savefig(FIG / f"{FIG_PREFIX}1_performance.png", dpi=130); plt.close(fig)


def fig_examples():
    rec = pd.read_pickle(RES / "recursive3h_test.pkl")
    st = pd.DatetimeIndex(rec["starts"])
    H = rec["obs"].shape[1]
    step = pd.Timedelta("30min")
    # measured temperature: the first step of every forecast covers each 30-min slot once
    meas = pd.Series(rec["obs"][:, 0], index=st + step)
    ahead = {k: pd.Series(rec[k][:, -1], index=st + H * step) for k in ("switched_xgb_other", "global_xgb")}
    weeks = {"Summer week, June 2024": ("2024-06-10", 7), "Summer week, July 2024": ("2024-07-08", 7)}
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4), sharey=False)
    for j, (name, (d0, n)) in enumerate(weeks.items()):
        t0 = pd.Timestamp(d0); t1 = t0 + pd.Timedelta(days=n)
        grid = pd.date_range(t0 + step, t1, freq=step)
        ax[j].plot(grid, meas.reindex(grid).values, color=T1, lw=1.6, label="Measured")
        ax[j].plot(grid, ahead["switched_xgb_other"].reindex(grid).values, color=AQUA, lw=1.8,
                   label="Switched XGBoost")
        ax[j].plot(grid, ahead["global_xgb"].reindex(grid).values, color=BLUE, lw=1.1, ls="--",
                   label="Global XGBoost")
        ax[j].xaxis.set_major_locator(mdates.DayLocator()); ax[j].xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
        ax[j].set_ylabel("Indoor temperature (°C)"); title(ax[j], name)
    fig.legend(*ax[0].get_legend_handles_labels(), frameon=False, loc="upper center", ncol=3, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / f"{FIG_PREFIX}_recursive3h_examples.png", dpi=300)
    plt.close(fig)


# def fig_attribution():
#     by_mode = pd.read_csv(RES / "mode_K_per_day.csv", index_col=0).loc[["winter", "spring", "summer", "autumn"]]
#     by_obs = pd.read_csv(RES / "mode_K_per_day_measured.csv", index_col=0).loc[by_mode.index]
#     absc = pd.read_csv(RES / "component_abs_Kph_by_mode.csv", index_col=0)
#     D = pd.read_pickle(RES / "decomposition.pkl")
#     raw = fm.load(DATA).reindex(D.index)
#     Q = raw.cooling_flow / 60 * 4186 * (raw.KB02_return - raw.KB02_supply)

#     fig = plt.figure(figsize=(15, 9.2)); gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
#     ax = fig.add_subplot(gs[0, :])
#     modes = ["beam_water_air", "beam_air", "off_no_sun", "off_sunny", "other"]
#     x = np.arange(len(by_mode)); w = 0.16
#     for i, m in enumerate(modes):
#         ax.bar(x + (i - 2) * w, by_mode[m], width=w - 0.02, color=MODE_COL[m], label=MODE_LAB[m])
#         ax.scatter(x + (i - 2) * w, by_obs[m], marker="_", s=180, linewidths=2, color=T1, zorder=4,
#                    label="Measured" if i == 0 else None)
#     ax.axhline(0, color=T2, lw=0.8); ax.set_xticks(x, [s.capitalize() for s in by_mode.index])
#     ax.set_ylabel("K per day"); ax.grid(axis="x", visible=False)
#     ax.legend(frameon=False, ncol=6, loc="upper left", fontsize=8)
#     title(ax, "Which modes move the room: temperature change accumulated in each mode, K per day (bars = model, black = measured)")

#     ax = fig.add_subplot(gs[1, 0])
#     comps = ["chilled_beam", "building_mass", "solar", "envelope", "occupancy", "schedule"]
#     M = absc.loc[modes, comps]
#     im = ax.imshow(M.values, cmap=matplotlib.colors.ListedColormap(plt.cm.Blues(np.linspace(0.05, 0.95, 256))),
#                    aspect="auto", vmin=0)
#     ax.set_xticks(range(len(comps)), [c.replace("_", " ") for c in comps], rotation=0)
#     ax.set_yticks(range(len(modes)), [MODE_LAB[m] for m in modes]); ax.grid(False)
#     for i in range(M.shape[0]):
#         for j in range(M.shape[1]):
#             v = M.values[i, j]
#             ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="white" if v > 0.09 else T1, fontsize=8)
#     cb = fig.colorbar(im, ax=ax, fraction=0.04); cb.set_label("K/h", color=T2); cb.outline.set_visible(False)
#     title(ax, "Which components move the room within each mode (mean |contribution|)")

#     ax = fig.add_subplot(gs[1, 1])
#     s = (D["mode"] == "beam_water_air") & Q.notna()
#     q = pd.qcut(Q[s], 12, duplicates="drop")
#     g = pd.DataFrame({"Q": Q[s], "c": D.loc[s, "chilled_beam"] * 2}).groupby(q, observed=True)
#     m = g.agg(Q=("Q", "mean"), c=("c", "mean"), sd=("c", "std"), n=("c", "size"))
#     ax.errorbar(m.Q, m.c, yerr=1.96 * m.sd / np.sqrt(m.n), fmt="o", color=AQUA, ms=6, mec=SURF, elinewidth=1.2)
#     r = np.corrcoef(Q[s], D.loc[s, "chilled_beam"])[0, 1]
#     ax.axhline(0, color=T2, lw=0.8)
#     ax.text(0.98, 0.94, f"r = {r:.2f} (all steps)", transform=ax.transAxes, ha="right", color=T1)
#     ax.set_xlabel("Measured chilled-beam power, flow × cp × ΔT (W)  — not a model input")
#     ax.set_ylabel("Model's chilled-beam contribution (K/h)")
#     title(ax, "Check: the beam contribution follows the measured beam power")
#     fig.tight_layout(); fig.savefig(FIG / f"{FIG_PREFIX}3_attribution.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    # fig_performance()
    fig_examples()
    # fig_attribution()
    print("figures ->", FIG)
