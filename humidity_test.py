import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import conditional_dist as cd


# tunable parameters: conditional_dist.CONFIG plus the overrides/additions below
CONFIG = cd._merge_config({
    "target_col": "dev",          # bin residual = dev - bin_median
    "make_plots": False,

    # outdoor RH from SMHI (dfs["humidity"] is supply-air RH, RH_sup)
    "rh_out_csv": "./smhi_humidity.csv",
    "rh_out_col": "RH_out",
    "pressure": 101325.0,         # Pa
    "solar_acc_halflife": "4h",
    "cool_base": 15.0,            # °C, cool_need = max(T_out_3h - cool_base, 0)
    "x_quantiles": (0.10, 0.90),
    "n_boot": 1000,
    "seed": 0,
})

FULL = "dev ~ T_out_3h + solar_acc + x_out + x_out:cool_need + sin_h + cos_h"
BASE = "dev ~ T_out_3h + solar_acc + sin_h + cos_h"
COLORS = {"without humidity": "#2a78d6", "with humidity": "#eb6834"}


def read_smhi_csv(path, freq="30min"):
    """
    SMHI export (date;time;value;quality, naive UTC) interpolated onto a 30 min grid,
    as read_smhi_csv in data_preprocessing.ipynb (which built dfs['outdoor_temperature']).
    """
    raw = pd.read_csv(path, sep=";", decimal=",", usecols=[0, 1, 2])
    ts = pd.to_datetime(raw.iloc[:, 0].astype(str) + " " + raw.iloc[:, 1].astype(str),
                        errors="coerce")
    s = pd.Series(pd.to_numeric(raw.iloc[:, 2], errors="coerce").to_numpy(), index=ts)
    s = s[s.index.notna()].sort_index()
    s = s[~s.index.duplicated(keep="first")]
    grid = pd.date_range(s.index.min().ceil(freq), s.index.max().floor(freq), freq=freq)
    s = s.reindex(s.index.union(grid)).interpolate(method="time").reindex(grid)
    return pd.DataFrame({"value": s})


def with_outdoor_rh(dfs, config=None):
    """dfs plus the SMHI outdoor RH, and a config that carries it through combine/preprocess."""
    cfg = cd._merge_config(config, CONFIG)
    dfs = {**dfs, "outdoor_humidity": read_smhi_csv(cfg["rh_out_csv"], cfg["resample_freq"])}
    cfg["key_map"] = {**cfg["key_map"], "outdoor_humidity": cfg["rh_out_col"]}
    cfg["context_cols"] = cfg["context_cols"] + [cfg["rh_out_col"]]
    return dfs, cfg


def moisture_content(T, RH, p=101325.0):
    """Moisture content x [g/kg dry air] from T [°C] and RH [%] (Magnus, over water)."""
    p_ws = 611.2 * np.exp(17.62 * T / (243.12 + T))   # Pa
    p_w = RH / 100.0 * p_ws
    return 622.0 * p_w / (p - p_w)


def add_model_columns(data, config=None):
    cfg = cd._merge_config(config, CONFIG)
    data = data.copy()
    data["x_out"] = moisture_content(data["T_out"], data[cfg["rh_out_col"]], cfg["pressure"])
    # on the full series, so the history before 08:00 counts
    data["solar_acc"] = data["solar"].ewm(halflife=cfg["solar_acc_halflife"], times=data.index).mean()
    data["cool_need"] = (data["T_out_3h"] - cfg["cool_base"]).clip(lower=0)
    hour = data.index.hour + data.index.minute / 60
    data["sin_h"] = np.sin(2 * np.pi * hour / 24)
    data["cos_h"] = np.cos(2 * np.pi * hour / 24)
    return data


def model_frame(data):
    """Occupied, not in recovery, all model variables present."""
    cols = ["dev", "T_out_3h", "solar_acc", "x_out", "cool_need", "sin_h", "cos_h"]
    ok = data["occupied"] & ~data["recovery"] & data[cols].notna().all(axis=1)
    return data.loc[ok]


def fit(formula, d, maxlags):
    return smf.ols(formula, data=d).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})


# Humidity effect
def humidity_effect(res, d, q=(0.10, 0.90)):
    """Per-row predicted dev with x_out at its q[1] quantile minus at its q[0] quantile."""
    lo, hi = d["x_out"].quantile(list(q))
    return res.predict(d.assign(x_out=hi)) - res.predict(d.assign(x_out=lo)), (lo, hi)


def block_bootstrap(d, config=None):
    """Mean humidity effect refitted on whole days resampled with replacement."""
    cfg = cd._merge_config(config, CONFIG)
    rng = np.random.default_rng(cfg["seed"])
    X = np.column_stack([np.ones(len(d)), d["T_out_3h"], d["solar_acc"], d["x_out"],
                         d["x_out"] * d["cool_need"], d["sin_h"], d["cos_h"]])
    y = d["dev"].to_numpy()
    x, cn = d["x_out"].to_numpy(), d["cool_need"].to_numpy()
    day_rows = list(pd.Series(np.arange(len(d))).groupby(d.index.normalize()).apply(np.array))
    q_lo, q_hi = cfg["x_quantiles"]

    out = np.empty(cfg["n_boot"])
    for b in range(cfg["n_boot"]):
        idx = np.concatenate([day_rows[i] for i in rng.integers(0, len(day_rows), len(day_rows))])
        beta = np.linalg.lstsq(X[idx], y[idx], rcond=None)[0]
        lo, hi = np.quantile(x[idx], [q_lo, q_hi])
        # only the x_out and x_out:cool_need terms change between the two predictions
        out[b] = np.mean((hi - lo) * (beta[3] + beta[4] * cn[idx]))
    return out


# Refit-based flags
def robust_flags(resid, cfg):
    mad = (resid - resid.median()).abs().median()
    z = resid / (cfg["mad_scale"] * mad)
    flag = (z.abs() > cfg["z_threshold"]) & (resid.abs() > cfg["min_abs_residual"])
    return z, flag, mad


def event_table(data, events, d, flags, resids):
    """Per occupied binning event: binning-flagged samples and how many each model still flags."""
    rows = []
    for eid, ev in events[events["regime"] == "occupied"].iterrows():
        idx = data.index[(data["event_id"] == eid) & data["anomaly"]]
        sign = 1 if ev["direction"] == "too warm" else -1
        in_d = idx.intersection(d.index)
        row = {"event_id": eid, "start": ev["start"], "duration": ev["duration"],
               "direction": ev["direction"], "max_abs_dev": ev["max_abs_dev"],
               "score": ev["score"], "n_flagged_bin": len(idx), "n_in_model": len(in_d)}
        for name in flags:
            same_dir = np.sign(resids[name].loc[in_d]) == sign
            row[f"n_{name}"] = int((flags[name].loc[in_d] & same_dir).sum())
            row[f"stays_{name}"] = row[f"n_{name}"] > 0
        rows.append(row)
    return pd.DataFrame(rows).set_index("event_id")


# Plots
def plot_residuals_vs_x(d, resids, flags):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    edges = np.quantile(d["x_out"], np.linspace(0, 1, 13))
    bins = pd.cut(d["x_out"], edges, include_lowest=True)
    for ax, (name, r) in zip(axes, resids.items()):
        color = COLORS[name]
        ax.scatter(d["x_out"], r, s=4, color=color, alpha=0.25, linewidths=0,
                   label="occupied samples")
        f = flags[name]
        ax.scatter(d.loc[f, "x_out"], r[f], s=18, facecolors="none", edgecolors="#222222",
                   linewidths=0.8, label=f"flagged (n={int(f.sum())})")
        med = r.groupby(bins, observed=True).median()
        mid = d["x_out"].groupby(bins, observed=True).median()
        ax.plot(mid, med, color="#222222", lw=2, marker="o", ms=4, label="binned median")
        ax.axhline(0, color="#808080", lw=0.8)
        ax.set(xlabel="x_out, outdoor moisture content [g/kg]", title=f"Model {name}")
        ax.grid(True, color="#e5e5e5", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    axes[0].set_ylabel("OLS residual of dev [°C]")
    fig.tight_layout()
    return fig


def plot_bootstrap(boot, point, ci):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(boot, bins=40, color=COLORS["with humidity"], edgecolor="white", lw=0.5)
    ax.axvspan(ci[0], ci[1], color="#808080", alpha=0.15, lw=0,
               label=f"95 % CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    for v in ci:
        ax.axvline(v, color="#555555", lw=1, ls="--")
    ax.axvline(point, color="#222222", lw=2, label=f"point estimate {point:+.3f} °C")
    ax.axvline(0, color="#808080", lw=0.8)
    ax.set(xlabel="mean humidity effect, dev(x_out p90) − dev(x_out p10) [°C]",
           ylabel="bootstrap replicates",
           title=f"Day-block bootstrap of the humidity effect ({len(boot)} replicates)")
    ax.grid(True, axis="y", color="#e5e5e5", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def run_analysis(dfs, config=None):
    dfs, cfg = with_outdoor_rh(dfs, config)

    # existing pipeline: regimes, recovery, bins, anomaly flags, events
    data, _ = cd.combine_data(dfs, cfg)
    data = cd.preprocess(data, cfg)
    data = cd.assign_bins(data, cfg)
    bin_stats = cd.compute_bin_stats(data, cfg)
    data = cd.flag_anomalies(data, bin_stats, cfg)
    data, events = cd.detect_events(data, cfg)

    data = add_model_columns(data, cfg)
    d = model_frame(data)
    maxlags = int(pd.Timedelta("1D") / pd.Timedelta(cfg["resample_freq"]))
    print(f"\nModel sample: {len(d)} occupied, non-recovery rows on "
          f"{d.index.normalize().nunique()} days; HAC maxlags = {maxlags}")
    print(f"x_out [g/kg]: " + ", ".join(f"{k} {v:.2f}" for k, v in
                                          d["x_out"].describe()[["min", "50%", "max"]].items()))

    # 1. OLS fits
    res = {"with humidity": fit(FULL, d, maxlags), "without humidity": fit(BASE, d, maxlags)}
    for name, r in res.items():
        print("\n" + "=" * 78 + f"\nModel {name}\n" + "=" * 78)
        print(r.summary())
    wt = res["with humidity"].wald_test("x_out = 0, x_out:cool_need = 0", scalar=True)
    print(f"\nJoint HAC Wald test, both humidity terms = 0: F = {wt.statistic:.2f}, "
          f"p = {wt.pvalue:.3g}")
    print(f"Adj. R²: without {res['without humidity'].rsquared_adj:.3f}, "
          f"with {res['with humidity'].rsquared_adj:.3f}")

    # 2. effect size
    eff, (x_lo, x_hi) = humidity_effect(res["with humidity"], d, cfg["x_quantiles"])
    print(f"\nHumidity effect: x_out p10 = {x_lo:.2f} -> p90 = {x_hi:.2f} g/kg")
    print(f"  mean {eff.mean():+.3f} °C")
    print("  distribution over rows [°C]:")
    print(eff.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).round(3).to_string())
    print(f"  rows with cool_need > 0: {(d['cool_need'] > 0).mean():.1%}; mean effect there "
          f"{eff[d['cool_need'] > 0].mean():+.3f}, elsewhere {eff[d['cool_need'] == 0].mean():+.3f}")

    # 3. block bootstrap
    boot = block_bootstrap(d, cfg)
    ci = np.percentile(boot, [2.5, 97.5])
    print(f"\nBlock bootstrap ({cfg['n_boot']} x whole days): mean effect {eff.mean():+.3f} °C, "
          f"95 % CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")

    # 4. anomaly sizes
    warm = data["occupied"] & data["anomaly"] & (data["direction"] > 0)
    a = data.loc[warm, "residual"].abs()
    print(f"\nFlagged too-warm anomalies (occupied, n = {len(a)}): |dev - bin_median| "
          f"median {a.median():.3f} °C, p90 {a.quantile(0.9):.3f} °C")
    print(f"  mean humidity effect / median anomaly = {abs(eff.mean()) / a.median():.0%}; "
          f"effect at those samples: mean {eff.reindex(a.index).mean():+.3f} °C")

    # 5. refit-based flags
    resids, flags = {}, {}
    print("\nRobust z from model residuals (|z| > "
          f"{cfg['z_threshold']} and |residual| > {cfg['min_abs_residual']} °C):")
    for name in ["without humidity", "with humidity"]:
        resids[name] = res[name].resid
        _, flags[name], mad = robust_flags(resids[name], cfg)
        print(f"  {name:17s}: MAD {mad:.3f} °C, flagged {int(flags[name].sum())} samples")

    bin_flag = data["occupied"] & data["anomaly"]
    in_d = bin_flag[bin_flag].index.intersection(d.index)
    print(f"\nBinning-flagged occupied samples: {int(bin_flag.sum())} "
          f"({len(in_d)} in the model sample)")
    for name in flags:
        sign = data.loc[in_d, "direction"]
        kept = flags[name].loc[in_d] & (np.sign(resids[name].loc[in_d]) == sign)
        print(f"  still flagged (same direction), {name:17s}: {int(kept.sum())} "
              f"({kept.mean():.0%})")

    ev = event_table(data, events, d, flags, resids)
    print(f"\nOccupied binning events: {len(ev)}")
    for name in flags:
        print(f"  stay flagged (>= 1 sample, same direction), {name:17s}: "
              f"{int(ev[f'stays_{name}'].sum())} ({ev[f'stays_{name}'].mean():.0%})")
    top = ev.sort_values("score", ascending=False).head(cfg["top_n_events"]).copy()
    top["start"] = top["start"].dt.strftime("%Y-%m-%d %H:%M")
    cols = ["start", "duration", "direction", "max_abs_dev", "score", "n_flagged_bin",
            "n_without humidity", "n_with humidity", "stays_with humidity"]
    print(f"\nTop {cfg['top_n_events']} occupied binning events by score "
          "(n_* = binning-flagged samples still flagged by each model):")
    with pd.option_context("display.width", 200):
        print(top[cols].round(2).to_string())

    # 6. plots
    figure_dir = cd.Path("./figure")
    figure_dir.mkdir(parents=True, exist_ok=True)
    plot_residuals_vs_x(d, {k: resids[k] for k in ["without humidity", "with humidity"]}, flags) \
        .savefig(figure_dir / "humidity_residuals_vs_x_out.png", dpi=300, bbox_inches="tight")
    plot_bootstrap(boot, eff.mean(), ci) \
        .savefig(figure_dir / "humidity_effect_bootstrap.png", dpi=300, bbox_inches="tight")
    return data, d, res, eff, boot, ev


if __name__ == "__main__":
    import pickle

    with open("./07-D2-02_data/dfs.pkl", "rb") as f:
        dfs = pickle.load(f)

    run_analysis(dfs)
