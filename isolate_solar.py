import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

DATA = "./07-D2-02_data/dfs.pkl"
SUMMER = [6, 7, 8]
CLOSED = {"cooling_valve": 1, "heating_valve": 1, "VAV_damper": 1, "co2": 450}   # max over the past 2 h
WINDOW = 4         # 2 h of 30min steps
MAX_GAP = 3.0      # K; |outdoor - room| below this counts as "no envelope exchange"
GAP_SWEEP = [1, 2, 3, 4, 5]


def load(path=DATA):
    d = pd.read_pickle(path)
    df = pd.concat({k: v["value"].rename(k) for k, v in d.items()}, axis=1)
    df.index = df.index.tz_localize("UTC").tz_convert("Europe/Stockholm")
    return df.asfreq("30min")


def passive_frame(df):
    s = df[df.index.month.isin(SUMMER)]
    passive = np.logical_and.reduce([s[c].rolling(WINDOW).max() < v for c, v in CLOSED.items()])
    room = s["indoor_temperature"]
    frame = pd.DataFrame({
        "dT": room.diff() * 2,                                                       # K/h
        "dT_env": s["outdoor_temperature"].rolling(6).mean().shift(1) - room.shift(1),  # 3-h mean outdoor - room, K
        "sun": s["solar_radiation"].rolling(4).mean().shift(1) / 100,       # 2h mean, per 100 W/m2
        "date": s.index.date,
        "hour": s.index.hour + s.index.minute / 60,
    })
    return frame[passive].dropna()


def neutral_frame(frame, max_gap=MAX_GAP):
    """Passive half-hours where outdoor is close to room temperature, so only sun (and internal gains) remain."""
    return frame[frame["dT_env"].abs() < max_gap]


def fit(frame, formula="dT ~ dT_env + sun"):
    """OLS; standard errors clustered by day, since half-hours of one day are not independent."""
    return smf.ols(formula, frame).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(frame["date"])[0]})


def fit_sun_only(neutral):
    """dT ~ sun on the neutral subset; None when there are too few days to cluster."""
    return fit(neutral, "dT ~ sun") if neutral["date"].nunique() >= 3 else None


def sweep(frame, gaps=GAP_SWEEP):
    """Sun-only fit for several gap thresholds, to show how stable the estimate is."""
    rows = []
    for g in gaps:
        n = neutral_frame(frame, g)
        m = fit_sun_only(n)
        ci = m.conf_int().loc["sun"] if m is not None else [np.nan, np.nan]
        rows.append({"max_gap_K": g, "n": len(n), "days": n["date"].nunique(),
                     "sunny_n": int((n["sun"] > 0.5).sum()),
                     "sun_coef": m.params["sun"] if m is not None else np.nan,
                     "ci_low": ci[0], "ci_high": ci[1]})
    return pd.DataFrame(rows).set_index("max_gap_K")


def plot(frame, m, neutral=None, m_sun=None):
    """Partial effect of each cause: room change minus the other cause's fitted effect, binned means.
    Third panel: raw room change vs sun on the outdoor ~ room subset (no adjustment needed)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    for ax, x, other, unit in [(axes[0], "dT_env", "sun", "outdoor - room (K)"),
                               (axes[1], "sun", "dT_env", "sun, 2-h mean (100 W/m2)")]:
        y = frame["dT"] - m.params[other] * frame[other]
        bins = pd.qcut(frame[x], 10, duplicates="drop")
        binned = y.groupby(bins, observed=True).agg(["mean", "sem"]).assign(x=frame[x].groupby(bins, observed=True).mean())
        ax.scatter(frame[x], y, s=4, color="#1f3b73", alpha=0.15)
        ax.errorbar(binned["x"], binned["mean"], yerr=1.96 * binned["sem"], fmt="o", color="#ff9f1c", ms=5)
        xs = np.linspace(frame[x].min(), frame[x].max(), 50)
        ax.plot(xs, m.params["Intercept"] + m.params[x] * xs, color="#ff9f1c", lw=2)
        ax.axhline(0, color="#999999", lw=0.8)
        ax.set_xlabel(unit)
        ax.set_ylim(np.percentile(y, [1, 99]))
    axes[0].set_ylabel("room change (K/h)")

    ax = axes[2]
    if neutral is not None and len(neutral):
        sc = ax.scatter(neutral["sun"], neutral["dT"], s=14, c=neutral["dT_env"], cmap="coolwarm",
                        vmin=-MAX_GAP, vmax=MAX_GAP)
        fig.colorbar(sc, ax=ax, label="outdoor - room (K)")
        if m_sun is not None:
            xs = np.linspace(0, neutral["sun"].max(), 50)
            ax.plot(xs, m_sun.params["Intercept"] + m_sun.params["sun"] * xs, color="#ff9f1c", lw=2)
        # full-model sun slope at dT_env = 0, for comparison
        xs = np.linspace(0, max(neutral["sun"].max(), 0.1), 50)
        ax.plot(xs, m.params["Intercept"] + m.params["sun"] * xs, color="#1f3b73", lw=1.5, ls="--",
                label="full model at outdoor = room")
        ax.legend(loc="upper left", fontsize=8)
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_xlabel("sun, 2-h mean (100 W/m2)")
    ax.set_title(f"|outdoor - room| < {MAX_GAP:g} K", fontsize=10)
    fig.suptitle("Summer, valves and damper closed, room empty", x=0.02, ha="left")
    return fig


if __name__ == "__main__":
    df = load()
    frame = passive_frame(df)
    m = fit(frame)
    ci = m.conf_int()
    print(f"Data {df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}; summer months {SUMMER}")
    print(f"Passive half-hours: {len(frame)} on {frame['date'].nunique()} days, "
          f"{(frame['sun'] > 0.5).sum()} with sun above 50 W/m2\n")
    for name, label in [("Intercept", "heat with outdoor = room and no sun (K/h)"),
                        ("dT_env", "per K of outdoor - room (K/h)"),
                        ("sun", "per 100 W/m2 of sun (K/h)")]:
        print(f"{label:42s} {m.params[name]:+.4f}  (95% CI {ci.loc[name, 0]:+.4f} to {ci.loc[name, 1]:+.4f})")
    print(f"R2 {m.rsquared:.2f}")

    # Outdoor ~ room: envelope term vanishes, so the room change is sun (+ internal/stored heat) alone
    neutral = neutral_frame(frame)
    m_sun = fit_sun_only(neutral)
    print(f"\nOutdoor within {MAX_GAP:g} K of room: {len(neutral)} half-hours on {neutral['date'].nunique()} days, "
          f"{(neutral['sun'] > 0.5).sum()} with sun above 50 W/m2")
    if len(neutral):
        print("Hours of day:", neutral["hour"].value_counts().sort_index().to_dict())
    if m_sun is not None:
        ci_s = m_sun.conf_int()
        for name, label in [("Intercept", "heat with no sun (K/h)"), ("sun", "per 100 W/m2 of sun (K/h)")]:
            print(f"{label:42s} {m_sun.params[name]:+.4f}  (95% CI {ci_s.loc[name, 0]:+.4f} to {ci_s.loc[name, 1]:+.4f})")
    else:
        print("Too few days for a clustered fit.")
    print("\nSensitivity to the gap threshold (sun-only fit):")
    print(sweep(frame).round(3).to_string())

    plot(frame, m, neutral, m_sun).savefig("passive_effects.png", dpi=200, bbox_inches="tight")
    plt.show()
