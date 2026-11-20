from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent

SIGNALS = {
    "indoor_temperature": "T",  # GT11
    "setpoint": "sp",    # logged real time setpoint
    "heating_valve": "hv",  # SV2x
    "VAV_damper": "dmp",   # ST4x
    "cooling_valve": "cv",   # SV4x
}


# Configuration
@dataclass
class ControlCard:
    deadzone_economy: float = 6.0
    deadzone_comfort: float = 2.0
    deviation_limit: float = 2.0


@dataclass
class CheckConfig:
    mode_source: str = "setpoint"
    economy_setpoint: float | None = None
    comfort_days: tuple = (0, 1, 2, 3, 4)
    comfort_start: str = "06:00"
    comfort_end: str = "19:00"

    open_thr: float = 5.0   # % : valve counted as open above this
    damper_min: float = 0.0    # % : ST4x position equal to min flow
    damper_min_tol: float = 5.0  # % : tolerance above min flow
    damper_max_for_cooling: float = 80.0    # % : What ST4x should be before SV4x opens
    saturation: float = 95.0    # % : actuator considered fully open
    margin: float = 0.5     # °C beyond a dead-zone edge before a response is required
    min_duration: str = "1h"        # flag must persist at least this long
    grace: str = "0.5h"        # ignore this long after a mode switch / setpoint step
    sp_step: float = 0.5      # °C : setpoint change treated as a step


CHECKS = {
    # name: (category, description)
    "temp_low": ("comfort", "GT11 below SP - deviation limit"),
    "temp_high": ("comfort", "GT11 above SP + deviation limit"),
    "simultaneous_heat_cool": ("sequence", "SV2x and SV4x open at the same time"),
    "heat_before_damper_min": ("sequence", "SV2x open while ST4x not at min flow"),
    "cool_before_damper_max": ("sequence", "SV4x open while ST4x not near max flow"),
    "heating_on_cool_demand": ("response", "SV2x open while GT11 above cooling edge"),
    "cooling_on_heat_demand": ("response", "SV4x/ST4x open while GT11 below heating edge"),
    "no_heat_response": ("response", "GT11 below heating edge, SV2x closed"),
    "no_cool_response": ("response", "GT11 above cooling edge, ST4x at min and SV4x closed"),
    "heating_in_deadzone": ("efficiency", "SV2x open while GT11 well inside dead zone"),
    "cooling_in_deadzone": ("efficiency", "SV4x/ST4x open while GT11 well inside dead zone"),
    "cooling_saturated": ("capacity", "ST4x and SV4x fully open, GT11 still above cooling edge"),
    "heating_saturated": ("capacity", "SV2x fully open, GT11 still below heating edge"),
}


# Loading
def load_room(room: str, base: Path = HERE):
    """Load <room>_data/dfs.pkl into one DataFrame with short column names."""
    path = base / f"{room}_data" / "dfs.pkl"
    dfs = pd.read_pickle(path)
    missing = [k for k in SIGNALS if k not in dfs]
    if missing:
        raise KeyError(f"{path}: missing signals {missing}")
    df = pd.concat({SIGNALS[k]: dfs[k]["value"] for k in SIGNALS}, axis=1).sort_index()
    return df


def list_rooms(base: Path = HERE):
    return sorted(p.parent.name[: -len("_data")] for p in base.glob("*_data/dfs.pkl"))


# Helpers
def _n_samples(index: pd.DatetimeIndex, duration: str) -> int:
    dt = pd.Series(index).diff().median()
    return max(1, int(np.ceil(pd.Timedelta(duration) / dt)))


def persist(mask: pd.Series, n: int):
    """Keep only True runs that last at least n consecutive samples."""
    m = mask.fillna(False).astype(bool)
    if n <= 1:
        return m
    run_id = (m != m.shift()).cumsum()
    run_len = m.groupby(run_id).transform("size")
    return m & (run_len >= n)


def detect_economy_setpoint(df: pd.DataFrame):
    """Most common logged setpoint at night (00:00-03:00), assumed economy mode."""
    night = df["sp"].between_time("00:00", "03:00").dropna()
    if night.empty:
        raise ValueError("no night-time setpoint data to detect the economy setpoint")
    return float(night.round(1).mode().iloc[0])


def comfort_mode(df: pd.DataFrame, cfg: CheckConfig):
    if cfg.mode_source == "setpoint":
        eco = cfg.economy_setpoint if cfg.economy_setpoint is not None \
            else detect_economy_setpoint(df)
        return (df["sp"] - eco).abs() > 0.05
    if cfg.mode_source == "schedule":
        t = df.index.strftime("%H:%M")
        in_hours = (t >= cfg.comfort_start) & (t < cfg.comfort_end)
        return pd.Series(in_hours & df.index.dayofweek.isin(cfg.comfort_days), index=df.index)
    raise ValueError(f"unknown mode_source {cfg.mode_source!r}")


# Core evaluation
def evaluate(df: pd.DataFrame, card: ControlCard | None = None,
             cfg: CheckConfig | None = None):
    """Return df extended with mode, bands, demand and one boolean column per check."""
    card = card or ControlCard()
    cfg = cfg or CheckConfig()
    out = df.copy()

    comfort = comfort_mode(out, cfg)
    out["mode"] = np.where(comfort, "comfort", "economy")

    sp = out["sp"]
    dz = pd.Series(np.where(comfort, card.deadzone_comfort, card.deadzone_economy), index=out.index)
    half = dz / 2
    lim = np.maximum(card.deviation_limit, half)

    out["heat_edge"] = sp - half
    out["cool_edge"] = sp + half
    out["temp_lo_limit"] = sp - lim
    out["temp_hi_limit"] = sp + lim

    T, hv, cv, dmp = out["T"], out["hv"], out["cv"], out["dmp"]
    out["demand"] = np.select([T < out["heat_edge"], T > out["cool_edge"]],
                              ["heat", "cool"], default="neutral")
    out.loc[T.isna() | sp.isna(), "demand"] = "unknown"

    valid = out[["T", "sp", "hv", "cv", "dmp"]].notna().all(axis=1)
    out["valid"] = valid

    # grace period after a mode switch or setpoint step
    change = (out["mode"] != out["mode"].shift()) | (sp.diff().abs() > cfg.sp_step)
    t = pd.Series(out.index, index=out.index)
    since_change = t - t.where(change).ffill()
    settled = since_change >= pd.Timedelta(cfg.grace)

    hv_open = hv > cfg.open_thr
    cv_open = cv > cfg.open_thr
    dmp_above_min = dmp > cfg.damper_min + cfg.damper_min_tol
    m = cfg.margin

    raw = {
        "temp_low": (T < out["temp_lo_limit"]) & settled,
        "temp_high": (T > out["temp_hi_limit"]) & settled,
        "simultaneous_heat_cool": hv_open & cv_open,
        "heat_before_damper_min": hv_open & dmp_above_min,
        "cool_before_damper_max": cv_open & (dmp < cfg.damper_max_for_cooling),
        "heating_on_cool_demand": (T > out["cool_edge"]) & hv_open & settled,
        "cooling_on_heat_demand": (T < out["heat_edge"]) & (cv_open | dmp_above_min) & settled,
        "no_heat_response": (T < out["heat_edge"] - m) & ~hv_open & settled,
        "no_cool_response": (T > out["cool_edge"] + m) & ~cv_open & ~dmp_above_min & settled,
        "heating_in_deadzone": (T > out["heat_edge"] + m) & (T < out["cool_edge"] - m) & hv_open & settled,
        "cooling_in_deadzone": (T > out["heat_edge"] + m) & (T < out["cool_edge"] - m)
                                & (cv_open | dmp_above_min) & settled,
        "cooling_saturated": (T > out["cool_edge"] + m) & (cv >= cfg.saturation)
                            & (dmp >= cfg.saturation),
        "heating_saturated": (T < out["heat_edge"] - m) & (hv >= cfg.saturation),
    }
    n = _n_samples(out.index, cfg.min_duration)
    for name, mask in raw.items():
        out[name] = persist(mask & valid, n)
    return out


# Reporting
def events(res: pd.DataFrame, check: str):
    """Contiguous runs of a flag, with context statistics."""
    f = res[check]
    if not f.any():
        return pd.DataFrame(columns=["start", "end", "duration_h", "mode", "T_mean", "T_min",
                                     "T_max", "sp_mean", "hv_mean", "dmp_mean", "cv_mean"])
    dt = pd.Series(res.index).diff().median()
    run_id = (f != f.shift()).cumsum()[f]
    g = res[f].groupby(run_id)
    ev = pd.DataFrame({
        "start": g.apply(lambda x: x.index[0]),
        "end": g.apply(lambda x: x.index[-1] + dt),
        "mode": g["mode"].agg(lambda s: s.mode().iloc[0]),
        "T_mean": g["T"].mean(), "T_min": g["T"].min(), "T_max": g["T"].max(),
        "sp_mean": g["sp"].mean(),
        "hv_mean": g["hv"].mean(), "dmp_mean": g["dmp"].mean(), "cv_mean": g["cv"].mean(),
    })
    ev.insert(2, "duration_h", (ev["end"] - ev["start"]) / pd.Timedelta("1h"))
    ev = ev.reset_index(drop=True)
    num = ev.select_dtypes("number").columns
    ev[num] = ev[num].round(2)
    return ev


def summarize(res: pd.DataFrame):
    dt_h = pd.Series(res.index).diff().median() / pd.Timedelta("1h")
    valid = res["valid"]
    comfort = valid & (res["mode"] == "comfort")
    economy = valid & (res["mode"] == "economy")
    rows = []
    for name, (cat, desc) in CHECKS.items():
        f = res[name]
        ev = events(res, name)
        rows.append({
            "check": name, # "category": cat, "description": desc,
            "description": desc,
            "hours": f.sum() * dt_h,
            "pct_time": 100 * f.sum() / max(valid.sum(), 1),
            # "pct_comfort": 100 * (f & comfort).sum() / max(comfort.sum(), 1),
            # "pct_economy": 100 * (f & economy).sum() / max(economy.sum(), 1),
            "n_events": len(ev),
            "longest_h": ev["duration_h"].max() if len(ev) else 0.0,
        })
    return pd.DataFrame(rows).set_index("check").round(2)


def plot_period(res: pd.DataFrame, start: str, end: str, path: Path | None = None,
                title: str = ""):
    import matplotlib
    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = res.loc[start:end]
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(13, 8), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 2, 1.4]})
    a1.fill_between(r.index, r["heat_edge"], r["cool_edge"], step="post",
                    color="#9a9890", alpha=0.2, label="dead zone")
    a1.step(r.index, r["sp"], where="post", color="k", lw=1, label="setpoint (logged)")
    a1.step(r.index, r["temp_lo_limit"], where="post", color="#2a78d6", ls="--", lw=0.8,
            label="deviation limit")
    a1.step(r.index, r["temp_hi_limit"], where="post", color="#eb6834", ls="--", lw=0.8)
    a1.plot(r.index, r["T"], color="#0b0b0b", lw=1.4, label="GT11")
    a1.set_ylabel("°C"); a1.legend(loc="upper left", fontsize=8, ncol=4)
    a1.set_title(title, loc="left")

    a2.plot(r.index, r["hv"], color="#eb6834", label="SV2x heating valve")
    a2.plot(r.index, r["dmp"], color="#1baf7a", label="ST4x VAV damper")
    a2.plot(r.index, r["cv"], color="#2a78d6", label="SV4x cooling valve")
    a2.set_ylabel("%"); a2.set_ylim(-3, 103); a2.legend(loc="upper left", fontsize=8, ncol=3)

    flagged = [c for c in CHECKS if r[c].any()]
    for i, c in enumerate(flagged):
        a3.plot(r.index, np.where(r[c], i, np.nan), lw=5, solid_capstyle="butt")
    a3.set_yticks(range(len(flagged))); a3.set_yticklabels(flagged, fontsize=7)
    a3.set_ylim(-0.7, max(len(flagged), 1) - 0.3)
    for a in (a1, a2, a3):
        a.grid(alpha=0.3)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=130); plt.close(fig)
    return fig


# CLI
def run_room(room: str, card: ControlCard, cfg: CheckConfig, out_dir: Path,
             plot: tuple[str, str] | None = None) -> pd.DataFrame:
    df = load_room(room)
    res = evaluate(df, card, cfg)
    summ = summarize(res)

    rdir = out_dir / room
    rdir.mkdir(parents=True, exist_ok=True)
    summ.to_csv(rdir / "summary.csv")
    res.to_csv(rdir / "flags_timeseries.csv")
    for c in CHECKS:
        ev = events(res, c)
        if len(ev):
            ev.to_csv(rdir / f"events_{c}.csv", index=False)
    if plot:
        plot_period(res, *plot, path=rdir / f"period_{plot[0]}_{plot[1]}.png", title=room)

    print(f"\n{'=' * 90}\n{room}   {res.index[0]:%Y-%m-%d} -> {res.index[-1]:%Y-%m-%d}   "
          f"valid {res['valid'].mean():.1%}   comfort {(res['mode'] == 'comfort').mean():.1%}")
    if cfg.mode_source == "setpoint" and cfg.economy_setpoint is None:
        print(f"Detected economy setpoint: {detect_economy_setpoint(df):.1f} °C")
    for mode in ("comfort", "economy"):
        s = res.loc[res["mode"] == mode, "sp"].dropna()
        if len(s):
            print(f"  {mode:8s} logged setpoint: median {s.median():.1f} °C, "
                  f"most common {s.round(1).value_counts().head(3).to_dict()}")
    print("\nChecks:")
    print(summ.drop(columns="description").to_string())
    return summ


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rooms", nargs="*", default=None, help="e.g. 05-A1-01 07-D4-01 (default: all)")
    p.add_argument("--mode-source", choices=["setpoint", "schedule"], default="setpoint")
    p.add_argument("--economy-setpoint", type=float, default=None,
                   help="logged setpoint value meaning economy mode (default: detect from data)")
    p.add_argument("--min-duration", default="2h")
    p.add_argument("--grace", default="1h")
    p.add_argument("--out", default=str(HERE / "control_check_results"))
    p.add_argument("--plot", nargs=2, metavar=("START", "END"), default=None)
    a = p.parse_args(argv)

    card = ControlCard()
    cfg = CheckConfig(mode_source=a.mode_source, economy_setpoint=a.economy_setpoint,
                      min_duration=a.min_duration, grace=a.grace)
    out_dir = Path(a.out)
    # print("Control card:", asdict(card))
    # print("Config:      ", asdict(cfg))

    rooms = a.rooms or list_rooms()
    summaries = {r: run_room(r, card, cfg, out_dir, tuple(a.plot) if a.plot else None)
                 for r in rooms}
    # overview = pd.concat({r: s["pct_time"] for r, s in summaries.items()}, axis=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    # overview.to_csv(out_dir / "overview_pct_time.csv")
    # print(f"\n{'=' * 90}\n% of valid time flagged, per room:\n{overview.to_string()}")
    print(f"\nResults written to {out_dir}")


if __name__ == "__main__":
    main()
