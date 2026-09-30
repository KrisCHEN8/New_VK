import copy
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import holidays


# tunable parameters
CONFIG = {
    "key_map": {
        "indoor_temperature": "T_in",
        "setpoint": "setpoint",
        "outdoor_temperature": "T_out",
        "solar_radiation": "solar",
        "heating": "heating",
        "cooling": "cooling",
        "VAV": "vav",
        "co2": "co2",
        "humidity": "RH_sup",
        "KB02_supply": "kb_sup",
    },
    "core_cols": ["T_in", "setpoint", "T_out", "solar"],
    "context_cols": ["heating", "cooling", "vav", "co2", "RH_sup", "kb_sup"],
    "value_col": ['value'],
    "time_col": "observationTime",
    "naive_tz": "UTC",
    "local_tz": "Europe/Stockholm",
    "resample_freq": "30min",

    # preprocessing
    "max_interp_gap": "1h",
    "step_cols": ["setpoint"],
    "T_out_window": "2h",
    "solar_window": "2h",
    "solar_halflife": "3h",    # accumulated solar, time-of-day proxy
    "rolling_min_frac": 0.5,   # min share of valid samples in a rolling window

    # conditional distribution: target variable and solar conditioning column
    "target_col": "T_in",      # "T_in" or "dev"
    "solar_col": "solar_acc",  # "solar_acc" or "solar_2h"

    # Occupancy schedule (weekday 0 = Monday)
    "occupied_weekdays": [0, 1, 2, 3, 4],
    "occupied_start": "08:00",
    "occupied_end": "18:00",
    "public_holidays": True,         # Swedish public holidays (holidays.Sweden) are unoccupied
    "extra_days_off": [],          # "YYYY-MM-DD"
    "co2_occupied_threshold": 400.0,   # ppm

    # Setpoint-change recovery
    "setpoint_change_min": 0.2,  # °C; smaller steps are ignored
    "recovery_window": "30min",

    # binning (per regime): bin_col x solar_col
    "bin_col": "kb_sup",      # KB02 supply temperature
    "bin_width": 0.5,         # °C
    "bin_edges": None,        # explicit edges override bin_width
    "solar_bin_edges": [-np.inf, 50.0, 200.0, np.inf],   # applied to solar_col
    "solar_bin_labels": ["low", "mid", "high"],

    # robust statistics
    "mad_scale": 1.4826,
    "mad_floor": 0.1,         # °C, applied to the raw MAD
    "min_bin_count": 50,

    # flagging
    "z_threshold": 3.5,
    "min_abs_residual": 0.5,         # °C, abs(dev - bin_median)

    # events
    "event_max_gap_samples": 1,
    "event_min_duration": "30min",

    # hints
    "near_max": {"heating": 90.0, "cooling": 90.0, "vav": 90.0},
    "near_zero": {"heating": 5.0, "cooling": 5.0, "vav": 5.0},
    "active": {"heating": 10.0, "cooling": 10.0},
    "simultaneous_min_frac": 0.25,   # share of event samples with both active
    "high_solar": 400.0,           # W/m²
    "high_rh": 55.0,               # %RH, supply air

    # sanity checks in inspect_inputs
    "plausible_ranges": {
        "T_in": (5.0, 40.0),
        "setpoint": (5.0, 35.0),
        "T_out": (-40.0, 45.0),
        "solar": (-10.0, 1500.0),
        "heating": (-1.0, 101.0),
        "cooling": (-1.0, 101.0),
        "vav": (-1.0, 101.0),
        "co2": (250.0, 5000.0),
        "RH_sup": (0.0, 100.0),
        "kb_sup": (5.0, 35.0),
    },

    # output
    "top_n_events": 10,
    "make_plots": True,
}

REGIMES = ["occupied", "unoccupied"]
DIRECTION = {1: "too warm", -1: "too cold"}


# Helpers
def _merge_config(config, base=None):
    """Return a copy of 'base' (default CONFIG) updated with the user overrides in 'config'."""
    cfg = copy.deepcopy(CONFIG if base is None else base)
    if config:
        cfg.update(copy.deepcopy(config))
    return cfg


def _localize(ts, tz):
    """Timestamp in time zone 'tz'"""
    ts = pd.Timestamp(ts)
    if tz is not None:
        ts = ts.tz_localize(tz) if ts.tz is None else ts.tz_convert(tz)
    return ts


def _extract_series(obj, cfg, name):
    notes = []
    obj = obj.copy()

    if isinstance(obj, pd.DataFrame):
        cols = list(obj.columns)
        if cfg["value_col"] is not None and cfg["value_col"] in cols:
            vcol = cfg["value_col"]
        elif "value" in cols:
            vcol = "value"
        else:
            numeric = [c for c in obj.select_dtypes(include=[np.number]).columns]
            if len(numeric) != 1:
                raise ValueError(
                    f"{name}: cannot pick a value column from {cols}; "
                    f"set CONFIG['value_col']."
                )
            vcol = numeric[0]
        s = obj[vcol]
        time_source = obj[cfg["time_col"]] if cfg["time_col"] in cols else None
    elif isinstance(obj, pd.Series):
        s = obj
        time_source = None
    else:
        raise ValueError(f"{name}: unsupported type {type(obj).__name__}")

    idx = s.index
    if not isinstance(idx, pd.DatetimeIndex):
        source = time_source if time_source is not None else idx
        notes.append("index is not a DatetimeIndex; parsed timestamps with utc=True")
        idx = pd.DatetimeIndex(pd.to_datetime(source, utc=True, format="mixed", errors="coerce"))
        if idx.isna().all():
            raise ValueError(f"{name}: could not parse any timestamps")
    if idx.tz is None:
        idx = idx.tz_localize(cfg["naive_tz"], ambiguous="NaT", nonexistent="NaT")
    idx = idx.tz_convert("UTC")

    out = pd.Series(pd.to_numeric(s.to_numpy(), errors="coerce"), index=idx, name=name)
    out = out[out.index.notna()].sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out, notes


def _run_lengths(mask):
    """Return (run_id, run_length) Series for True-runs in a boolean Series."""
    mask = mask.astype(bool)
    run_id = (mask != mask.shift(fill_value=False)).cumsum()
    lengths = mask.groupby(run_id).transform("sum")
    return run_id.where(mask), lengths.where(mask)


def _fill_short_gaps(s, max_samples, step=False):
    isna = s.isna()
    _, run_len = _run_lengths(isna)
    if step:
        filled = s.ffill()
        # do not fill trailing NaNs (no later value -> not "inside")
        filled[s.bfill().isna()] = np.nan
    else:
        filled = s.interpolate(method="time", limit_area="inside")
    too_long = isna & (run_len > max_samples)
    filled[too_long] = np.nan
    return filled


def _freq_seconds(freq):
    return pd.Timedelta(pd.tseries.frequencies.to_offset(freq)).total_seconds()


# Step 0: inspection and combination
def inspect_inputs(dfs, config=None):
    """
    Print type, columns, index type/tz, time range, sampling interval, NaN
    count and min/median/max for each object in dfs.
    """
    cfg = _merge_config(config)
    issues = []
    target_s = _freq_seconds(cfg["resample_freq"])
    tzs = {}

    for key, short in cfg["key_map"].items():
        print("=" * 78)
        if key not in dfs:
            print(f"{key}: MISSING")
            level = "ERROR" if short in cfg["core_cols"] else "WARN"
            issues.append((level, f"{key}: missing from dfs"))
            continue
        obj = dfs[key]
        print(f"{key} -> {short}")
        print(f"  type        : {type(obj).__name__}")
        if isinstance(obj, pd.DataFrame):
            print(f"  columns     : {list(obj.columns)}")
        print(f"  index type  : {type(obj.index).__name__}, dtype={obj.index.dtype}")
        if isinstance(obj.index, pd.DatetimeIndex):
            tzs[key] = str(obj.index.tz)
            print(f"  index tz    : {obj.index.tz}")
            if not obj.index.is_monotonic_increasing:
                issues.append(("WARN", f"{key}: index not sorted (will be sorted)"))

        try:
            s, notes = _extract_series(obj, cfg, short)
        except ValueError as e:
            print(f"  !! {e}")
            issues.append(("ERROR", str(e)))
            continue
        for n in notes:
            print(f"  note        : {n}")
            issues.append(("WARN", f"{key}: {n}"))

        n_dup = obj.index.duplicated().sum() if isinstance(obj.index, pd.DatetimeIndex) else 0
        dt = s.index.to_series().diff().dropna()
        med_dt = dt.median() if len(dt) else pd.NaT
        valid = s.dropna()
        print(f"  time range  : {s.index.min()} -> {s.index.max()} (UTC)")
        print(f"  n rows      : {len(s)}  (duplicate timestamps: {n_dup})")
        print(f"  interval    : median {med_dt}, min {dt.min() if len(dt) else 'n/a'}, "
              f"max {dt.max() if len(dt) else 'n/a'}")
        print(f"  NaNs        : {s.isna().sum()} ({s.isna().mean():.1%})")
        if len(valid):
            print(f"  min/med/max : {valid.min():.3g} / {valid.median():.3g} / {valid.max():.3g}")
        else:
            issues.append(("ERROR" if short in cfg["core_cols"] else "WARN", f"{key}: no valid values"))
            continue

        # sampling interval vs. target resolution
        if pd.notna(med_dt) and med_dt.total_seconds() > target_s * 1.01:
            issues.append((
                "WARN",
                f"{key}: native interval {med_dt} is coarser than resample_freq "
                f"{cfg['resample_freq']}; resampling by mean leaves empty slots that "
                f"are then filled by short-gap interpolation. Consider "
                f"CONFIG['resample_freq'] = '{med_dt}'."
            ))

        # sanity checks
        if short in ("T_in", "setpoint", "T_out"):
            if med > 200:
                issues.append(("ERROR", f"{key}: median {med:.1f} looks like Kelvin"))
            elif short != "T_out" and med > 45:
                issues.append(("ERROR", f"{key}: median {med:.1f} looks like °F"))
        if short == "solar":
            zero_share = (valid.abs() < 5).mean()
            if zero_share < 0.2:
                issues.append(("WARN", f"{key}: only {zero_share:.0%} near-zero values; "
                                       f"is this really solar radiation (no nights)?"))
        if short == "T_out":
            if (valid < -5).mean() == 0 and valid.max() > 200:
                issues.append(("ERROR", f"{key}: values look like solar radiation, not temperature"))
        lo, hi = cfg["plausible_ranges"].get(short, (-np.inf, np.inf))
        n_out = ((valid < lo) | (valid > hi)).sum()
        if n_out:
            issues.append(("WARN", f"{key}: {n_out} values outside plausible range [{lo}, {hi}]"))

    if len(set(tzs.values())) > 1:
        issues.append(("WARN", f"mixed index time zones {tzs}; everything is converted to UTC"))

    print("=" * 78)
    if issues:
        print("ISSUES:")
        for level, msg in issues:
            print(f"  [{level}] {msg}")
    else:
        print("No issues found.")
    return issues


def combine_data(dfs, config=None):
    cfg = _merge_config(config)
    freq = cfg["resample_freq"]
    series = {}
    for key, short in cfg["key_map"].items():
        if key not in dfs:
            continue
        s, _ = _extract_series(dfs[key], cfg, short)
        series[short] = s.resample(freq).mean()

    data = pd.concat(series, axis=1)
    data = data.reindex(pd.date_range(data.index.min(), data.index.max(), freq=freq))
    for c in cfg["core_cols"] + cfg["context_cols"]:
        if c not in data:
            data[c] = np.nan

    firsts = {c: data[c].first_valid_index() for c in cfg["core_cols"]}
    lasts = {c: data[c].last_valid_index() for c in cfg["core_cols"]}
    if any(v is None for v in firsts.values()):
        raise ValueError(f"A core signal has no data: {firsts}")
    start, end = max(firsts.values()), min(lasts.values())
    if start >= end:
        raise ValueError(f"Core signals do not overlap: first={firsts}, last={lasts}")

    local = cfg["local_tz"]
    # print("\nCore-signal availability (local time):")
    # for c in cfg["core_cols"]:
    #     print(f"  {c:9s}: {firsts[c].tz_convert(local)} -> {lasts[c].tz_convert(local)}")
    data = data.loc[start:end].copy()
    data.index = data.index.tz_convert(local)
    data.index.name = "time"
    overlap = (data.index[0], data.index[-1])
    # print(f"Overlap used for analysis: {overlap[0]} -> {overlap[1]} "
    #       f"({len(data)} samples at {freq})")
    return data, overlap


# Step 1: preprocessing
def report_gaps(data, cols, freq):
    """Return a table with missing-value and gap statistics per column."""
    step = pd.Timedelta(freq)
    rows = []
    for c in cols:
        isna = data[c].isna()
        run_id, run_len = _run_lengths(isna)
        lens = run_len.groupby(run_id).first() if isna.any() else pd.Series(dtype=float)
        rows.append({
            "column": c,
            "n_missing": int(isna.sum()),
            "share_missing": isna.mean(),
            "n_gaps": len(lens),
            "longest_gap": lens.max() * step if len(lens) else pd.Timedelta(0),
        })
    return pd.DataFrame(rows).set_index("column")


def preprocess(data, config=None):
    cfg = _merge_config(config)
    data = data.copy()
    freq = cfg["resample_freq"]
    cols = cfg["core_cols"] + cfg["context_cols"]

    # print("\nMissing values / gaps BEFORE filling:")
    # print(report_gaps(data, cols, freq).to_string(float_format=lambda x: f"{x:.3f}"))

    max_samples = int(pd.Timedelta(cfg["max_interp_gap"]) / pd.Timedelta(freq))
    core_missing_before = data[cfg["core_cols"]].isna().any(axis=1)
    for c in cols:
        data[c] = _fill_short_gaps(data[c], max_samples, step=c in cfg["step_cols"])
    data["core_filled"] = core_missing_before & data[cfg["core_cols"]].notna().all(axis=1)

    # print(f"\nAfter filling gaps <= {cfg['max_interp_gap']}:")
    # print(report_gaps(data, cols, freq).to_string(float_format=lambda x: f"{x:.3f}"))
    # print(f"Core samples filled: {data['core_filled'].sum()} "
    #       f"({data['core_filled'].mean():.1%})")

    # deviation and lagged drivers
    data["dev"] = data["T_in"] - data["setpoint"]
    for src, win, dst in [("T_out", cfg["T_out_window"], "T_out_3h"),
                          ("solar", cfg["solar_window"], "solar_2h")]:
        n_win = pd.Timedelta(win) / pd.Timedelta(freq)
        min_p = max(1, int(np.ceil(cfg["rolling_min_frac"] * n_win)))
        data[dst] = data[src].rolling(win, min_periods=min_p).mean()
    # exponentially weighted solar history: recent sun counts most, so it also
    # separates morning (little accumulated) from afternoon (much accumulated)
    data["solar_acc"] = data["solar"].ewm(halflife=cfg["solar_halflife"], times=data.index).mean()

    data = add_schedule(data, cfg)

    # setpoint changes and recovery periods
    sp = data["setpoint"].dropna()
    change = sp.diff().abs() >= cfg["setpoint_change_min"]
    data["setpoint_change"] = change.reindex(data.index, fill_value=False).astype(bool)
    data["recovery"] = (data["setpoint_change"].astype(float)
                        .rolling(cfg["recovery_window"], min_periods=1).max()
                        .fillna(0).astype(bool))
    print("\nSetpoint changes: "
          f"{int(data['setpoint_change'].sum())}; samples marked 'recovery': "
          f"{int(data['recovery'].sum())} ({data['recovery'].mean():.1%})")
    return data


def add_schedule(data, cfg):
    """Add day_off (Swedish public holidays + extra_days_off), occupied and regime columns."""
    # regimes from the schedule (local time)
    idx = data.index
    to_min = lambda hhmm: int(hhmm.split(":")[0]) * 60 + int(hhmm.split(":")[1])
    minute_of_day = idx.hour * 60 + idx.minute
    in_hours = (minute_of_day >= to_min(cfg["occupied_start"])) & (minute_of_day < to_min(cfg["occupied_end"]))
    local_dates = idx.tz_convert(cfg["local_tz"]).normalize().tz_localize(None)
    days_off = pd.DatetimeIndex(pd.to_datetime(cfg["extra_days_off"]))
    # filter out Swedish red days
    if cfg["public_holidays"]:
        years = range(local_dates.min().year, local_dates.max().year + 1)
        se = holidays.Sweden(years=years, include_sundays=False)
        days_off = days_off.union(pd.DatetimeIndex(pd.to_datetime(list(se.keys()))))
    data["day_off"] = local_dates.isin(days_off)
    data["occupied"] = idx.weekday.isin(cfg["occupied_weekdays"]) & in_hours & ~data["day_off"]
    n_off = pd.Index(local_dates[data["day_off"] & idx.weekday.isin(cfg["occupied_weekdays"])]).nunique()
    # print(f"\nDays off on scheduled weekdays (holidays + extra_days_off): {n_off}")
    data["regime"] = np.where(data["occupied"], "occupied", "unoccupied")

    # # CO2-based occupancy (report only)
    # co2_valid = data["co2"].notna()
    # data["co2_occupied"] = (data["co2"] > cfg["co2_occupied_threshold"]).where(co2_valid)
    # if co2_valid.any():
    #     sub = data.loc[co2_valid]
    #     agree = (sub["co2_occupied"].astype(bool) == sub["occupied"]).mean()
    #     print(f"\nCO2 occupancy (co2 > {cfg['co2_occupied_threshold']:.0f} ppm) vs. schedule "
    #           f"({co2_valid.sum()} samples with CO2):")
    #     print(f"  overall agreement: {agree:.1%}")
    #     ct = pd.crosstab(sub["regime"], sub["co2_occupied"].astype(bool)
    #                      .map({True: "co2 high", False: "co2 low"}), normalize="index")
    #     print((ct * 100).round(1).to_string())
    # else:
    #     print("\nNo CO2 data; occupancy agreement not computed.")
    return data


# Step 2: binning
def make_bin_edges(values, config=None):
    """Return bin edges: CONFIG['bin_edges'] or width-aligned edges covering `values`."""
    cfg = _merge_config(config)
    if cfg["bin_edges"] is not None:
        return np.asarray(cfg["bin_edges"], dtype=float)
    w = cfg["bin_width"]
    v = pd.Series(values).dropna()
    lo = np.floor(v.min() / w) * w
    hi = np.floor(v.max() / w) * w + w
    return np.arange(lo, hi + w / 2, w)


def assign_bins(data, config=None):
    """
    Add bin (label) and bin_left (float) of CONFIG['bin_col'] and solar_bin (label)
    to copy of 'data'
    """
    cfg = _merge_config(config)
    data = data.copy()
    x = data[cfg["bin_col"]]
    edges = make_bin_edges(x, cfg)
    labels = [f"[{a:g}, {b:g})" for a, b in zip(edges[:-1], edges[1:])]
    data["bin"] = pd.cut(x, bins=edges, right=False, labels=labels).astype(object)
    data["bin_left"] = pd.cut(x, bins=edges, right=False, labels=edges[:-1]).astype(float)
    data["solar_bin"] = pd.cut(data[cfg["solar_col"]], bins=cfg["solar_bin_edges"], right=False,
                               labels=cfg["solar_bin_labels"]).astype(object)
    return data


# Step 3: per-bin robust statistics
BIN_KEYS = ["regime", "bin_left", "solar_bin"]


def _eligible(data, target="dev"):
    """Points usable for bin statistics: valid target and bins, not in recovery."""
    return (data[target].notna() & data["bin_left"].notna()
            & data["solar_bin"].notna() & ~data["recovery"])


def compute_bin_stats(data, config=None):
    """
    Per (regime, bin_col bin, solar bin): count, median and MAD of target_col,
    the floored MAD used for z-scores, and whether the bin has enough data.
    Recovery points are excluded.
    """
    cfg = _merge_config(config)
    tgt = cfg["target_col"]
    d = data.loc[_eligible(data, tgt)]
    g = d.groupby(BIN_KEYS)[tgt]
    med = g.median().rename("median")
    stats = pd.concat([g.size().rename("count"), med], axis=1)
    abs_res = (d[tgt] - d.groupby(BIN_KEYS)[tgt].transform("median")).abs()
    stats["mad"] = abs_res.groupby([d[k] for k in BIN_KEYS]).median()
    stats["mad_used"] = stats["mad"].clip(lower=cfg["mad_floor"])
    stats["sufficient"] = stats["count"] >= cfg["min_bin_count"]
    stats = stats.reset_index()

    labels = (data.dropna(subset=["bin_left"])
              .drop_duplicates("bin_left")
              .set_index("bin_left")["bin"])
    stats.insert(1, "bin", stats["bin_left"].map(labels))
    stats["solar_bin"] = pd.Categorical(stats["solar_bin"], categories=cfg["solar_bin_labels"],
                                        ordered=True)
    stats = stats.sort_values(BIN_KEYS).reset_index(drop=True)
    stats["solar_bin"] = stats["solar_bin"].astype(object)
    return stats


# Step 4: flagging
def flag_anomalies(data, bin_stats, config=None):
    """
    Add bin_median, bin_mad, bin_count, bin_sufficient, residual, z,
    anomaly, direction and status columns to a copy of `data`.

    status: 'missing' | 'recovery' | 'insufficient_data' | 'normal' | 'anomaly'
    """
    cfg = _merge_config(config)
    data = data.copy()
    tgt = cfg["target_col"]
    s = bin_stats.set_index(BIN_KEYS)[["median", "mad_used", "count", "sufficient"]]
    s.columns = ["bin_median", "bin_mad", "bin_count", "bin_sufficient"]
    for c in s.columns:
        if c in data:
            data = data.drop(columns=c)
    data = data.join(s, on=BIN_KEYS)
    data["bin_sufficient"] = data["bin_sufficient"].eq(True)

    elig = _eligible(data, tgt)
    scored = elig & data["bin_sufficient"]
    data["residual"] = (data[tgt] - data["bin_median"]).where(scored)
    data["z"] = data["residual"] / (cfg["mad_scale"] * data["bin_mad"])
    data["anomaly"] = (scored & (data["z"].abs() > cfg["z_threshold"])
                       & (data["residual"].abs() > cfg["min_abs_residual"]))
    data["direction"] = np.where(data["anomaly"], np.sign(data["residual"]), 0).astype(int)

    status = pd.Series("normal", index=data.index, dtype=object)
    missing = data[tgt].isna() | data["bin_left"].isna() | data["solar_bin"].isna()
    status[elig & ~data["bin_sufficient"]] = "insufficient_data"
    status[data["recovery"] & ~missing] = "recovery"
    status[missing] = "missing"
    status[data["anomaly"]] = "anomaly"
    data["status"] = status
    return data


# Step 5: events
def _hints(ev, cfg):
    """Return a list of hint codes for one event row."""
    nm, nz, act = cfg["near_max"], cfg["near_zero"], cfg["active"]
    cold = ev["direction"] == "too cold"
    h, c, v = ev["heating_mean"], ev["cooling_mean"], ev["vav_mean"]
    hints = []
    if all(pd.isna(x) for x in (h, c, v)):
        return ["no HVAC data"]
    if cold and h >= nm["heating"]:
        hints.append("heating saturated: capacity/supply issue")
    if not cold and c >= nm["cooling"]:
        hints.append("cooling saturated: capacity issue or high solar/internal gains")
    if cold and h <= nz["heating"]:
        hints.append("too cold, heating ~0: control issue or HVAC off")
    if not cold and c <= nz["cooling"] and (pd.isna(v) or v <= nz["vav"]):
        hints.append("too warm, cooling & VAV ~0: control issue or HVAC off")
    if cold and c >= act["cooling"]:
        hints.append("cooling active while too cold")
    if not cold and h >= act["heating"]:
        hints.append("heating active while too warm")
    if ev["simultaneous_frac"] >= cfg["simultaneous_min_frac"]:
        hints.append("simultaneous heating/cooling")
    if not hints:
        hints.append("unexplained (actuators partly active)")
    if not cold and ev["solar_2h_mean"] >= cfg["high_solar"]:
        hints.append("high solar")
    rh = ev.get("RH_sup_mean", np.nan)
    if not cold and pd.notna(rh) and rh >= cfg["high_rh"]:
        # latent load only limits sensible cooling while cooling is running
        if pd.notna(c) and c >= act["cooling"]:
            hints.append("humid supply air: latent load / chilled-beam dew-point limit")
        else:
            hints.append("humid supply air")
    return hints


def collect_events(data, cfg, extra_stats, score_col, score_of):
    """
    Group consecutive anomalous points (same regime and direction, gaps up to
    event_max_gap_samples) into events, keep those >= event_min_duration.

    extra_stats: {column: fn(segment) -> value}, inserted after max_abs_dev.
    score_of: fn(events) -> ranking Series stored in `score_col`.
    Returns (data, events): data gets an event_id column (-1 = none).
    """
    data = data.copy()
    freq = pd.Timedelta(cfg["resample_freq"])
    data["event_id"] = -1

    cols = ["start", "end", "duration", "regime", "direction", "n_anomalous", "max_abs_dev"]
    cols += list(extra_stats) + ["T_out_3h_mean", "solar_2h_mean"]
    cols += [f"{c}_{a}" for c in cfg["context_cols"] for a in ("mean", "max")]
    cols += ["simultaneous_frac", "hint", score_col]

    pos = pd.Series(np.arange(len(data)), index=data.index)
    an = data.loc[data["anomaly"], ["regime", "direction"]].copy()
    if an.empty:
        return data, pd.DataFrame(columns=cols)
    an["pos"] = pos[an.index]
    new = ((an["pos"].diff() > cfg["event_max_gap_samples"] + 1)
           | (an["direction"] != an["direction"].shift())
           | (an["regime"] != an["regime"].shift()))
    an["grp"] = new.cumsum()

    act = cfg["active"]
    rows = []
    for _, g in an.groupby("grp"):
        start, end = g.index[0], g.index[-1]
        duration = end - start + freq
        if duration < pd.Timedelta(cfg["event_min_duration"]):
            continue
        seg = data.loc[start:end]
        both = (seg["heating"] >= act["heating"]) & (seg["cooling"] >= act["cooling"])
        valid_both = seg["heating"].notna() & seg["cooling"].notna()
        row = {
            "start": start, "end": end, "duration": duration,
            "regime": g["regime"].iloc[0],
            "direction": DIRECTION[int(g["direction"].iloc[0])],
            "n_anomalous": len(g),
            "max_abs_dev": seg["dev"].abs().max(),
            **{k: fn(seg) for k, fn in extra_stats.items()},
            "T_out_3h_mean": seg["T_out_3h"].mean(),
            "solar_2h_mean": seg["solar_2h"].mean(),
            "simultaneous_frac": both[valid_both].mean() if valid_both.any() else 0.0,
        }
        for c in cfg["context_cols"]:
            row[f"{c}_mean"] = seg[c].mean()
            row[f"{c}_max"] = seg[c].max()
        rows.append(row)

    events = pd.DataFrame(rows, columns=[c for c in cols if c not in ("hint", score_col)])
    if events.empty:
        return data, pd.DataFrame(columns=cols)
    events["hint"] = ["; ".join(_hints(r, cfg)) for _, r in events.iterrows()]
    events[score_col] = score_of(events)
    events.index.name = "event_id"
    for eid, r in events.iterrows():
        data.loc[r["start"]:r["end"], "event_id"] = eid
    return data, events


def detect_events(data, config=None):
    """
    Events from the z-score flags (see collect_events).
    score = max|dev| [°C] × duration [h]. Returns (data, events).
    """
    cfg = _merge_config(config)
    extra = {"mean_dev": lambda s: s["dev"].mean(),
             "max_abs_z": lambda s: s["z"].abs().max()}
    return collect_events(data, cfg, extra, "score",
                          lambda ev: ev["max_abs_dev"] * ev["duration"].dt.total_seconds() / 3600)


# Step 6: plots
def plot_dev_heatmap(data, vmax=None):
    """Heatmap of dev: date on y, hour of day on x (hourly means), diverging around 0."""
    d = data["dev"]
    df = pd.DataFrame({"date": d.index.date, "hour": d.index.hour, "dev": d.values})
    grid = df.pivot_table(index="date", columns="hour", values="dev", aggfunc="mean")
    grid = grid.reindex(columns=range(24))
    if vmax is None:
        vmax = np.nanpercentile(np.abs(grid.values), 98) if np.isfinite(grid.values).any() else 1
    fig, ax = plt.subplots(figsize=(9, max(4, 0.12 * len(grid))))
    im = ax.imshow(grid.values, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest", extent=[-0.5, 23.5, len(grid) - 0.5, -0.5])
    step = max(1, len(grid) // 20)
    ax.set_yticks(range(0, len(grid), step))
    ax.set_yticklabels([str(x) for x in grid.index[::step]])
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day")
    ax.set_title("Deviation from setpoint, T_in − setpoint [°C] (hourly mean; white = 0, grey = no data)")
    ax.set_facecolor("#d9d9d9")
    fig.colorbar(im, ax=ax, label="dev [°C]")
    fig.tight_layout()
    return fig


def plot_dev_scatter(data, config=None):
    """Scatter of target_col vs bin_col coloured by solar_col, one panel per regime, anomalies ringed."""
    cfg = _merge_config(config)
    tgt, sol, x = cfg["target_col"], cfg["solar_col"], cfg["bin_col"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    vmax = np.nanpercentile(data[sol], 99) if data[sol].notna().any() else 1
    sc = None
    for ax, reg in zip(axes, REGIMES):
        d = data[data["regime"] == reg]
        ok = d[d["status"].isin(["normal", "insufficient_data"])]
        sc = ax.scatter(ok[x], ok[tgt], c=ok[sol], cmap="YlOrRd",
                        vmin=0, vmax=vmax, s=8, alpha=0.6, linewidths=0)
        rec = d[d["status"] == "recovery"]
        ax.scatter(rec[x], rec[tgt], s=8, c="#b0b0b0", alpha=0.5,
                   linewidths=0, label="recovery (excluded)")
        an = d[d["anomaly"]]
        ax.scatter(an[x], an[tgt], s=36, facecolors="none", edgecolors="#1f3b73",
                   linewidths=1.2, label=f"anomaly (n={len(an)})")
        # vertical gridlines at the bin edges
        for e in make_bin_edges(data[x], cfg):
            ax.axvline(e, color="#e5e5e5", lw=0.6, zorder=0)
        if tgt == "dev":
            ax.axhline(0, color="#808080", lw=0.8)
        ax.set_title(f"{reg} (n={d[tgt].notna().sum()})")
        ax.set_xlabel(f"{x} [°C]")
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    axes[0].set_ylabel("dev = T_in - setpoint [°C]" if tgt == "dev" else "T_in [°C]")
    sol_label = (f"solar weighted accumulation {cfg['solar_halflife']} [W/m²]" if sol == "solar_acc"
                 else "solar, 2 h moving average [W/m²]")
    fig.colorbar(sc, ax=axes, label=sol_label, shrink=0.9)

    return fig


def plot_bin_grids(bin_stats, config=None):
    """Heatmaps over the bin grid per regime: sample count and median target_col (insufficient bins shaded)."""
    cfg = _merge_config(config)
    solar_labels = cfg["solar_bin_labels"]
    lefts = np.sort(bin_stats["bin_left"].unique())
    w = np.diff(lefts).min() if len(lefts) > 1 else cfg["bin_width"]
    fig, axes = plt.subplots(2, 2, figsize=(max(10, 0.35 * len(lefts) * 2), 7), squeeze=False)
    if cfg["target_col"] == "dev":
        med_lim = max(0.5, np.nanpercentile(bin_stats["median"].abs(), 98)) if len(bin_stats) else 1
        med_cmap, med_range = "RdBu_r", (-med_lim, med_lim)
    else:
        med_cmap = "RdYlBu_r"
        med_range = tuple(np.nanpercentile(bin_stats["median"], [2, 98])) if len(bin_stats) else (0, 1)

    for r, reg in enumerate(REGIMES):
        s = bin_stats[bin_stats["regime"] == reg]
        cnt = s.pivot(index="solar_bin", columns="bin_left", values="count") \
               .reindex(index=solar_labels, columns=lefts)
        med = s.pivot(index="solar_bin", columns="bin_left", values="median") \
               .reindex(index=solar_labels, columns=lefts)
        suff = s.pivot(index="solar_bin", columns="bin_left", values="sufficient") \
                .reindex(index=solar_labels, columns=lefts)
        for c, (grid, cmap, vmin, vmax, title) in enumerate([
            (cnt, "Blues", 0, np.nanmax(cnt.values) if np.isfinite(cnt.values).any() else 1,
             "sample count"),
            (med, med_cmap, *med_range, f"median {cfg['target_col']} [°C]"),
        ]):
            ax = axes[r, c]
            ax.set_facecolor("#eeeeee")
            im = ax.imshow(grid.values.astype(float), aspect="auto", cmap=cmap, vmin=vmin,
                           vmax=vmax, origin="lower", interpolation="nearest")
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    val = grid.values[i, j]
                    if pd.isna(val):
                        continue
                    ok = bool(suff.values[i, j])
                    txt = f"{int(val)}" if c == 0 else f"{val:.1f}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=6, rotation=90,
                            color="#222222" if ok else "#888888")
                    if not ok:
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                                   hatch="///", edgecolor="#999999", lw=0))
            ax.set_yticks(range(len(solar_labels)))
            ax.set_yticklabels(solar_labels)
            ax.set_xticks(range(len(lefts)))
            ax.set_xticklabels([f"{a:g}" for a in lefts], fontsize=7, rotation=90)
            ax.set_xlabel(f"{cfg['bin_col']} bin, left edge [°C] (width {w:g})")
            ax.set_ylabel(cfg["solar_col"])
            ax.set_title(f"{reg}: {title}  (shaded = < {cfg['min_bin_count']} samples)")
            fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def _working_days(data, cfg):
    """Scheduled weekdays that are not public holidays / extra days off."""
    return data.index.weekday.isin(cfg["occupied_weekdays"]) & ~data["day_off"]


def plot_z_working_days(data, config=None, ylim=None):
    cfg = _merge_config(config)
    thr = cfg["z_threshold"]
    d = data.loc[_working_days(data, cfg) & data["z"].notna()]
    tod = pd.Series(d.index.hour + d.index.minute / 60, index=d.index)
    an = d["anomaly"]
    groups = [(~an, "#1f3b73", 3, 0.25, "normal"),
              (an & (d["direction"] > 0), "#d7301f", 10, 0.9, "too warm"),
              (an & (d["direction"] < 0), "#3182bd", 10, 0.9, "too cold")]
    if ylim is None and len(d):
        lo, hi = np.nanpercentile(d["z"], [0.5, 99.5])
        ylim = (min(lo, -1.3 * thr), max(hi, 1.3 * thr))

    fig, axes = plt.subplots(2, 1, figsize=(13, 9), gridspec_kw={"height_ratios": [1, 1.2]})

    # (1) z over time
    ax = axes[0]
    for m, color, s, alpha, lab in groups:
        ax.scatter(d.index[m], d.loc[m, "z"], s=s, c=color, alpha=alpha, linewidths=0,
                   label=f"{lab} (n={int(m.sum())})")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d", tz=data.index.tz))
    ax.set_title(f"z-score on working days (n={len(d)} scored samples)")

    # (2) z vs time of day
    ax = axes[1]
    to_h = lambda hhmm: int(hhmm.split(":")[0]) + int(hhmm.split(":")[1]) / 60
    ax.axvspan(to_h(cfg["occupied_start"]), to_h(cfg["occupied_end"]), color="#f2f2f2",
               lw=0, zorder=0, label="occupied hours")
    for m, color, s, alpha, lab in groups:
        ax.scatter(tod[m], d.loc[m, "z"], s=s, c=color, alpha=alpha, linewidths=0)
    q = d["z"].groupby(tod).quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
    ax.fill_between(q.index, q[0.05], q[0.95], color="#ff9f1c", alpha=0.15, lw=0,
                    label="5-95 %")
    ax.fill_between(q.index, q[0.25], q[0.75], color="#ff9f1c", alpha=0.35, lw=0,
                    label="25-75 %")
    ax.plot(q.index, q[0.5], color="#c46a00", lw=1.8, label="median")
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlabel("Time of day")
    ax.set_title("Distribution of z-score across the day (working days)")

    for ax in axes:
        for y in (-thr, thr):
            ax.axhline(y, color="#808080", lw=0.8, ls="--")
        ax.axhline(0, color="#808080", lw=0.6)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_ylabel("z")
        ax.grid(True, color="#e5e5e5", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper right", ncol=5, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_anomaly_hours(data, config=None):
    """
    Anomalies in occupied hours per hour of day: stacked counts by direction
    (bars) and anomaly rate = anomalies / scored samples in that hour (line).
    """
    cfg = _merge_config(config)
    d = data.loc[data["occupied"]]
    if d.empty:
        raise ValueError("No occupied samples to plot.")
    hours = range(d.index.hour.min(), d.index.hour.max() + 1)
    hour = d.index.hour
    count = lambda m: m.groupby(hour).sum().reindex(hours, fill_value=0)
    warm = count(d["anomaly"] & (d["direction"] > 0))
    cold = count(d["anomaly"] & (d["direction"] < 0))
    scored = count(d["z"].notna())
    rate = (warm + cold) / scored.replace(0, np.nan) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.array(list(hours))
    ax.bar(x, warm, color="#d7301f", width=0.8, label=f"too warm (n={int(warm.sum())})")
    ax.bar(x, cold, bottom=warm, color="#3182bd", width=0.8,
           label=f"too cold (n={int(cold.sum())})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:02d}-{h + 1:02d}" for h in x], fontsize=8)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Anomalous samples")
    ax.grid(True, axis="y", color="#e5e5e5", lw=0.6)
    ax.spines[["top"]].set_visible(False)

    ax2 = ax.twinx()
    ax2.plot(x, rate, color="#222222", marker="o", lw=1.5, label="anomaly rate")
    ax2.set_ylabel("Anomaly rate [% of scored samples]")
    ax2.set_ylim(bottom=0)
    ax2.spines[["top"]].set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8, frameon=False)
    ax.set_title(f"Occupied-hour anomalies by hour of day "
                 f"({int(warm.sum() + cold.sum())} of {int(scored.sum())} scored samples)")
    fig.tight_layout()
    return fig


def _plot_T_in(ax, d):
    ax.plot(d.index, d["T_in"], color="#1f3b73", lw=1.5, label="T_in")
    ax.plot(d.index, d["setpoint"], color="#808080", lw=1.5, ls="--", label="setpoint",
            drawstyle="steps-post")
    ax.set_ylabel("°C")


def _shade_recovery(ax, d):
    ax.fill_between(d.index, 0, 1, where=d["recovery"].astype(bool),
                    transform=ax.get_xaxis_transform(), color="#cccccc", alpha=0.4, lw=0, step="post", label="recovery")


def _shade_events(ax, events, start, end, step, alpha=0.2, label=True):
    """Shade events overlapping [start, end]; red = too warm, blue = too cold."""
    if events is None or not len(events):
        return
    ev = events[(events["end"] >= start) & (events["start"] <= end)]
    shown = set()
    for _, e in ev.iterrows():
        color = "#d7301f" if e["direction"] == "too warm" else "#3182bd"
        lab = e["direction"] if label and e["direction"] not in shown else None
        shown.add(e["direction"])
        ax.axvspan(e["start"], e["end"] + step, color=color, alpha=alpha, lw=0, label=lab)


def _plot_context_panels(axes, d):
    """Three panels: T_out & solar (twin axis), heating/cooling/vav, co2."""
    ax = axes[0]
    ax.plot(d.index, d["T_out"], color="#1f3b73", lw=1.2, label="T_out")
    ax.plot(d.index, d["T_out_3h"], color="#1f3b73", lw=1.2, ls=":", label="T_out_3h")
    ax.set_ylabel("T_out [°C]", color="#1f3b73")
    ax2 = ax.twinx()
    ax2.fill_between(d.index, 0, d["solar"].fillna(0), color="#f0a830", alpha=0.35, lw=0,
                     label="solar")
    ax2.set_ylabel("solar [W/m²]", color="#b8741a")
    ax2.set_ylim(bottom=0)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", ncol=3, fontsize=8, frameon=False)

    ax = axes[1]
    for c, color in [("heating", "#d7301f"), ("cooling", "#3182bd"), ("vav", "#555555")]:
        ax.plot(d.index, d[c], color=color, lw=1.2, label=c)
    ax.set_ylabel("%")
    ax.legend(loc="upper left", ncol=3, fontsize=8, frameon=False)

    ax = axes[2]
    ax.plot(d.index, d["co2"], color="#555555", lw=1.2)
    ax.set_ylabel("CO₂ [ppm]")


def _finish_time_axes(fig, axes, tz):
    for ax in axes:
        ax.grid(True, color="#e5e5e5", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=tz))
    fig.autofmt_xdate()
    fig.tight_layout()


def plot_period(data, events, start, end):
    """
    Four stacked panels sharing x for [start, end]:
      (1) T_in & setpoint with events shaded, (2) T_out & solar (twin axis),
      (3) heating, cooling, vav, (4) co2.
    `start`/`end` may be strings; they are interpreted in the data's time zone.
    """
    tz = data.index.tz
    start, end = _localize(start, tz), _localize(end, tz)
    d = data.loc[start:end]
    step = d.index.to_series().diff().median() if len(d) > 1 else pd.Timedelta("15min")
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.3, 1.3, 1]})

    ax = axes[0]
    _plot_T_in(ax, d)
    _shade_recovery(ax, d)
    _shade_events(ax, events, start, end, step)
    ax.set_title(f"07-D2-02: {start:%Y-%m-%d %H:%M} – {end:%Y-%m-%d %H:%M}")
    ax.legend(loc="upper left", ncol=5, fontsize=8, frameon=False)

    _plot_context_panels(axes[1:], d)
    _finish_time_axes(fig, axes, tz)
    return fig


def pick_anomaly_days(data, config=None, min_scored_frac=0.8, T_out_tol=2.0, solar_tol=0.3):
    """
    Return (worst_day, best_day, per_day table)
    worst = most anomalous samples; best = fewest anomalies among days with at
    least min_scored_frac of the median number of scored samples per working
    day and similar outdoor weather to the
    worst day: mean T_out within T_out_tol °C and mean solar within solar_tol
    (fraction) of the worst day's. Ties broken by weather distance, then lowest
    mean |z|.
    """
    cfg = _merge_config(config)
    d = data.loc[_working_days(data, cfg)]
    # d = d[d.index.month.isin((7, 8, 9))]
    # d = d[d.index.weekday.isin((2, 3, 4, 5, 6, 7))]
    day = pd.Index(d.index.date, name="date")
    per_day = pd.DataFrame({
        "anomalies": d["anomaly"].groupby(day).sum().astype(int),
        "too_warm": (d["anomaly"] & (d["direction"] > 0)).groupby(day).sum().astype(int),
        "too_cold": (d["anomaly"] & (d["direction"] < 0)).groupby(day).sum().astype(int),
        "scored": d["z"].notna().groupby(day).sum().astype(int),
        "mean_abs_z": d["z"].abs().groupby(day).mean(),
        "T_out_mean": d["T_out"].groupby(day).mean(),
        "solar_mean": d["solar"].groupby(day).mean(),
    })
    worst = per_day["anomalies"].idxmax()
    ref = per_day.loc[worst]
    # weather distance to the worst day, 1 = at the edge of the tolerance
    dT = (per_day["T_out_mean"] - ref["T_out_mean"]) / T_out_tol
    dS = (per_day["solar_mean"] - ref["solar_mean"]) / max(solar_tol * ref["solar_mean"], 1.0)
    per_day["weather_dist"] = np.sqrt(dT**2 + dS**2)

    cand = per_day[(per_day["scored"] >= min_scored_frac * per_day["scored"].median())
                   & (per_day.index != worst) & per_day["weather_dist"].notna()]
    similar = cand[(dT.loc[cand.index].abs() <= 1) & (dS.loc[cand.index].abs() <= 1)]
    if similar.empty:
        print(f"pick_anomaly_days: no day within T_out ±{T_out_tol} °C and solar ±{solar_tol:.0%} "
              f"of {worst}; using the 5 most weather-similar days")
        similar = cand.nsmallest(5, "weather_dist")
    best = similar.sort_values(["anomalies", "weather_dist", "mean_abs_z"]).index[0]
    return worst, best, per_day


def plot_anomaly_days(data, config=None):
    """
    Print the working day with the most anomalies and the one with the fewest,
    then plot them side by side.
    """
    cfg = _merge_config(config)
    worst, best, per_day = pick_anomaly_days(data, cfg)
    print("\nAnomalies per working day: most / fewest")
    print(per_day.loc[[worst, best]].rename(index={worst: f"{worst} (most)", best: f"{best} (fewest)"})
          .round(2).to_string())

    tz = data.index.tz
    step = pd.Timedelta(cfg["resample_freq"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex="col", sharey="row",
                             gridspec_kw={"height_ratios": [1.3, 1]})
    sol_axes = []
    for col, (day, tag) in enumerate([(worst, "most anomalies"), (best, "fewest anomalies")]):
        start = pd.Timestamp(day).tz_localize(tz)
        d = data.loc[start:start + pd.Timedelta("1D") - step]
        warm = d["anomaly"] & (d["direction"] > 0)
        cold = d["anomaly"] & (d["direction"] < 0)

        ax = axes[0, col]
        _plot_T_in(ax, d)
        ax.scatter(d.index[warm], d.loc[warm, "T_in"], s=18, c="#d7301f", zorder=3,
                   label="anomaly: too warm" if warm.any() else None)
        ax.scatter(d.index[cold], d.loc[cold, "T_in"], s=18, c="#3182bd", zorder=3,
                   label="anomaly: too cold" if cold.any() else None)
        ax.plot(d.index, d["T_out"], color="#2a9d8f", lw=1.2, label="T_out")
        ax_sol = ax.twinx()
        ax_sol.fill_between(d.index, 0, d["solar"].fillna(0), color="#f0a830", alpha=0.3, lw=0,
                            label="solar")
        ax_sol.set_ylabel("solar [W/m²]", color="#b8741a")
        # keep the temperatures on top of the solar fill
        ax.set_zorder(ax_sol.get_zorder() + 1)
        ax.patch.set_visible(False)
        sol_axes.append(ax_sol)
        ax.set_title(f"{day:%a %Y-%m-%d}: {tag} ({int(d['anomaly'].sum())} anomalous, "
                     f"{int(d['z'].notna().sum())} scored samples)")

        ax = axes[1, col]
        # vav dashed underneath so cooling stays visible when both sit at 100 %
        ax.plot(d.index, d["vav"], color="#555555", lw=1.2, ls="--", label="vav", zorder=1)
        ax.plot(d.index, d["heating"], color="#d7301f", lw=1.4, label="heating", zorder=2)
        ax.plot(d.index, d["cooling"], color="#3182bd", lw=1.4, label="cooling", zorder=2)
        ax.set_ylabel("%")

        for ax in axes[:, col]:
            # _shade_recovery(ax, d)
            ax.grid(True, color="#e5e5e5", lw=0.6, linestyle='--')
            ax.spines["top"].set_visible(False)
        axes[1, col].spines["right"].set_visible(False)
        axes[1, col].legend(loc="upper left", ncol=3, fontsize=7, frameon=False)
        handles, labels = [], []
        for a in (axes[0, col], ax_sol):
            h, l = a.get_legend_handles_labels()
            handles += h
            labels += l
        axes[0, col].legend(handles, labels, loc="upper left", ncol=3, fontsize=7, frameon=False)
        axes[1, col].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
        axes[1, col].set_xlabel("Time of day")
    # same solar scale in both columns
    top = max(a.get_ylim()[1] for a in sol_axes)
    for a in sol_axes:
        a.set_ylim(0, top)
    fig.tight_layout()
    return fig


# Step 7: summary
# def _print_regime_status(data):
#     print("\n" + "=" * 78 + "\nSUMMARY\n" + "=" * 78)
#     print("Share of samples per regime:")
#     print(data["regime"].value_counts(normalize=True).mul(100).round(1).to_string())
#     print("\nStatus per regime [% of samples]:")
#     print((pd.crosstab(data["regime"], data["status"], normalize="index") * 100)
#           .round(2).to_string())


# def _print_events(events, cfg, show, score_col, score_label):
#     """Event counts per regime/direction and hint, then the top events by `score_col`."""
#     if events is None or events.empty:
#         print("\nNo events detected.")
#         return
#     print(f"\nEvents (>= {cfg['event_min_duration']}): {len(events)}")
#     print(pd.crosstab(events["regime"], events["direction"], margins=True).to_string())
#     print("\nEvents per hint category (an event can have several hints):")
#     print(events["hint"].str.split("; ").explode().value_counts().to_string())

#     n = cfg["top_n_events"]
#     top = events.sort_values(score_col, ascending=False).head(n)
#     num = top.select_dtypes("number", exclude="timedelta").columns
#     top[num] = top[num].round(2)
#     print(f"\nTop {n} events by {score_label}:")
#     with pd.option_context("display.width", 250, "display.max_colwidth", 80):
#         print(top[show + [score_col]].to_string())


# def print_summary(data, bin_stats, events, config=None):
#     """Print regime shares, flag rates, event counts per direction/hint and the top events."""
#     cfg = _merge_config(config)
#     _print_regime_status(data)
#     for reg in REGIMES:
#         d = data[data["regime"] == reg]
#         scored = d["z"].notna().sum()
#         print(f"\n{reg}: {int(d['anomaly'].sum())} flagged of {scored} scored samples "
#               f"({d['anomaly'].sum() / max(scored, 1):.2%}); "
#               f"bins sufficient: {int(bin_stats.loc[bin_stats['regime'] == reg, 'sufficient'].sum())}"
#               f"/{(bin_stats['regime'] == reg).sum()}")

#     show = ["start", "duration", "regime", "direction", "max_abs_dev", "mean_dev",
#             "T_out_3h_mean", "solar_2h_mean", "heating_mean", "cooling_mean", "vav_mean",
#             "co2_mean", "hint"]
#     _print_events(events, cfg, show, "score", "max|dev| * duration [°C·h]")


# main workflow
def run_analysis(dfs, config=None, strict=True):
    cfg = _merge_config(config)
    # issues = inspect_inputs(dfs, cfg)
    # errors = [m for lvl, m in issues if lvl == "ERROR"]
    # if errors and strict:
    #     raise ValueError("Input inspection found errors; fix them or pass strict=False:\n  "
    #                      + "\n  ".join(errors))

    data, _ = combine_data(dfs, cfg)
    data = preprocess(data, cfg)
    data = assign_bins(data, cfg)
    bin_stats = compute_bin_stats(data, cfg)
    data = flag_anomalies(data, bin_stats, cfg)
    data, events = detect_events(data, cfg)
    # print_summary(data, bin_stats, events, cfg)

    if cfg["make_plots"]:
        figure_dir = Path("./figure")
        figure_dir.mkdir(parents=True, exist_ok=True)
        # plot_dev_heatmap(data).savefig(figure_dir / "dev_heatmap.png", dpi=300,
        #                                bbox_inches="tight")
        prefix = f"{cfg['target_col']}_{cfg['bin_col']}_{cfg['solar_col']}"
        plot_dev_scatter(data, cfg).savefig(figure_dir / f"{prefix}_scatter.png", dpi=300,
                                            bbox_inches="tight")
        plot_anomaly_days(data, cfg).savefig(figure_dir / f"{prefix}_anomaly_days.png",
                                             dpi=300, bbox_inches="tight")
        plot_bin_grids(bin_stats, cfg).savefig(figure_dir / f"{prefix}_bin_grids.png", dpi=300,
                                                bbox_inches="tight")
        # plot_z_working_days(data, cfg).savefig(figure_dir / "z_working_days.png", dpi=300,
        #                                        bbox_inches="tight")
        # plot_anomaly_hours(data, cfg).savefig(figure_dir / "anomaly_hours.png", dpi=300,
        #                                       bbox_inches="tight")
        plt.show()
    return data, bin_stats, events


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description="Run the anomaly detection pipeline on CSV files.")
    # parser.add_argument("files", nargs="+", help="CSV files to process")
    # parser.add_argument("--strict", action="store_true", help="Stop on input errors")
    # args = parser.parse_args()
    import pickle

    with open("./07-D2-02_data/dfs.pkl", "rb") as f:
        dfs = pickle.load(f)

    run_analysis(dfs, strict=True)
