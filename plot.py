from __future__ import annotations

import argparse
import math
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
TZ = ZoneInfo("Europe/Stockholm")   # dfs.pkl timestamps are UTC; plot in local time

# Colours
BG = "#fafaf8"
GRID = "#e6e6e2"
SPINE = "#d6d6d1"
MUTED = "#6b6b66"
INK = "#1f1f1c"
WEEKEND = "#ecebe5"

# Each row: (y label, [(signal key in dfs.pkl, legend label, colour, line style)])
ROWS = [
    ("°C", [("indoor_temperature", "Indoor temperature", "#2f74d0", "-"),
            ("setpoint", "Setpoint", "#4a4a46", "--")]),
    ("°C", [("outdoor_temperature", "Outdoor temperature", "#e0663a", "-")]),
    ("W/m²", [("solar_radiation", "Solar radiation", "#e3a21a", "-")]),
    ("% open", [("cooling_valve", "Chilled beam valve", "#1baf7a", "-"),
                ("VAV_damper", "VAV damper", "#6a4fd0", "-"),
                ("heating_valve", "Radiator valve", "#e0663a", "-")]),
]
SIGNALS = [key for _, series in ROWS for key, *_ in series]


# Loading
def load_room(room: str, base: Path = HERE) -> pd.DataFrame:
    """Load <room>_data/dfs.pkl into one DataFrame (local time) with the plotted signals."""
    path = base / f"{room}_data" / "dfs.pkl"
    dfs = pd.read_pickle(path)
    missing = [k for k in SIGNALS if k not in dfs]
    if missing:
        print(f"[warn] {path}: missing signals {missing}, they are left out of the plot")
    df = pd.concat({k: dfs[k]["value"] for k in SIGNALS if k in dfs}, axis=1).sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(TZ)
    return df


def week_start(date: str) -> pd.Timestamp:
    """Monday 00:00 local time of the week containing `date`."""
    d = pd.Timestamp(date).normalize()
    return (d - pd.Timedelta(days=d.weekday())).tz_localize(TZ)


def week_label(name: str, start: pd.Timestamp) -> str:
    end = start + pd.Timedelta(days=6)
    if start.month == end.month:
        span = f"{start.day}–{end.day} {end:%b %Y}"
    elif start.year == end.year:
        span = f"{start.day} {start:%b} – {end.day} {end:%b %Y}"
    else:
        span = f"{start.day} {start:%b %Y} – {end.day} {end:%b %Y}"
    return f"{name} week ({span})"


# Styling
def style_axis(ax):
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, color=SPINE)
    ax.yaxis.label.set_color(MUTED)


def shade_weekends(ax, start: pd.Timestamp, end: pd.Timestamp):
    for day in pd.date_range(start, end, freq="D", inclusive="left"):
        if day.weekday() >= 5:
            ax.axvspan(day, day + pd.Timedelta(days=1), color=WEEKEND, lw=0, zorder=0)


def day_formatter(x, _pos):
    d = mdates.num2date(x, tz=TZ)
    return f"{d:%a} {d.day}"


# Plot
def plot_weeks(df: pd.DataFrame, room: str, weeks: list[tuple[str, pd.Timestamp]]):
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Inter", "Segoe UI", "Arial", "DejaVu Sans"]})

    fig, axes = plt.subplots(len(ROWS), len(weeks), figsize=(14, 9.5), sharex="col",
                             squeeze=False, facecolor=BG)
    fig.subplots_adjust(left=0.05, right=0.84, top=0.91, bottom=0.06, hspace=0.12, wspace=0.12)

    slices = []
    for c, (name, start) in enumerate(weeks):
        end = start + pd.Timedelta(days=7)
        w = df.loc[start:end]
        if w.empty:
            print(f"[warn] {room}: no data for {name.lower()} week starting {start:%Y-%m-%d}")
        slices.append(w)

        for r, (ylabel, series) in enumerate(ROWS):
            ax = axes[r, c]
            style_axis(ax)
            shade_weekends(ax, start, end)
            for key, label, colour, ls in series:
                if key in w:
                    ax.plot(w.index, w[key], color=colour, ls=ls, lw=1.6 if ls == "-" else 1.2,
                            label=label)
            ax.set_ylabel(ylabel)
            ax.set_xlim(start, end)

        axes[0, c].set_title(week_label(name, start), loc="left", fontsize=11,
                             fontweight="bold", color=INK)
        bottom = axes[-1, c]
        bottom.xaxis.set_major_locator(mdates.DayLocator(tz=TZ))
        bottom.xaxis.set_major_formatter(FuncFormatter(day_formatter))

    # Indoor-temperature row: same y range in all columns so weeks compare directly
    temp_cols = [k for k, *_ in ROWS[0][1]]
    vals = pd.concat([w[[k for k in temp_cols if k in w]] for w in slices]).stack()
    if not vals.empty:
        lo, hi = math.floor(vals.min() - 0.5), math.ceil(vals.max() + 0.5)
        for ax in axes[0]:
            ax.set_ylim(lo, hi)

    # Actuator row: fixed 0–100 %
    for ax in axes[-1]:
        ax.set_ylim(-4, 104)
        ax.set_yticks(range(0, 101, 20))

    # Legends to the right of the last column, one per row
    for r in range(len(ROWS)):
        ax = axes[r, -1]
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.03, 1.0),
                      frameon=False, fontsize=9.5, labelcolor=INK, handlelength=2)

    fig.text(0.05, 0.965, f"Room {room}: indoor temperature, weather and HVAC signals "
                          f"(grey = weekend)", ha="left", va="center", fontsize=13, color=INK)
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("room", nargs="?", default="07-D2-02",
                   help="room id, e.g. 07-D2-02 (reads <room>_data/dfs.pkl; default %(default)s)")
    p.add_argument("--winter", default="2023-01-16", help="any date in the winter week (default %(default)s)")
    p.add_argument("--summer", default="2023-06-12", help="any date in the summer week (default %(default)s)")
    p.add_argument("--out", type=Path, default=None,
                   help="output image (default figure/<room>_week_overview.png)")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--show", action="store_true", help="also open the figure window")
    args = p.parse_args()

    df = load_room(args.room)
    weeks = [("Winter", week_start(args.winter)), ("Summer", week_start(args.summer))]
    fig = plot_weeks(df, args.room, weeks)

    out = args.out or HERE / "figure" / f"{args.room}_week_overview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, facecolor=fig.get_facecolor())
    print(f"saved {out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
