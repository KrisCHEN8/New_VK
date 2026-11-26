import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import holidays as hd

HERE = Path(__file__).resolve().parent


# Configuration
Country = "SE"  # for holidays
START, END = "2022-03-01", "2024-10-01"
BEAM_ON = 5.0  # cooling_valve  above which the chilled water counts as on, %
VAV_ON = 1.0    # VAV_damper % above which the primary air counts as ON
SUN_ON = 20.0   # solar_radiation W/m2 above which a period counts as "sunny"
MIN_RUN_H = 1.0   # keep only uninterrupted runs of the same mode at least this long
DT_H = 0.5   # sampling step in hours
CP_WATER = 4186.0  # J/(kg K), water density taken as 1 kg/l

MODE_NAMES = {
    "beam_water_air": "chilled-beam water + ventilation (VAV)",
    "beam_air": "primary air (VAV) only, beam valve closed",
    "off_no_sun": "all HVAC off, no sun -> outdoor + background only",
    "off_sunny": "all HVAC off, sunny -> outdoor + sun + background",
}


def load(path):
    d = pd.read_pickle(path)
    df = pd.concat({k: v["value"] for k, v in d.items()}, axis=1)
    df.index.name = "time"
    return df.sort_index()


def add_features(df):
    # chilled-beam cooling power [W]
    df["Q_beam_W"] = df.cooling_flow / 60.0 * CP_WATER * (df.KB02_return - df.KB02_supply)
    # ventilation cooling "drive" [K]
    df["vent_drive"] = df.VAV_damper / 100.0 * (df.indoor_temperature - df.AHU_supply_temperature)
    df["dT_out_in"] = df.outdoor_temperature - df.indoor_temperature
    df["hour_sin"] = np.sin(2 * np.pi * (df.index.hour + df.index.minute / 60) / 24)
    df["hour_cos"] = np.cos(2 * np.pi * (df.index.hour + df.index.minute / 60) / 24)
    hol = hd.country_holidays(Country, years=range(df.index.year.min(), df.index.year.max() + 1))
    df["holiday"] = df.index.normalize().isin(pd.to_datetime(list(hol)))
    df["month"] = df.index.month
    return df


def assign_modes(df):
    beam = df.cooling_valve > BEAM_ON
    vav = df.VAV_damper > VAV_ON
    sun = df.solar_radiation > SUN_ON
    df["mode"] = np.select(
        [beam & vav, ~beam & vav, ~beam & ~vav & ~sun, ~beam & ~vav & sun],
        ["beam_water_air", "beam_air", "off_no_sun", "off_sunny"],
        default="unknown",   # includes beam on with VAV off (rare)
    )
    # rows with missing key signals cannot be classified
    key = ["indoor_temperature", "cooling_valve", "VAV_damper",
           "outdoor_temperature", "solar_radiation"]
    df.loc[df[key].isna().any(axis=1), "mode"] = "unknown"

    # no residual class: every unknown row inherits the mode of the previous time step
    # (leading unknowns, which have no previous step, take the first valid mode instead)
    unknown = df["mode"] == "unknown"
    df["mode_filled"] = unknown
    df["mode"] = df["mode"].mask(unknown).ffill().bfill()

    # runs = uninterrupted stretches of one mode
    df["run_id"] = (df["mode"] != df["mode"].shift()).cumsum()
    g = df.groupby("run_id")
    df["run_len_h"] = g["mode"].transform("size") * DT_H
    df["t_in_mode_h"] = g.cumcount() * DT_H          # hours since this mode started (for rebound filtering)

    # # response variable: indoor temperature change over the NEXT step, in K/h.
    # # Inputs at time t act on T(t+DT) - T(t). Only valid if t+DT is still in the same run.
    # df["dTdt_Kph"] = (df.indoor_temperature.shift(-1) - df.indoor_temperature) / DT_H
    # df["step_valid"] = (df.run_id.shift(-1) == df.run_id) & df.dTdt_Kph.notna()
    return df


def main():
    p = argparse.ArgumentParser(description="Split operational data into different modes.")
    p.add_argument("room", nargs="?", default="07-D4-01",
                   help="room id (default: 07-D2-02)")
    args = p.parse_args()

    out = HERE / f"{args.room}_data"        # results go next to the room's dfs.pkl
    src = out / "dfs.pkl"
    if not src.exists():
        raise FileNotFoundError(f"{src} not found")

    df = load(src).loc[START:END]
    df = assign_modes(add_features(df))
    df.to_pickle(out / "all_with_modes.pkl")

    rows = []
    for mode, desc in MODE_NAMES.items():
        m = df[(df["mode"] == mode) & (df.run_len_h >= MIN_RUN_H)]
        m.to_pickle(out / f"mode_{mode}.pkl")
        rows.append(dict(
            mode=mode, description=desc,
            hours=len(m) * DT_H, occurrence=m.run_id.nunique(),
        ))
    summary = pd.DataFrame(rows).set_index("mode")
    # summary.to_csv(out / "mode_summary.csv")

    pd.set_option("display.width", 200)
    # print(f"Saved to {out.resolve()}  (runs shorter than {MIN_RUN_H} h dropped)\n")
    print(summary.drop(columns="description").round(2).to_string())
    print("\nunknown rows reassigned to previous mode:", int(df["mode_filled"].sum()))


if __name__ == "__main__":
    main()
