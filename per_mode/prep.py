import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent     # main project folder
USER = ROOT


ROOM = "07-D2-02"
DATA_DIR = ROOT / f"{ROOM}_data"
DATA = DATA_DIR / "dfs.pkl"
RES = ROOT / "results" / f"{ROOM}_per_mode"
FIG = ROOT / "figure"
FIG_PREFIX = f"{ROOM}_per_mode_"


def _import(name):
    spec = importlib.util.spec_from_file_location(name, USER / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


fm = _import("filter_modes")
gcm = _import("GCM_generation")
T = gcm.T
MODES = list(fm.MODE_NAMES)      # beam_water_air, beam_air, off_no_sun, off_sunny

# training/validation/test split
TRAIN_END, VAL_END = "2023-06-30 23:30", "2023-12-31 23:30"


def build(data_path):
    raw = fm.load(data_path)

    # modes
    md = fm.assign_modes(fm.add_features(raw.copy()).loc[fm.START:fm.END])
    md["in_mode_file"] = md["mode"].isin(MODES) & (md.run_len_h >= fm.MIN_RUN_H) 

    # the causal graph and its feature columns
    G = gcm.build_gcm()
    feats = gcm.make_features(raw, G)
    y, X = gcm.node_frame(G, T, feats)
    keep = (y.index >= pd.Timestamp(fm.START)) & (y.index <= pd.Timestamp(fm.END))
    y, X = y[keep], X[keep]

    prev = md[["mode", "in_mode_file", "run_id", "t_in_mode_h"]].shift(1, freq="30min")   # mode at t-1
    prev = prev.reindex(y.index)
    info = pd.DataFrame(index=y.index)
    info["mode"] = prev["mode"]
    info["in_mode_file"] = prev["in_mode_file"].fillna(False).astype(bool)
    info["mode_used"] = info["mode"]
    info["run_id"] = prev["run_id"]
    info["t_in_mode_h"] = prev["t_in_mode_h"]
    info["split"] = np.where(y.index <= TRAIN_END, "train", np.where(y.index <= VAL_END, "val", "test"))
    # raw signals used for checks, interventions and simulation
    for c in ["indoor_temperature", "cooling_valve", "VAV_damper", "outdoor_temperature", # "heating_valve", 
              "solar_radiation", "setpoint", "KB02_supply", "AHU_supply_temperature", "cooling_flow"]:
        info[c] = raw[c].reindex(y.index)
    return G, feats, y, X, info


if __name__ == "__main__":
    G, feats, y, X, info = build(DATA)
    pd.set_option("display.width", 220)
    # print("design matrix columns:", list(X.columns))
    # print("monotone:", dict(zip(X.columns, gcm.monotone_constraints(G, T, list(X.columns)))))
    # print("groups:", {g: c for g, c in G.nodes[T]["groups"].items()})
    print(pd.crosstab(info.mode_used, info.split, margins=True))
    print("\ny (K per 30 min) by mode:")
    print(y.groupby(info.mode_used).describe()[["mean", "std", "min", "max"]].round(3))
    # pd.to_pickle(dict(y=y, X=X, info=info), DATA_DIR / "prepared.pkl")
