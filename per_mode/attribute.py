import sys
from pathlib import Path
import numpy as np
import pandas as pd
from prep import MODES, DATA, RES
import study as S


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


GROUP_ORDER = ["chilled_beam", "building_mass", "schedule", "solar", "envelope", "occupancy"]
SEASON = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
          6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}


def decompose(xg, y, X, info):
    parts = []
    fit_rows = info.split.isin(["train", "val"])
    for mode in MODES + ["other"]:
        mdl = xg[mode] if mode in xg else xg["global"]
        s = info.mode_used == mode
        fr = (fit_rows & s) if mode in xg else fit_rows
        g_all = mdl.group_contributions(X.loc[s, mdl.cols])
        g_fit = mdl.group_contributions(X.loc[fr, mdl.cols])
        mean_fit = g_fit.mean()
        c = g_all.drop(columns="_intercept") - mean_fit.drop("_intercept")
        c["baseline"] = mean_fit.sum()
        parts.append(c)
    D = pd.concat(parts).reindex(y.index).fillna(0.0)
    D = D[["baseline"] + [g for g in GROUP_ORDER if g in D]]
    D["pred"] = D.sum(axis=1)
    D["obs"] = y
    D["mode"] = info.mode_used
    return D


def main():
    G, feats, y, X, info, mono, groups = S.load()
    F = pd.read_pickle(RES / "final_models.pkl")
    xg = F["xg"]
    D = decompose(xg, y, X, info)
    D.to_pickle(RES / "decomposition.pkl")
    comps = [g for g in GROUP_ORDER if g in D]

    # which modes move the room: K/day from mode baselines, by season
    day = D.assign(season=D.index.month.map(SEASON), date=D.index.normalize())
    n_days = day.groupby("season").date.nunique()
    by_mode = day.groupby(["season", "mode"]).pred.sum().unstack().div(n_days, axis=0)    # model, K/day
    by_mode_obs = day.groupby(["season", "mode"]).obs.sum().unstack().div(n_days, axis=0)      # measured
    # by_mode_obs.to_csv(RES / "mode_K_per_day_measured.csv")
    hours = (day.groupby(["season", "mode"]).size().unstack() / 2).div(n_days, axis=0)
    print("MODE effect (model, K/day; measured in second table; hours/day in brackets)")
    print((by_mode.round(2).astype(str) + " (" + hours.round(1).astype(str) + "h)").to_string())
    print(by_mode_obs.round(2).to_string())
    # by_mode.to_csv(RES / "mode_K_per_day.csv")
    hours.to_csv(RES / "mode_hours_per_day.csv")

    # which components move the room within a mode
    absc = D.groupby("mode")[comps].apply(lambda d: d.abs().mean() * 2)        # K/h
    print("\nmean |contribution| per step, K/h")
    print(absc.round(3).to_string())
    absc.to_csv(RES / "component_abs_Kph_by_mode.csv")
    comp_season = day.groupby("season")[comps].sum().div(n_days, axis=0)
    print("\ncomponent push vs typical, K/day by season")
    print(comp_season.round(2).to_string())
    comp_season.to_csv(RES / "component_K_per_day_by_season.csv")

    # sanity checks
    chk = []
    from prep import fm
    raw = fm.load(DATA).reindex(info.index)
    Q = raw.cooling_flow / 60 * 4186 * (raw.KB02_return - raw.KB02_supply)     # measured beam power, W
    for mode in MODES:
        s = D["mode"] == mode
        x = X[s]
        def slope(c, v):
            v = v.values; c = c.values
            ok = np.isfinite(v) & np.isfinite(c)
            return np.polyfit(v[ok], c[ok], 1)[0] if ok.sum() > 50 and np.std(v[ok]) > 0 else np.nan
        chk.append(dict(
            mode=mode,
            solar_K_per_h_per_100Wm2=slope(D.loc[s, "solar"], x.solar_radiation_ma4_lag1 / 100) * 2,
            envelope_K_per_h_per_K=slope(D.loc[s, "envelope"], x.outdoor_temperature_ma6_lag1) * 2,
            mass_slope_per_h=slope(D.loc[s, "building_mass"], x.indoor_temperature_lag1) * 2,
            beam_vs_Qbeam_corr=float(np.corrcoef(D.loc[s, "chilled_beam"], Q[s].fillna(0))[0, 1])
            if D.loc[s, "chilled_beam"].std() > 0 else np.nan,
        ))
    chk = pd.DataFrame(chk).set_index("mode")
    chk["mass_time_constant_h"] = -1 / chk.mass_slope_per_h
    print("\nSANITY")
    print(chk.round(4).to_string())
    chk.to_csv(RES / "sanity_checks.csv")

    # daily closure: predicted vs observed daily net change
    dd = D.groupby(D.index.normalize())[["pred", "obs"]].sum()
    print("\ndaily net change: corr(pred, obs) = %.2f, mean obs %.3f, mean pred %.3f K/day"
          % (dd.corr().iloc[0, 1], dd.obs.mean(), dd.pred.mean()))
    return D


if __name__ == "__main__":
    main()
