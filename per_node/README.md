# Per-mode study of the indoor-temperature mechanism (room 05-A1-01)

Uses `filter_modes.py` and `GCM_generation.py` from the main folder unchanged. All paths are set at the top of
`prep.py` (`ROOM = "05-A1-01"`), relative to the main folder: data from `<ROOM>_data/dfs.pkl`, results to
`results/<ROOM>_per_node/`, figures to `figure/<ROOM>_per_node_*.png`.

## Run order
```
python prep.py        # modes (your filter) + design matrix (your graph's node_frame)  -> <ROOM>_data/prepared.pkl
python study.py all   # tuning, final fit, assembly comparison, 24 h free-run          -> results/<ROOM>_per_node/
python attribute.py   # exact group decomposition + sanity checks                       -> results/<ROOM>_per_node/
python plots.py       # figures                                                         -> figure/
```
`python study.py final` reuses the tuned settings in `results/<ROOM>_per_node/best_params.json`.

## Set-up
* Target: the graph's `indoor_temperature` mechanism, i.e. y = T(t) - T(t-1) from inputs at t-1 (+ calendar).
  Each step belongs to the mode at t-1, the interval in which its inputs act. Steps outside the runs your
  filter saves (runs < 4 h, unclassified) are labelled `other` (mostly start-up / shut-down transitions).
* Split: train 2022-03..2023-06, validation 2023 H2 (tuning, early stopping, blend weights),
  final refit on train + validation, test = 2024 (never used for fitting or choices).
* Models: sign-constrained linear (graph signs; adds valve x water-dT and damper x air-dT, which the
  chilled_beam group allows), XGBoost with the graph's monotone + interaction (group) constraints,
  unconstrained XGBoost only as a reference.
* Structural zeros: in a mode where an emitter is off by definition (beam water in air-only, beam water and
  air in the off modes) its columns are left out. Validation cost < 1 %; avoids spurious attributions.
* Assembly compared: global model; switched (model of the step's mode, global for `other`);
  switched + a model for `other`; per-mode blend of switched and global.

## Main results (2024 test)
| | one-step R² | one-step RMSE (K/30 min) | 24 h free-run RMSE (K) |
|---|---|---|---|
| **Switched XGBoost + transition model** | **0.428** | **0.150** | **0.55** |
| Blend switched/global XGBoost | 0.425 | | 0.55 |
| Global XGBoost | 0.382 | 0.155 | 0.62 |
| Switched linear | 0.203 | | 0.64 |
| No change / persistence | 0 | 0.198 | 1.37 |

## Files in results/<ROOM>_per_node/
`tuning.csv`, `best_params.json`, `onestep_test.csv`, `freerun_test.csv`, `mode_K_per_day*.csv`,
`component_*.csv`, `sanity_checks.csv`, `final_models.pkl` (fitted models), `decomposition.pkl` (per-step split).
