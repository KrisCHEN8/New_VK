# Per-mode study of the indoor-temperature mechanism (room 05-A1-01)

Uses `filter_modes.py` and `GCM_generation.py` from the main folder unchanged. All paths are set at the top of
`prep.py` (`ROOM = "05-A1-01"`), relative to the main folder: data from `<ROOM>_data/dfs.pkl`, results to
`results/<ROOM>_per_node/`, figures to `figure/<ROOM>_per_node_*.png`.

## Run order
```
python prep.py        # modes (your filter) + design matrix (your graph's node_frame)
python study.py all   # tuning, final fit, 24 h free-run + 3 h recursive evaluation  -> results/<ROOM>_per_node/
python attribute.py   # exact group decomposition + sanity checks                     -> results/<ROOM>_per_node/
python plots.py       # figures                                                       -> figure/
```
`python study.py final` reuses the tuned settings in `results/<ROOM>_per_node/best_params.json`.

## Set-up
* Target: the graph's `indoor_temperature` mechanism, i.e. y = T(t) - T(t-1) from inputs at t-1 (+ calendar).
  Each step belongs to the mode at t-1, the interval in which its inputs act. Steps outside the runs your
  filter saves (runs < 4 h, unclassified) are labelled `other` (mostly start-up / shut-down transitions).
* Split: train 2022-03..2023-06, validation 2023 H2 (tuning, early stopping),
  final refit on train + validation, test = 2024 (never used for fitting or choices).
* Model: XGBoost with the graph's monotone + interaction (group) constraints.
* Structural zeros: in a mode where an emitter is off by definition (beam water in air-only, beam water and
  air in the off modes) its columns are left out. Validation cost < 1 %; avoids spurious attributions.
* Methods compared:
  * **Persistence**: temperature stays at its value at the start.
  * **Global XGBoost**: one model for all steps.
  * **Switched XGBoost + transition model**: the model of the step's mode, and an `other` model for transition steps.
* Evaluation (2024, recursive: predicted T fed back, valves/dampers/weather/calendar measured):
  * **24 h free-run**, restarted every midnight (48 steps).
  * **3 h recursive forecast**, restarted at every 30-min step (6 steps).

## Main results (2024 test)
| | 3 h recursive RMSE (K) | 24 h free-run RMSE (K) |
|---|---|---|
| **Switched XGBoost + transition model** | *re-run* | 0.55 |
| Global XGBoost | *re-run* | 0.62 |
| Persistence | *re-run* | 1.37 |

## Files in results/<ROOM>_per_node/
`tuning.csv`, `best_params.json`, `final_models.pkl` (fitted models),
`freerun_test.{pkl,csv}`, `recursive3h_test.{pkl,csv}` (csv = overall RMSE and MAE per method),
`mode_K_per_day*.csv`, `component_*.csv`, `sanity_checks.csv`, `decomposition.pkl` (per-step split).
