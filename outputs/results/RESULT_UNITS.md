# Result units and group codes

The paper reports soil organic matter in g kg−1. Several files still store the recorded mass percent (g per 100 g). Multiply those columns by 10. Do not multiply `q_log`, DI, proportions, counts, or columns that are already in g kg−1.

Group codes `outer_group`, `test_group` and `cal_group` use 0, 1, 2, 3 and 4 for the west-to-east groups G1–G5 in the paper. The same codes are the keys of `group_metrics` in `nested_selection.json`.

## Columns stored as recorded percent (multiply by 10)

| File | Columns |
| --- | --- |
| `outputs/maps/lattice_predictions.csv` | `OM`, `OM_all`, `lo_abs`, `hi_abs`, `lo_log`, `hi_log` |
| `outputs/results/conformal_points.csv` | `pred`, `lo_abs`, `hi_abs`, `lo_log`, `hi_log`, `is_abs`, `is_log` |
| `outputs/results/conformal_rotation20.csv` | `q_abs`, `rmse_test`, `w_abs_cfg`, `w_log_cfg`, `is_abs`, `is_log` |
| `outputs/results/conformal_summary.csv` | `q_abs`, `rmse_test`, `w_abs`, `w_log`, `is_abs`, `is_log` |
| `outputs/models/deployment_linear.json` | `q_abs`, `cal_rmse` |
| `outputs/results/runs_summary.csv` | `rmse`, `mae`, `bias`, `rmse_g0`–`rmse_g4` |
| `outputs/results/nested_selection.json` | `rmse`, `mae` and `bias` in `metrics` and `group_metrics` |
| `outputs/results/permutation_null_rmse.csv` | `rmse` |

`lo_log` and `hi_log` are interval limits in concentration units, not logarithms, so they take the factor of 10. `w_log`, `w_log_cfg` and the interval scores `is_abs` and `is_log` are in the same recorded-percent units.

## Do not multiply by 10

- `q_log` in the conformal files and in `deployment_linear.json` (log-ratio score)
- `DI`, correlations, *R*² and concordance
- coverage proportions (`cov_abs`, `cov_log`, `cov_abs_cfg`, `cov_log_cfg`) and integer coverage counts (`n_cov_abs`, `n_cov_log`)
- `nested_selection_oof.csv` column `prediction`, which is already in g kg−1
- `nested_selection_log.csv` columns `candidate_rmse_g_kg` and `reference_rmse_g_kg`
- `permutation_null_rmse.csv` column `rmse_g_kg`
- `map_fit_stability.json` differences `diff_*` and quantiles `q05`–`q95`, which were scaled to g kg−1 when the file was written
- `ac_changed` and `ac_changed_within_AC`, which are proportions

## Class-transition denominators

`map_fit_stability.json` keeps both rates. The manuscript uses the second.

| Field | Numerator / denominator | Value |
| --- | --- | --- |
| `ac_changed` | 460 transitions / 3546 predictable nodes | 0.129724 |
| `ac_changed_within_AC` | 460 transitions / 2216 nodes that were class A or C | 0.207581 |

The separate class A rate in the manuscript is 427/811. Do not replace the 20.8% class A or C rate with `ac_changed`.
