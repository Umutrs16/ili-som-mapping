# Ili River Valley cropland soil organic matter

Analysis code and derived results for mapping topsoil organic matter on maize fields in the Ili River Valley, Northwest China. The 121 laboratory spectra and measured soil parameters are available from the corresponding author and are not included here.

The manuscript and the scripts that write it are not in this repository.

## Layout

- `src/` — covariate extraction, spectral processing, spatial cross-validation, model comparison, conformal intervals and mapping
- `data/source/Yili/` — valley boundary
- `data/external/gee/` — point covariates extracted on the fixed 20 m grid
- `data/external/srf/` — Sentinel-2 spectral response functions used to build the simulated-band spectra
- `outputs/results/` — selection logs, out-of-fold predictions, conformal scores and the permutation replicates
- `outputs/models/deployment_linear.json` — mapped ridge coefficients and calibration quantiles
- `outputs/maps/lattice_predictions.csv` — systematic 20 m map nodes

## Files cited by the paper

| File | Contents |
| --- | --- |
| `outputs/results/nested_selection_oof.csv` | Outer-fold predictions, 121 samples, without the measured SOM |
| `outputs/results/nested_selection.json` | Per-fold adoption and pooled metrics |
| `outputs/results/nested_selection_log.csv` | Candidate scores inside each outer training set |
| `outputs/results/fitset_selection.json` | The same rule on the 86-point fitting set |
| `outputs/results/deployment_split.json` | 86 fitting, 28 calibration and 7 buffer samples |
| `outputs/results/permutation_null_rmse.csv` | Additional within-training shuffles |
| `outputs/results/conformal_points.csv` | Sample-level conformal scores |
| `outputs/results/conformal_rotation20.csv` | Two models by 20 calibration assignments, with integer coverage counts |
| `outputs/models/deployment_linear.json` | Mapped coefficients and calibration quantiles |
| `outputs/maps/lattice_predictions.csv` | Map nodes, flags and verification classes |

`permutation_null_rmse.csv` has 398 rows: 199 additional shuffles for the public-latent control E6a-P and 199 for E6b-P. The original shuffled control is the corresponding E6 row of `outputs/results/runs_summary.csv`, not a row of the permutation file. Each 200-value null is that original control plus its 199 additional shuffles. `conformal_rotation20.csv` stores the coverage counts `n_cov_abs` and `n_cov_log`, not only rounded percentages.

Counts in these files: 121 samples; spatial groups from west to east 24, 35, 12, 28 and 22; split 86/28/7; absolute-score coverage 105/121 overall and 9/22 in the eastern group; 427 of 811 initially low-SOM nodes cross 15 g kg−1 when the model is refitted on all samples with the applicability flags held fixed.

The 121 laboratory spectra and measured soil parameters are not in this repository. They are available from the corresponding author. Result tables that originally carried the measured SOM keep the predictions, groups and scores, but not the measured values.

## Not redistributed

Public spectral libraries (OSSL v1.2, including ICRAF–ISRIC and LUCAS) and Sentinel, climate, terrain and embedding covariates remain at the sources named in the paper. The Open Soil Spectral Library file used locally is larger than GitHub's file limit. TabPFN v2 weights are used under the Prior Labs License 1.1 and are not included; the checkpoint SHA-256 is stated in the paper. Repeating the Earth Engine extraction requires access to the project named in `src/config.py`.

## Software

Python 3.13. Package versions are listed in `requirements.txt`.
