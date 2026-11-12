"""Central configuration for the Ili Valley SOM mapping project.

Every script imports its paths, seeds, data windows and thresholds from here so
that a single edit propagates through the whole pipeline.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent

# ---------------------------------------------------------------- inputs
DATA_IN = ROOT / "data" / "source"
SOIL_XLSX = DATA_IN / "ili_maize_soil_121.xlsx"
SPEC_XLSX = DATA_IN / "Soil_reflectace.xlsx"
ROI_SHP = DATA_IN / "Yili" / "Yili.shp"

# ---------------------------------------------------------------- project dirs
DATA = ROOT / "data"
RAW = DATA / "raw"
EXT = DATA / "external"
OSSL_DIR = EXT / "ossl"
GEE_DIR = EXT / "gee"
SRF_DIR = EXT / "srf"
BND_DIR = EXT / "boundaries"
TABPFN_CKPT = EXT / "tabpfn" / "tabpfn-v2-regressor.ckpt"

OUT = ROOT / "outputs"
AUDIT = OUT / "audit"
FEAT = OUT / "features"
FOLDS = OUT / "folds"
OOF = OUT / "oof"
MODELS = OUT / "models"
MAPS = OUT / "maps"
FIGS = OUT / "figures"
TABS = OUT / "tables"
LOGS = OUT / "logs"
RES = OUT / "results"
MANUSCRIPT = ROOT / "manuscript"

for _d in (RAW, OSSL_DIR, GEE_DIR, SRF_DIR, BND_DIR, AUDIT, FEAT, FOLDS, OOF,
           MODELS, MAPS, FIGS, TABS, LOGS, RES, MANUSCRIPT):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- reproducibility
SEED = 20261112

# ---------------------------------------------------------------- soil labels
TARGET = "OM"
OM_REPORT_FACTOR = 10.0          # laboratory % (g per 100 g) -> g kg-1 for reporting
OM_REPORT_UNIT = "g kg$^{-1}$"                      # soil organic matter, % (g per 100 g), 0-20 cm composite
AUX_TARGETS = ["pH", "EC"]         # predictability check only
OM_UNIT = "%"
SAMPLING_DEPTH = "0-20 cm"
SAMPLING_DATE = "2026-04-05"

# ---------------------------------------------------------------- Earth Engine
GEE_PROJECT = "hallowed-chain-458005-g9"
CRS_UTM = "EPSG:32644"
CRS_EQAREA = "+proj=aea +lat_1=42 +lat_2=45 +lon_0=82 +datum=WGS84"   # for areas
GRID_SCALE = 20                    # main output grid (m)
SCALES_SENS = [10, 20, 30]
POINT_BUFFER_M = 10                # support radius for point extraction

COVARIATE_END = "2026-04-05"       # exclusive: nothing on or after sampling day
S2_START = "2023-01-01"
BARE_MONTHS = [3, 4, 5, 9, 10, 11]
CS_CDF_MIN = 0.60
NDVI_MAX_MAIN = 0.25
NDVI_MAX_SENS = [0.20, 0.25, 0.30]
MNDWI_MAX = 0.0
NBR2_MAX_SENS = 0.075              # optional residue / moisture filter (sensitivity)
MIN_BARE_OBS = 3
SCL_EXCLUDE = [0, 1, 3, 6, 8, 9, 10, 11]
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

VEG_YEAR = 2025                    # antecedent growing season
VEG_MONTHS = (4, 10)
S1_START = "2023-01-01"
S1_REF_ANGLE = 39.0                # incidence-angle normalisation reference (deg)
CLIM_YEARS = (2016, 2025)
AE_YEAR_MAIN = 2024
AE_YEAR_SENS = 2025
DW_YEAR = 2025

GEE_IDS = {
    "s2": "COPERNICUS/S2_SR_HARMONIZED",
    "cs": "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED",
    "s1": "COPERNICUS/S1_GRD",
    "dem": "COPERNICUS/DEM/GLO30_2024_1",
    "ae": "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
    "dw": "GOOGLE/DYNAMICWORLD/V1",
    "wc": "ESA/WorldCover/v200",
    "era5": "ECMWF/ERA5_LAND/MONTHLY_AGGR",
    "sg_soc": "projects/soilgrids-isric/soc_mean",
    "sg_clay": "projects/soilgrids-isric/clay_mean",
}
GEE_ASSET_ROOT = f"projects/{GEE_PROJECT}/assets/ili_som"

# ---------------------------------------------------------------- spectra
WL_MIN, WL_MAX, WL_STEP = 400, 2450, 5
SG_WINDOW, SG_POLY = 21, 2         # Savitzky-Golay on the 1 nm local spectra
WATER_BANDS = [(1350, 1450), (1800, 1950)]
OSSL_VERSION = "v1.2"
OSSL_FILES = {
    "visnir": "ossl_visnir_L0_v1.2.csv.gz",
    "site": "ossl_soilsite_L0_v1.2.csv.gz",
    "lab": "ossl_soillab_L0_v1.2.csv.gz",
}
OSSL_URL = "https://storage.googleapis.com/soilspec4gg-public/"

# ---------------------------------------------------------------- validation
N_OUTER_GROUPS = 5
BUFFER_MAIN_KM = 5.0
BUFFER_SENS_KM = [0.0, 5.0, 10.0, 20.0]   # 20 km added: 5 km does not bind at group edges
BLOCK_KM = 20.0
N_BOOT = 2000
CONFORMAL_ALPHA = 0.10
LATENT_DIMS = [2, 4, 8]
LEARNING_SIZES = [30, 60, 90]
LEARNING_REPEATS = 10
