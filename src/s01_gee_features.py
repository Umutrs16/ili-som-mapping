"""Step 1 - Earth Engine covariates and point extraction.

Builds every covariate group on a fixed EPSG:32644 grid and extracts point
values for the main configuration and the pre-registered sensitivity variants.
All inputs end before the sampling day (2026-04-05, exclusive).

Groups
  opt  : bare-soil Sentinel-2 composite (10 bands, BSI, NBR2, log observation count)
  veg  : antecedent 2025 growing-season NDVI statistics
  sar  : ascending Sentinel-1 VV/VH (angle-normalised) median, IQR and VV-VH
  ter  : Copernicus DEM GLO-30 2024_1 elevation, slope, northness, eastness, TPI
  clim : ERA5-Land 2016-2025 mean annual temperature and precipitation
  ae   : Satellite Embedding V1 (AlphaEarth Foundations) annual 64-D vector
  sg   : SoilGrids 2.0 SOC and clay, 0-20 cm thickness-weighted (sensitivity only)
  qa   : observation counts, time span, spacecraft counts, masks
"""
from __future__ import annotations

import math
import time

import ee
import numpy as np
import pandas as pd

import gee_common as G
from utils import C, get_logger, load_soil, save_json

log = get_logger("s01_gee")

AE_BANDS = [f"A{i:02d}" for i in range(64)]
AE_NAMES = [f"AE{i:02d}" for i in range(64)]
OPT_FEATURES = C.S2_BANDS + ["BSI", "NBR2", "bare_n_log"]
VEG_FEATURES = ["NDVI_p95", "NDVI_mean", "NDVI_amp"]
SAR_FEATURES = ["VV_med", "VH_med", "VVmVH_med", "VV_iqr", "VH_iqr"]
TER_FEATURES = ["elev", "slope", "northness", "eastness", "TPI"]
CLIM_FEATURES = ["MAT", "MAP"]
SG_FEATURES = ["SG_SOC", "SG_clay"]
QA_BANDS = ["bare_n", "bare_span", "clear_n", "n_S2A", "n_S2B", "n_S2C", "s1_n", "s1_angle"]
MASK_BANDS = ["wc_class", "dw_crop_prob", "dw_label_mode"]


def _months_filter(months):
    flt = [ee.Filter.calendarRange(m, m, "month") for m in months]
    return ee.Filter.Or(*flt)


def _distinct_days(col: ee.ImageCollection, band: str, name: str) -> ee.Image:
    """Number of distinct acquisition days with a valid pixel.

    Overlapping MGRS tiles / GRD slices repeat the same acquisition, so days
    rather than granules are counted. In time order the day values are
    non-decreasing and the number of runs equals the number of distinct days
    (countDistinct is avoided: it counts the masked state as one more value).
    """
    t0 = ee.Date(C.S2_START)
    days = col.sort("system:time_start").map(
        lambda im: im.select([band]).multiply(0).add(im.date().difference(t0, "day").floor())
        .int().rename("d"))
    return days.reduce(ee.Reducer.countRuns()).unmask(0).rename(name)


def s2_clean(region, start, end, months=None) -> ee.ImageCollection:
    col = (ee.ImageCollection(C.GEE_IDS["s2"]).filterBounds(region).filterDate(start, end))
    if months:
        col = col.filter(_months_filter(months))
    col = col.linkCollection(ee.ImageCollection(C.GEE_IDS["cs"]), ["cs_cdf"])

    def prep(im):
        scl = im.select("SCL")
        ok = im.select("cs_cdf").gte(C.CS_CDF_MIN)
        for v in C.SCL_EXCLUDE:
            ok = ok.And(scl.neq(v))
        refl = im.select(C.S2_BANDS).multiply(0.0001)
        return refl.updateMask(ok).copyProperties(im, ["system:time_start", "SPACECRAFT_NAME"])

    return col.map(prep)


def bare_mask(im, ndvi_max, nbr2_max=None):
    ndvi = im.normalizedDifference(["B8", "B4"])
    mndwi = im.normalizedDifference(["B3", "B11"])
    m = ndvi.gte(0).And(ndvi.lt(ndvi_max)).And(mndwi.lt(C.MNDWI_MAX))
    if nbr2_max is not None:
        m = m.And(im.normalizedDifference(["B11", "B12"]).lt(nbr2_max))
    return im.updateMask(m)


def optical_group(region, ndvi_max=C.NDVI_MAX_MAIN, method="median", nbr2_max=None, full_qa=True):
    clean = s2_clean(region, C.S2_START, C.COVARIATE_END, C.BARE_MONTHS)
    bare = clean.map(lambda im: bare_mask(im, ndvi_max, nbr2_max))
    n = _distinct_days(bare, "B11", "bare_n")
    if full_qa:
        clear_n = _distinct_days(clean, "B11", "clear_n")
        t0 = ee.Date(C.S2_START)
        tt = bare.map(lambda im: ee.Image.constant(im.date().difference(t0, "day")).float()
                      .updateMask(im.select("B11").mask()).rename("t"))
        span = tt.max().subtract(tt.min()).unmask(0).rename("bare_span")
        sc = [_distinct_days(bare.filter(ee.Filter.eq("SPACECRAFT_NAME", f"Sentinel-2{s}")),
                             "B11", f"n_S2{s}") for s in ("A", "B", "C")]

    if method == "median":
        comp = bare.median()
    elif method == "p90":
        comp = bare.reduce(ee.Reducer.percentile([90])).rename(C.S2_BANDS)
    elif method == "medoid":
        med = bare.median()
        comp = bare.map(lambda im: im.addBands(
            im.subtract(med).pow(2).reduce(ee.Reducer.sum()).multiply(-1).rename("negd"))) \
            .qualityMosaic("negd").select(C.S2_BANDS)
    else:
        raise ValueError(method)
    comp = comp.updateMask(n.gte(C.MIN_BARE_OBS))
    bsi = comp.expression("((b11 + b4) - (b8 + b2)) / ((b11 + b4) + (b8 + b2))",
                          {"b11": comp.select("B11"), "b4": comp.select("B4"),
                           "b8": comp.select("B8"), "b2": comp.select("B2")}).rename("BSI")
    nbr2 = comp.normalizedDifference(["B11", "B12"]).rename("NBR2")
    nlog = n.add(1).log().rename("bare_n_log")
    feats = comp.select(C.S2_BANDS).addBands([bsi, nbr2, nlog])
    qa = ee.Image.cat([n, span, clear_n] + sc) if full_qa else n
    return feats, qa


def veg_group(region):
    y = C.VEG_YEAR
    col = s2_clean(region, f"{y}-{C.VEG_MONTHS[0]:02d}-01", f"{y}-{C.VEG_MONTHS[1] + 1:02d}-01")
    ndvi = col.map(lambda im: im.normalizedDifference(["B8", "B4"]).rename("NDVI"))
    p = ndvi.reduce(ee.Reducer.percentile([5, 95]))
    mean = ndvi.mean().rename("NDVI_mean")
    p95 = p.select("NDVI_p95")
    amp = p95.subtract(p.select("NDVI_p5")).rename("NDVI_amp")
    return ee.Image.cat([p95.rename("NDVI_p95"), mean, amp])


def sar_group(region):
    col = (ee.ImageCollection(C.GEE_IDS["s1"]).filterBounds(region)
           .filterDate(C.S1_START, C.COVARIATE_END)
           .filter(_months_filter(C.BARE_MONTHS))
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    ref = 20 * math.log10(math.cos(math.radians(C.S1_REF_ANGLE)))

    def norm(im):
        ang = im.select("angle")
        corr = ang.multiply(math.pi / 180).cos().log10().multiply(-20).add(ref)
        vv = im.select("VV").add(corr).rename("VV")
        vh = im.select("VH").add(corr).rename("VH")
        return vv.addBands(vh).addBands(ang.rename("angle")) \
            .copyProperties(im, ["system:time_start", "relativeOrbitNumber_start"])

    col = col.map(norm)
    pr = col.select(["VV", "VH"]).reduce(ee.Reducer.percentile([25, 50, 75]))
    vv_med = pr.select("VV_p50").rename("VV_med")
    vh_med = pr.select("VH_p50").rename("VH_med")
    feats = ee.Image.cat([
        vv_med, vh_med, vv_med.subtract(vh_med).rename("VVmVH_med"),
        pr.select("VV_p75").subtract(pr.select("VV_p25")).rename("VV_iqr"),
        pr.select("VH_p75").subtract(pr.select("VH_p25")).rename("VH_iqr"),
    ])
    qa = ee.Image.cat([_distinct_days(col, "VV", "s1_n"),
                       col.select("angle").mean().rename("s1_angle")])
    return feats, qa


def terrain_group(region):
    dc = ee.ImageCollection(C.GEE_IDS["dem"]).filterBounds(region).select("DEM")
    dem = dc.mosaic().setDefaultProjection(dc.first().projection())
    slope = ee.Terrain.slope(dem).rename("slope")
    asp = ee.Terrain.aspect(dem).multiply(math.pi / 180)
    tpi = dem.subtract(dem.focalMean(500, "circle", "meters")).rename("TPI")
    img = ee.Image.cat([dem.rename("elev"), slope, asp.cos().rename("northness"),
                        asp.sin().rename("eastness"), tpi])
    return img.resample("bilinear")


def climate_group():
    y0, y1 = C.CLIM_YEARS
    era = ee.ImageCollection(C.GEE_IDS["era5"]).filterDate(f"{y0}-01-01", f"{y1 + 1}-01-01")
    n_years = y1 - y0 + 1
    native = era.first().projection()
    mat = era.select("temperature_2m").mean().subtract(273.15).rename("MAT")
    pmm = era.select("total_precipitation_sum").sum().multiply(1000 / n_years).rename("MAP")
    return ee.Image.cat([mat, pmm]).setDefaultProjection(native).resample("bilinear")


def ae_group(region, year):
    col = (ee.ImageCollection(C.GEE_IDS["ae"]).filterBounds(region)
           .filterDate(f"{year}-01-01", f"{year + 1}-01-01"))
    return col.mosaic().select(AE_BANDS, AE_NAMES)


def soilgrids_group():
    soc = ee.Image(C.GEE_IDS["sg_soc"])
    clay = ee.Image(C.GEE_IDS["sg_clay"])

    def w(img, prefix):
        a, b, c = (img.select(f"{prefix}_{d}_mean") for d in ("0-5cm", "5-15cm", "15-30cm"))
        return a.multiply(5).add(b.multiply(10)).add(c.multiply(5)).divide(20)

    # SOC dg/kg -> g/kg ; clay g/kg*10 -> %
    return ee.Image.cat([w(soc, "soc").divide(10).rename("SG_SOC"),
                         w(clay, "clay").divide(10).rename("SG_clay")]).resample("bilinear")


def mask_group(region):
    wc = ee.ImageCollection(C.GEE_IDS["wc"]).first().select("Map").rename("wc_class")
    dw = (ee.ImageCollection(C.GEE_IDS["dw"]).filterBounds(region)
          .filterDate(f"{C.DW_YEAR}-04-01", f"{C.DW_YEAR}-11-01"))
    prob = dw.select("crops").mean().rename("dw_crop_prob")
    mode = dw.select("label").mode().rename("dw_label_mode")
    return ee.Image.cat([wc, prob, mode])


def to_grid(img: ee.Image, scale: int, native10: bool) -> ee.Image:
    """Put an image on the fixed grid; 10 m sources are mean-aggregated."""
    tr = G.grid_transform(scale)
    if native10 and scale > 10:
        img = img.setDefaultProjection(crs=C.CRS_UTM, scale=10) \
            .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=64)
    return img.reproject(crs=C.CRS_UTM, crsTransform=tr)


def build_stack(region, scale=C.GRID_SCALE, ndvi_max=C.NDVI_MAX_MAIN, method="median",
                nbr2_max=None, ae_year=C.AE_YEAR_MAIN, groups=("opt", "veg", "sar", "ter",
                                                                 "clim", "ae", "sg", "qa",
                                                                 "mask"),
                opt_native10=False):
    """`qa_bare` (instead of `qa`) adds only the bare-soil day count, which is all the map needs.

    The Sentinel-2 composite is computed on the output grid, 10 m bands entering through
    Earth Engine's mean pyramid; opt_native10=True composites at 10 m and averages to the
    grid instead (30 m variant, where the pyramid would be resampled by nearest neighbour).
    """
    parts = []
    if "opt" in groups or "qa" in groups or "qa_bare" in groups:
        opt, qa_opt = optical_group(region, ndvi_max, method, nbr2_max, full_qa="qa" in groups)
        if "opt" in groups:
            parts.append(to_grid(opt, scale, opt_native10))
        if "qa_bare" in groups:
            parts.append(qa_opt.reproject(crs=C.CRS_UTM, crsTransform=G.grid_transform(scale)))
    if "veg" in groups:
        parts.append(to_grid(veg_group(region), scale, True))
    if "sar" in groups or "qa" in groups:
        sar, qa_sar = sar_group(region)
        if "sar" in groups:
            parts.append(to_grid(sar, scale, True))
    if "ter" in groups:
        parts.append(terrain_group(region).reproject(crs=C.CRS_UTM,
                                                     crsTransform=G.grid_transform(scale)))
    if "clim" in groups:
        parts.append(climate_group().reproject(crs=C.CRS_UTM,
                                               crsTransform=G.grid_transform(scale)))
    if "ae" in groups:
        parts.append(to_grid(ae_group(region, ae_year), scale, True))
    if "sg" in groups:
        parts.append(soilgrids_group().reproject(crs=C.CRS_UTM,
                                                 crsTransform=G.grid_transform(scale)))
    if "qa" in groups:
        qa = ee.Image.cat([qa_opt, qa_sar])
        parts.append(qa.reproject(crs=C.CRS_UTM, crsTransform=G.grid_transform(scale)))
    if "mask" in groups:
        parts.append(mask_group(region).reproject(crs=C.CRS_UTM,
                                                  crsTransform=G.grid_transform(scale)))
    return ee.Image.cat(parts)


def spatial_batches(n_batches: int = 16) -> list[pd.DataFrame]:
    """Spatially compact point batches (k-means on coordinates)."""
    from sklearn.cluster import KMeans
    soil = load_soil()
    lab = KMeans(n_batches, n_init=10, random_state=C.SEED).fit_predict(soil[["lon", "lat"]])
    return [soil[lab == k] for k in range(n_batches)]


def _extract_batch(sub: pd.DataFrame, build_kw: dict, scale: int, support: str,
                   retries: int = 5) -> list[dict]:
    lon0, lon1 = sub.lon.min() - 0.02, sub.lon.max() + 0.02
    lat0, lat1 = sub.lat.min() - 0.02, sub.lat.max() + 0.02
    region = ee.Geometry.Rectangle([lon0, lat0, lon1, lat1])
    stack = build_stack(region, scale=scale, **build_kw)
    feats = []
    for sid, r in sub.iterrows():
        g = ee.Geometry.Point([float(r.lon), float(r.lat)])
        if support == "buffer":
            g = g.buffer(C.POINT_BUFFER_M)
        feats.append(ee.Feature(g, {"sample_id": sid}))
    reducer = ee.Reducer.mean() if support == "buffer" else ee.Reducer.first()
    out = stack.reduceRegions(collection=ee.FeatureCollection(feats), reducer=reducer,
                              crs=C.CRS_UTM, crsTransform=G.grid_transform(scale), tileScale=4)
    for k in range(retries):
        try:
            return [f["properties"] for f in out.getInfo()["features"]]
        except Exception as e:  # transient EE errors / rate limits
            log.warning("batch (%d pts) attempt %d failed: %s", len(sub), k + 1, str(e)[:160])
            time.sleep(15 * (k + 1))
    raise RuntimeError("EE extraction failed repeatedly")


def extract(build_kw: dict, scale: int, support: str, workers: int = 6) -> pd.DataFrame:
    """Point values: support='buffer' (10 m radius mean) or 'center' (pixel)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    batches = spatial_batches()
    rows, done = [], 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_extract_batch, b, build_kw, scale, support) for b in batches]
        for f in as_completed(futs):
            rows += f.result()
            done += 1
            log.info("  batches %d/%d", done, len(batches))
    return pd.DataFrame(rows).set_index("sample_id").sort_index()


def main() -> None:
    G.init()
    meta = {"variants": {}}

    variants = {
        "main_s20_buffer": dict(scale=20, support="buffer"),
        "main_s20_center": dict(scale=20, support="center"),
        "s10_buffer": dict(scale=10, support="buffer",
                           groups=("opt", "veg", "sar", "ter", "clim", "ae")),
        "s30_buffer": dict(scale=30, support="buffer", opt_native10=True,
                           groups=("opt", "veg", "sar", "ter", "clim", "ae")),
        "ndvi020": dict(scale=20, support="buffer", ndvi_max=0.20, groups=("opt", "qa")),
        "ndvi030": dict(scale=20, support="buffer", ndvi_max=0.30, groups=("opt", "qa")),
        "nbr2": dict(scale=20, support="buffer", nbr2_max=C.NBR2_MAX_SENS, groups=("opt", "qa")),
        "medoid": dict(scale=20, support="buffer", method="medoid", groups=("opt",)),
        "p90": dict(scale=20, support="buffer", method="p90", groups=("opt",)),
        "c10": dict(scale=20, support="buffer", opt_native10=True, groups=("opt",)),
        "ae2025": dict(scale=20, support="buffer", ae_year=C.AE_YEAR_SENS, groups=("ae",)),
    }
    for name, v in variants.items():
        path = C.GEE_DIR / f"points_{name}.csv"
        if path.exists():
            log.info("skip %s (exists)", name)
            continue
        kw = {k: v[k] for k in ("ndvi_max", "method", "nbr2_max", "ae_year", "groups",
                                "opt_native10") if k in v}
        log.info("variant %s: scale=%s %s", name, v["scale"], kw)
        t0 = time.time()
        df = extract(kw, v["scale"], v["support"])
        df.to_csv(path)
        meta["variants"][name] = {**kw, "scale": v["scale"], "support": v["support"],
                                  "n_rows": len(df), "n_cols": df.shape[1],
                                  "seconds": round(time.time() - t0, 1)}
        log.info("variant %s -> %s (%d x %d) in %.0f s", name, path.name, *df.shape,
                 time.time() - t0)
    meta["feature_groups"] = {"opt": OPT_FEATURES, "veg": VEG_FEATURES, "sar": SAR_FEATURES,
                              "ter": TER_FEATURES, "clim": CLIM_FEATURES, "ae": AE_NAMES,
                              "sg": SG_FEATURES, "qa": QA_BANDS, "mask": MASK_BANDS}
    meta["gee_ids"] = C.GEE_IDS
    meta["windows"] = {"s2": [C.S2_START, C.COVARIATE_END], "bare_months": C.BARE_MONTHS,
                       "veg_year": C.VEG_YEAR, "s1": [C.S1_START, C.COVARIATE_END],
                       "climate_years": C.CLIM_YEARS, "ae_main": C.AE_YEAR_MAIN}
    meta["thresholds"] = {"cs_cdf_min": C.CS_CDF_MIN, "ndvi_max": C.NDVI_MAX_MAIN,
                          "mndwi_max": C.MNDWI_MAX, "min_bare_obs": C.MIN_BARE_OBS,
                          "scl_exclude": C.SCL_EXCLUDE, "s1_ref_angle": C.S1_REF_ANGLE}
    old = C.GEE_DIR / "feature_manifest.json"
    if old.exists():
        from utils import load_json
        prev = load_json(old)
        prev.get("variants", {}).update(meta["variants"])
        meta["variants"] = prev.get("variants", meta["variants"])
    save_json(meta, old)


if __name__ == "__main__":
    main()
