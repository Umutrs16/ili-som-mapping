"""Deployment mapping.

1. fit     : tune and fit the deployed configuration on the pre-locked fit set,
             calibrate split-conformal q on the calibration blocks, fit the
             AOA on the fit set; also a 121-point fit (point-prediction
             sensitivity only, no conformal interval).
2. lattice : the map is a systematic sample of the 20 m grid, because computing
             the deployed covariates wall-to-wall exceeds the Earth Engine
             budget. Nodes are 20 m pixel centres on a LATTICE_M lattice with a
             random start; the cropland domain is read at every node and the
             deployed covariates (pixel values, as a wall-to-wall map would hold
             them) at domain nodes. Predictions, intervals, DI/LPD and flags are
             computed locally with the fitted pipeline.
3. window  : a WINDOW_M square at the densest sampling is mapped wall-to-wall at
             20 m. The deployed pipelines are linear in the covariates (imputer
             -> scaler -> ridge/PLS), so the exact linear predictor and the
             DI/LPD computation are ported to Earth Engine; the lattice nodes in
             the window check the port.
4. summary : design-based areas and shares (binomial intervals) and quantiles.

    python s08_mapping.py [fit|lattice|window|summary]
"""
from __future__ import annotations

import pickle
import sys
import time

import numpy as np
import pandas as pd

import engine as En
import models as M
from s06_uncertainty import conformal_q, experiment, fit_aoa
from utils import C, get_logger, load_json, save_json

log = get_logger("s08_map")
DEP_PKL = C.MODELS / "deployment.pkl"
DEP_JSON = C.MODELS / "deployment_linear.json"
BANDS = ["OM", "lo_abs", "hi_abs", "lo_log", "hi_log", "DI", "LPD", "bare_n",
         "flags", "domain", "OM_all"]
SCALE_OF = {"OM": 100, "lo_abs": 100, "hi_abs": 100, "lo_log": 100, "hi_log": 100,
            "DI": 1000, "LPD": 1, "bare_n": 1, "flags": 1, "domain": 1,
            "OM_all": 100}
LPD_LOW = 2
LATTICE_M = 2000
BLOCK_M = 30_000
WINDOW_M = 10_000
CHUNK_PX = 125
NODES = C.MAPS / "lattice_nodes.csv"
FEATS = C.MAPS / "lattice_features.csv"
PRED = C.MAPS / "lattice_predictions.csv"
PARTS = C.MAPS / "lattice_parts"
WIN_TIF = C.MAPS / "window_20m.tif"
LOW_SOM_PCT = 1.5      # 15 g kg-1; relative stratification, see manuscript Section 2.6
PRIORITY = {1: "A: relatively low SOM, supported", 2: "B: verification priority",
            3: "C: other supported cropland"}


# ------------------------------------------------------------------ 1. fit
def linear_predictor(fitted):
    """Return f(X) on the model's internal scale (log or original), before back-transform."""
    m = fitted.model
    if isinstance(m, M.TransferModel):
        return lambda X: np.asarray(m.H_.e_.predict(np.c_[X, m.G_.predict(X)])).ravel()
    return lambda X: np.asarray(m.e_.predict(X)).ravel()


def linearize(fitted, X_ref):
    f = linear_predictor(fitted)
    p = X_ref.shape[1]
    a0 = float(f(np.zeros((1, p)))[0])
    coef = f(np.eye(p)) - a0
    err = float(np.max(np.abs(f(X_ref) - (a0 + X_ref @ coef))))
    m = fitted.model
    lt = m.H_ if isinstance(m, M.TransferModel) else m
    return {"a0": a0, "coef": coef.tolist(), "log": bool(lt.log), "smear": float(lt.smear_),
            "linearity_max_abs_err": err}


def map_choice():
    """Map the model returned by nested selection when every outer fold agrees.

    The engineering-sequence record in deployment_decision.json is left unchanged.
    """
    path = C.RES / "nested_selection.json"
    if path.exists():
        ns = load_json(path)
        pairs = [(r["adopted"], r["student"]) for r in ns["folds"]]
        if pairs and pairs.count(pairs[0]) == len(pairs):
            return pairs[0][0], pairs[0][1], "nested_selection"
    dec = load_json(C.RES / "deployment_decision.json")
    return dec["deployed_experiment"], dec["deployed_student"], "engineering_sequence"


def fit():
    name, student, source = map_choice()
    exp = experiment(name, student)
    D = En.get_data()
    split = pd.read_csv(C.FOLDS / "deployment_split.csv")
    fit_idx = np.where(split.role.to_numpy() == "fit")[0]
    cal_idx = np.where(split.role.to_numpy() == "cal")[0]
    feats = M.features_for(exp.groups)
    X = D.X[feats].to_numpy(float)

    theta = En.select_params(exp, [fit_idx])[0]
    fitted = En.fit_model(exp, theta, fit_idx)
    pc = fitted.predict(X[cal_idx])
    yc = D.y[cal_idx]
    q_abs = conformal_q(np.abs(yc - pc))
    q_log = conformal_q(np.abs(np.log(yc) - np.log(np.clip(pc, 1e-3, None))))
    aoa, w = fit_aoa(fitted, X[fit_idx], D.y[fit_idx], D.xy[fit_idx])

    all_idx = np.arange(len(D.y))
    theta_all = En.select_params(exp, [all_idx])[0]
    fitted_all = En.fit_model(exp, theta_all, all_idx)

    with open(DEP_PKL, "wb") as f:
        pickle.dump({"exp": exp, "theta": theta, "fitted": fitted, "aoa": aoa, "weights": w,
                     "q_abs": q_abs, "q_log": q_log, "fit_idx": fit_idx, "cal_idx": cal_idx,
                     "theta_all": theta_all, "fitted_all": fitted_all}, f)

    linear = exp.student in ("ridge", "pls")
    out = {"experiment": exp.name, "student": exp.student, "source": source, "features": feats,
           "theta": theta, "theta_all": theta_all, "q_abs": q_abs, "q_log": q_log,
           "n_fit": len(fit_idx), "n_cal": len(cal_idx), "alpha": C.CONFORMAL_ALPHA,
           "cal_rmse": En.rmse(yc, pc), "linear": linear,
           "medians": np.nanmedian(X[fit_idx], axis=0).tolist(),
           "medians_all": np.nanmedian(X, axis=0).tolist(),
           "bare_n_min_train": float(D.X["bare_n"].iloc[fit_idx].min()),
           "aoa": {"m": aoa.m.tolist(), "s": aoa.s.tolist(), "w": aoa.w.tolist(),
                   "dbar": aoa.dbar, "thr": aoa.thr, "T": aoa.T.tolist(),
                   "di_train_cv": aoa.di_train.tolist()},
           "fit_ids": D.ids[fit_idx].tolist()}
    if linear:
        out["model"] = linearize(fitted, X[fit_idx])
        out["model_all"] = linearize(fitted_all, X)
    save_json(out, DEP_JSON)
    log.info(f"map model {exp.name}/{exp.student} ({source}): theta={theta} q_abs={q_abs:.3f} "
             f"q_log={q_log:.3f} cal_RMSE={out['cal_rmse']:.3f} AOA thr={aoa.thr:.3f} "
             f"linear={linear} lin_err={out.get('model', {}).get('linearity_max_abs_err')}")


# ------------------------------------------------------------------ Earth Engine pieces
def _apply_linear(ee, img, feats, medians, lin):
    x = img.select(feats).unmask(ee.Image.constant(medians).rename(feats))
    eta = x.multiply(ee.Image.constant(lin["coef"]).rename(feats)) \
        .reduce(ee.Reducer.sum()).add(lin["a0"])
    om = eta.exp().multiply(lin["smear"]) if lin["log"] else eta.max(0)
    return om.rename("OM"), x


def cropland_domain(roi):
    """1 = WorldCover cropland, 2 = Dynamic World crops mode, 3 = both (>= half of a 20 m cell)."""
    import ee

    from s01_gee_features import to_grid
    wc = ee.ImageCollection(C.GEE_IDS["wc"]).first().select("Map").eq(40)
    dw = (ee.ImageCollection(C.GEE_IDS["dw"]).filterBounds(roi)
          .filterDate(f"{C.DW_YEAR}-04-01", f"{C.DW_YEAR}-11-01").select("label").mode().eq(4))
    fr = to_grid(ee.Image.cat([wc.rename("wc"), dw.rename("dw")]).float(), C.GRID_SCALE, True)
    return fr.select("wc").gte(0.5).add(fr.select("dw").gte(0.5).multiply(2)).rename("domain")


def stack_groups(feats):
    """build_stack groups holding the deployed features, plus the bare-soil day count."""
    groups = {"O": "opt", "T": "ter", "R": "sar", "V": "veg", "C": "clim", "A": "ae"}
    return tuple(sorted({groups[g] for g in M.GROUPS if g in groups and
                         any(f in M.GROUPS[g] for f in feats)} | {"opt", "qa_bare"}))


def map_image():
    """Deployed prediction, intervals, DI, LPD and flags as int16 bands on the 20 m grid."""
    import ee

    import gee_common as G
    from s01_gee_features import build_stack
    dep = load_json(DEP_JSON)
    if not dep["linear"]:
        raise SystemExit("deployed learner is not linear; the window needs a linear port")
    feats = dep["features"]
    roi = G.roi_geometry(simplify_m=100)
    tr = G.grid_transform(C.GRID_SCALE)
    domain = cropland_domain(roi)
    stack = build_stack(roi, scale=C.GRID_SCALE, groups=stack_groups(feats))

    om, x = _apply_linear(ee, stack, feats, dep["medians"], dep["model"])
    om_all, _ = _apply_linear(ee, stack, feats, dep["medians_all"], dep["model_all"])
    qa, ql = dep["q_abs"], dep["q_log"]
    lo_abs, hi_abs = om.subtract(qa).max(0).rename("lo_abs"), om.add(qa).rename("hi_abs")
    lo_log, hi_log = om.multiply(np.exp(-ql)).rename("lo_log"), om.multiply(np.exp(ql)).rename("hi_log")

    a = dep["aoa"]
    w = np.array(a["w"])
    keep = np.where(w > 0)[0]
    kf = [feats[i] for i in keep]
    t = x.select(kf).subtract(ee.Image.constant([a["m"][i] for i in keep]).rename(kf)) \
        .divide(ee.Image.constant([a["s"][i] for i in keep]).rename(kf)) \
        .multiply(ee.Image.constant([float(w[i]) for i in keep]).rename(kf))
    T = np.array(a["T"])[:, keep]
    tcol = t.toArray().toArray(1)
    dots = ee.Image(ee.Array(T.tolist())).matrixMultiply(tcol).arrayProject([0])
    tt = t.pow(2).reduce(ee.Reducer.sum())
    d2 = ee.Image(ee.Array((T ** 2).sum(1).tolist())).subtract(dots.multiply(2)).add(tt).max(0)
    di = d2.arrayReduce(ee.Reducer.min(), [0]).arrayGet([0]).sqrt().divide(a["dbar"]).rename("DI")
    lpd = d2.lte((a["thr"] * a["dbar"]) ** 2).arrayReduce(ee.Reducer.sum(), [0]) \
        .arrayGet([0]).rename("LPD")

    bare_n = stack.select("bare_n").rename("bare_n")
    flags = di.gt(a["thr"]).add(bare_n.lt(dep["bare_n_min_train"]).multiply(2)) \
        .add(lpd.lte(LPD_LOW).And(di.lte(a["thr"])).multiply(4)).rename("flags")

    valid = bare_n.gte(C.MIN_BARE_OBS).And(domain.gt(0))
    out = ee.Image.cat([om, lo_abs, hi_abs, lo_log, hi_log, di, lpd, bare_n, flags,
                        domain, om_all.rename("OM_all")])
    out = ee.Image.cat([out.select(b).multiply(SCALE_OF[b]).round().int16() for b in BANDS])
    return out.updateMask(valid).clip(roi).reproject(crs=C.CRS_UTM, crsTransform=tr), roi, tr


def _eecu(ids):
    """Total EECU-seconds of the profiled requests (None if the profile is unavailable)."""
    import ee
    if not ids:
        return None
    for _ in range(10):
        try:
            prof = ee.ApiFunction.lookup("Profile.getProfiles").call(ids=ids).getInfo()
            break
        except ee.EEException:
            time.sleep(5)
    else:
        return None
    first = [ln.split()[0] for ln in prof.splitlines()[1:] if ln.split()]
    return sum(float(v) for v in first if v.replace(".", "", 1).isdigit())


def _profiled(fn, what, retries=5):
    import ee
    for k in range(retries):
        ids = []
        try:
            with ee.data.profiling(ids.append):
                res = fn()
            return res, _eecu(ids)
        except Exception as e:  # transient EE errors / rate limits
            log.warning(f"{what} attempt {k + 1}: {str(e)[:160]}")
            time.sleep(15 * (k + 1))
    raise RuntimeError(f"{what} failed repeatedly")


# ------------------------------------------------------------------ 2. lattice
def lattice_nodes():
    """20 m pixel centres on a LATTICE_M lattice with a random start, inside the ROI."""
    import geopandas as gpd
    rng = np.random.default_rng(C.SEED)
    g = C.GRID_SCALE
    off = rng.integers(0, LATTICE_M // g, size=2) * g + g / 2
    roi = gpd.read_file(C.ROI_SHP).to_crs(C.CRS_UTM).geometry.union_all()
    x0, y0, x1, y1 = roi.bounds
    xs = np.arange(np.floor(x0 / LATTICE_M) * LATTICE_M + off[0], x1, LATTICE_M)
    ys = np.arange(np.floor(y0 / LATTICE_M) * LATTICE_M + off[1], y1, LATTICE_M)
    X, Y = (a.ravel() for a in np.meshgrid(xs, ys))
    inside = gpd.GeoSeries(gpd.points_from_xy(X, Y), crs=C.CRS_UTM).within(roi).to_numpy()
    df = pd.DataFrame({"x_utm": X[inside], "y_utm": Y[inside]})
    df.index.name = "node"
    df["block"] = [f"{int(x // BLOCK_M)}_{int(y // BLOCK_M)}" for x, y in zip(df.x_utm, df.y_utm)]
    return df, {"lattice_m": LATTICE_M, "offset_m": off.tolist(), "roi_km2": roi_km2(),
                "n_nodes": len(df)}


def roi_km2():
    import geopandas as gpd
    return float(gpd.read_file(C.ROI_SHP).to_crs(C.CRS_EQAREA).area.sum() / 1e6)


def _batches(df, max_pts):
    out = []
    for b, sub in df.groupby("block"):
        sub = sub.sort_values(["y_utm", "x_utm"])
        for k in range(0, len(sub), max_pts):
            out.append((f"{b}_{k // max_pts}", sub.iloc[k:k + max_pts]))
    return out


def _at_nodes(kind, name, sub, make_image, support="center"):
    """Extract at lattice nodes; support='buffer' matches the 10 m training radius."""
    import ee

    import gee_common as G
    path = PARTS / kind / f"{name}.json"
    if path.exists():
        return load_json(path)
    pad = 1000
    region = ee.Geometry.Rectangle([sub.x_utm.min() - pad, sub.y_utm.min() - pad,
                                    sub.x_utm.max() + pad, sub.y_utm.max() + pad],
                                   proj=C.CRS_UTM, geodesic=False)
    feats = []
    for n, x, y in zip(sub.index, sub.x_utm, sub.y_utm):
        g = ee.Geometry.Point([float(x), float(y)], C.CRS_UTM)
        if support == "buffer":
            g = g.buffer(C.POINT_BUFFER_M)
        feats.append(ee.Feature(g, {"node": int(n)}))
    fc = ee.FeatureCollection(feats)
    reducer = ee.Reducer.mean() if support == "buffer" else ee.Reducer.first()
    red = make_image(region).reduceRegions(collection=fc, reducer=reducer,
                                           crs=C.CRS_UTM,
                                           crsTransform=G.grid_transform(C.GRID_SCALE),
                                           tileScale=4)
    rows, eecu = _profiled(lambda: [f["properties"] for f in red.getInfo()["features"]],
                           f"{kind} {name} ({len(sub)} nodes)")
    res = {"rows": rows, "eecu_s": eecu}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_json(res, path)
    return res


def _run_batches(kind, df, max_pts, make_image, workers=8, support="center"):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    jobs = _batches(df, max_pts)
    rows, eecu, done = [], [], 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(_at_nodes, kind, nm, sub, make_image, support) for nm, sub in jobs]
        for f in as_completed(futs):
            r = f.result()
            rows += r["rows"]
            eecu.append(r["eecu_s"])
            done += 1
            if done % 10 == 0 or done == len(jobs):
                tot = sum(e for e in eecu if e is not None)
                log.info(f"  {kind}: {done}/{len(jobs)} batches, {tot / 3600:.2f} EECU-h")
    known = [e for e in eecu if e is not None]
    return pd.DataFrame(rows).set_index("node").sort_index(), \
        {"batches": len(jobs), "eecu_h": sum(known) / 3600, "batches_profiled": len(known)}


def lattice():
    import gee_common as G
    from s01_gee_features import build_stack
    G.init()
    dep = load_json(DEP_JSON)
    nodes, meta = lattice_nodes()
    log.info(f"{meta['n_nodes']} lattice nodes ({LATTICE_M / 1000:g} km, offset "
             f"{meta['offset_m']} m) in {meta['roi_km2']:,.0f} km2")

    dom, meta["cost_domain"] = _run_batches("domain", nodes, 250, cropland_domain)
    col = "domain" if "domain" in dom else "first"
    nodes["domain"] = dom[col].reindex(nodes.index).fillna(0).astype(int)
    nodes.to_csv(NODES)
    sel = nodes[nodes.domain > 0]
    log.info(f"{len(sel)} nodes in the cropland domain")

    groups = stack_groups(dep["features"])
    # Match the training point support (10 m radius mean on the 20 m grid).
    X, meta["cost_features"] = _run_batches(
        "features_buf10", sel, 40,
        lambda region: build_stack(region, scale=C.GRID_SCALE, groups=groups),
        support="buffer")
    X = X.reindex(sel.index)
    X.to_csv(FEATS)
    meta["groups"] = list(groups)
    meta["feature_support"] = {"type": "buffer", "radius_m": C.POINT_BUFFER_M}
    save_json(meta, C.MAPS / "lattice_meta.json")
    predict_nodes()


def predict_nodes():
    dep = load_json(DEP_JSON)
    with open(DEP_PKL, "rb") as f:
        pk = pickle.load(f)
    nodes = pd.read_csv(NODES, index_col="node")
    X = pd.read_csv(FEATS, index_col="node")
    A = X[dep["features"]].to_numpy(float)
    om, om_all = pk["fitted"].predict(A), pk["fitted_all"].predict(A)
    ev = pk["aoa"].evaluate(A)
    thr = pk["aoa"].thr
    bare = X["bare_n"].fillna(0).to_numpy(float)
    flags = (ev["DI"] > thr).astype(int) + 2 * (bare < dep["bare_n_min_train"]) \
        + 4 * ((ev["LPD"] <= LPD_LOW) & (ev["DI"] <= thr))
    P = nodes.loc[X.index, ["x_utm", "y_utm", "domain"]].copy()
    P["bare_n"] = bare
    P["OM"], P["OM_all"] = om, om_all
    P["lo_abs"], P["hi_abs"] = np.clip(om - pk["q_abs"], 0, None), om + pk["q_abs"]
    P["lo_log"], P["hi_log"] = om * np.exp(-pk["q_log"]), om * np.exp(pk["q_log"])
    P["DI"], P["LPD"], P["flags"] = ev["DI"], ev["LPD"], flags
    P["valid"] = bare >= C.MIN_BARE_OBS
    P["cls"] = np.where(flags > 0, 2, np.where(om < LOW_SOM_PCT, 1, 3))
    P.loc[~P.valid, "cls"] = 0
    P.to_csv(PRED)
    log.info(f"{int(P.valid.sum())} mapped nodes; flags {np.bincount(flags[P.valid], minlength=8)}")


# ------------------------------------------------------------------ 3. window
def window_bounds():
    """WINDOW_M square, on the 20 m grid, centred on the densest cluster of samples."""
    soil = pd.read_csv(C.RAW / "soil_audited.csv")
    xy = soil[["x_utm", "y_utm"]].to_numpy()
    half = WINDOW_M / 2
    cnt = [(np.abs(xy - p).max(axis=1) <= half).sum() for p in xy]
    c = xy[np.abs(xy - xy[int(np.argmax(cnt))]).max(axis=1) <= half].mean(axis=0)
    g = C.GRID_SCALE
    x0, y1 = np.round((c[0] - half) / g) * g, np.round((c[1] + half) / g) * g
    return float(x0), float(y1 - WINDOW_M), float(x0 + WINDOW_M), float(y1)


def window(workers=4):
    import ee
    import rasterio
    from concurrent.futures import ThreadPoolExecutor
    from rasterio.transform import from_origin
    from rasterio.windows import Window

    import gee_common as G
    G.init()
    out, _, tr = map_image()
    img = out.unmask(-32768, sameFootprint=False)
    x0, y0, x1, y1 = window_bounds()
    n = int(WINDOW_M // C.GRID_SCALE)
    jobs = [(r, c) for r in range(0, n, CHUNK_PX) for c in range(0, n, CHUNK_PX)]
    dep = load_json(DEP_JSON)
    cache = PARTS / f"window_{dep['experiment']}_{x0:.0f}_{y1:.0f}"
    cache.mkdir(parents=True, exist_ok=True)

    def get(rc):
        r, c = rc
        part = cache / f"{r}_{c}.npz"
        if part.exists():
            z = np.load(part)
            return r, c, z["a"], (float(z["eecu"]) if z["eecu"] >= 0 else None)
        h, w = min(CHUNK_PX, n - r), min(CHUNK_PX, n - c)
        req = {"expression": img, "fileFormat": "NUMPY_NDARRAY",
               "grid": {"dimensions": {"width": w, "height": h},
                        "affineTransform": {"scaleX": tr[0], "shearX": 0,
                                            "translateX": x0 + c * tr[0], "shearY": 0,
                                            "scaleY": tr[4], "translateY": y1 + r * tr[4]},
                        "crsCode": C.CRS_UTM}}
        arr, e = _profiled(lambda: ee.data.computePixels(req), f"window chunk {rc}")
        a = np.stack([arr[b] for b in BANDS]).astype("int16")
        np.savez_compressed(part, a=a, eecu=-1.0 if e is None else e)
        return r, c, a, e

    profile = {"driver": "GTiff", "width": n, "height": n, "count": len(BANDS), "dtype": "int16",
               "crs": C.CRS_UTM, "nodata": -32768, "compress": "deflate", "predictor": 2,
               "transform": from_origin(x0, y1, C.GRID_SCALE, C.GRID_SCALE)}
    eecu = []
    with rasterio.open(WIN_TIF, "w", **profile) as dst:
        dst.descriptions = tuple(BANDS)
        with ThreadPoolExecutor(workers) as ex:
            for k, (r, c, a, e) in enumerate(ex.map(get, jobs), 1):
                dst.write(a, window=Window(c, r, a.shape[2], a.shape[1]))
                eecu.append(e)
                log.info(f"  window chunk {k}/{len(jobs)} ({e or 0:.0f} EECU-s)")
        dst.update_tags(scales=str(SCALE_OF), lpd_low=LPD_LOW,
                        flags="1=outside AOA; 2=bare-soil observations below training range; "
                              "4=inside AOA with LPD<=2",
                        domain="1=WorldCover cropland only; 2=Dynamic World crops only; 3=both")
    known = [e for e in eecu if e is not None]
    save_json({"bounds": [x0, y0, x1, y1], "pixels": n * n, "chunks": len(jobs),
               "eecu_h": sum(known) / 3600, "chunks_profiled": len(known)},
              C.MAPS / "window_meta.json")
    log.info(f"wrote {WIN_TIF} ({sum(known) / 3600:.2f} EECU-h)")


# ------------------------------------------------------------------ 4. summary
def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / d
    return [float(c - h), float(c + h)]


def window_parity(P):
    import rasterio
    if not WIN_TIF.exists():
        return None
    with rasterio.open(WIN_TIF) as src:
        b = src.bounds
        inw = P[P.x_utm.between(b.left, b.right) & P.y_utm.between(b.bottom, b.top)]
        smp = np.array(list(src.sample(zip(inw.x_utm, inw.y_utm),
                                       indexes=[BANDS.index(k) + 1 for k in ("OM", "DI", "flags")])))
    mapped = smp[:, 0] != -32768
    both = mapped & inw.valid.to_numpy()
    d_om = np.abs(smp[both, 0] / 100 - inw.OM.to_numpy()[both]) * C.OM_REPORT_FACTOR
    d_di = np.abs(smp[both, 1] / 1000 - inw.DI.to_numpy()[both])
    return {"n_nodes": int(len(inw)), "n_mapped_both": int(both.sum()),
            "n_valid_mismatch": int((mapped != inw.valid.to_numpy()).sum()),
            "max_abs_diff_g_kg": float(d_om.max()) if both.any() else None,
            "max_abs_diff_di": float(d_di.max()) if both.any() else None,
            "flags_equal": bool((smp[both, 2] == inw["flags"].to_numpy()[both]).all())}


def summary():
    dep = load_json(DEP_JSON)
    meta = load_json(C.MAPS / "lattice_meta.json")
    nodes = pd.read_csv(NODES, index_col="node")
    P = pd.read_csv(PRED, index_col="node")
    meta["roi_km2"] = roi_km2()
    N, A = len(nodes), meta["roi_km2"]
    v = P[P.valid]
    n = len(v)
    F = C.OM_REPORT_FACTOR

    def area(k):
        p = k / N
        return {"km2": p * A, "se_km2": A * np.sqrt(p * (1 - p) / N),
                "ci95_km2": [x * A for x in wilson(k, N)]}

    def share(mask):
        k = int(mask.sum())
        return {"k": k, "p": k / n, "ci95": wilson(k, n)}

    fl = v["flags"].astype(int)
    som = v.OM * F
    diff = (v.OM_all - v.OM) * F

    def quant(s):
        return {f"p{int(p * 100):02d}": float(s.quantile(p)) for p in (0.05, 0.25, 0.5, 0.75, 0.95)}

    def rho_x(m):
        from scipy.stats import spearmanr
        return float(spearmanr(v.x_utm[m], v.OM[m])[0])

    def diff_stats(m):
        return {"mean": float(diff[m].mean()), "mae": float(diff[m].abs().mean()),
                "median": float(diff[m].median())}

    in_aoa = (fl & 1) == 0

    res = {
        "design": {**{k: meta[k] for k in ("lattice_m", "offset_m", "roi_km2", "n_nodes")},
                   "node_km2": (LATTICE_M / 1000) ** 2,
                   "n_domain": int((nodes.domain > 0).sum()), "n_mapped": n,
                   "n_domain_few_bare": int((~P.valid).sum()),
                   "eecu_h": {"domain": meta["cost_domain"]["eecu_h"],
                              "features": meta["cost_features"]["eecu_h"]}},
        "domain_area": area(int((nodes.domain > 0).sum())),
        "mapped_area": area(n),
        "area_km2": n / N * A,
        "domain_share": {"worldcover_only": share(v.domain == 1), "dynamicworld_only": share(v.domain == 2),
                         "both": share(v.domain == 3)},
        "share_outside_aoa": share((fl & 1) > 0),
        "share_bare_below_train": share((fl & 2) > 0),
        "share_low_lpd_inside_aoa": share((fl & 4) > 0),
        "share_any_flag": share(fl > 0),
        "priority_share": {PRIORITY[c]: share(v.cls == c) for c in (1, 2, 3)},
        "priority_area": {PRIORITY[c]: area(int((v.cls == c).sum())) for c in (1, 2, 3)},
        "low_som_threshold_g_kg": LOW_SOM_PCT * F,
        "som_quantiles_g_kg": {"all": quant(som), "aoa": quant(som[in_aoa]),
                               "supported": quant(som[fl == 0]),
                               "class_A": quant(som[v.cls == 1]),
                               "class_C": quant(som[v.cls == 3])},
        "som_mean_g_kg": {"all": float(som.mean()), "aoa": float(som[in_aoa].mean()),
                          "supported": float(som[fl == 0].mean())},
        "optical_shortfall": area(int((nodes.domain > 0).sum()) - n),
        "spearman_som_easting": {"all": rho_x(np.ones(n, bool)), "aoa": rho_x(in_aoa.to_numpy())},
        "all_fit_minus_deployed_mean_g_kg": float(diff.mean()),
        "all_fit_minus_deployed_mae_g_kg": float(diff.abs().mean()),
        "all_fit_minus_deployed_by_aoa_g_kg": {"inside": diff_stats(in_aoa), "outside": diff_stats(~in_aoa)},
        "interval_abs_halfwidth_g_kg": dep["q_abs"] * F,
        "interval_log_factor": float(np.exp(dep["q_log"])),
        "window": None,
    }
    wm = C.MAPS / "window_meta.json"
    if wm.exists():
        res["window"] = {**load_json(wm), "parity": window_parity(P)}
    save_json(res, C.MAPS / "map_summary.json")
    log.info(f"domain {res['domain_area']['km2']:,.0f} km2, mapped {res['area_km2']:,.0f} km2 "
             f"({n} nodes); any flag {res['share_any_flag']['p']:.3f}; window {res['window']}")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "fit"
    {"fit": fit, "lattice": lattice, "predict": predict_nodes, "window": window,
     "summary": summary}[step]()
