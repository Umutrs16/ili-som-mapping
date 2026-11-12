"""Compare a 10 m buffer mean with the native 20 m cell.

Same Point, buffer, CRS and crsTransform as the mapping extraction, on a
synthetic image. Adjacent cells differ by 1, so the difference is the neighbour weight.

    python s08c_support_check.py
"""
from __future__ import annotations

import pandas as pd

import gee_common as G
from utils import C, get_logger, save_json, utm_xy

log = get_logger("s08c")


def main():
    import ee
    G.init()
    tr = G.grid_transform(C.GRID_SCALE)
    proj = ee.Projection(C.CRS_UTM, transform=tr)
    # With a metre transform, pixelCoordinates returns pixel indices, not metres.
    # Adjacent 20 m cells therefore differ by 1. A weight of w on a neighbour
    # moves the mean by w.
    xy = ee.Image.pixelCoordinates(proj)
    img = xy.select("x").floor().multiply(1000).add(xy.select("y").floor()).rename("code")
    img = img.setDefaultProjection(proj)

    nodes = pd.read_csv(C.MAPS / "lattice_predictions.csv")
    centres = nodes.loc[nodes.valid == 1, ["x_utm", "y_utm"]].iloc[::400].head(8)
    soil = pd.read_csv(C.RAW / "soil_audited.csv")
    sx, sy = utm_xy(soil.lon, soil.lat)
    samples = pd.DataFrame({"x_utm": sx, "y_utm": sy}).iloc[[0, 30, 60, 90, 120]]

    feats = []
    for kind, df in (("centre", centres), ("sample", samples)):
        for i, r in df.iterrows():
            g = ee.Geometry.Point([float(r.x_utm), float(r.y_utm)], C.CRS_UTM)
            feats.append(ee.Feature(g.buffer(C.POINT_BUFFER_M),
                                    {"i": int(i), "kind": kind, "op": "buffer"}))
            feats.append(ee.Feature(g, {"i": int(i), "kind": kind, "op": "point"}))
    red = img.reduceRegions(ee.FeatureCollection(feats), ee.Reducer.mean(),
                            crs=C.CRS_UTM, crsTransform=tr)
    got = [f["properties"] for f in red.getInfo()["features"]]
    rec = {}
    for f in got:
        rec.setdefault((f["kind"], f["i"]), {})[f["op"]] = f["mean"]
    rows = []
    for (kind, i), d in rec.items():
        rows.append({"kind": kind, "i": i, "buffer": d["buffer"], "point": d["point"],
                     "abs_diff": abs(d["buffer"] - d["point"])})
    pdf = pd.DataFrame(rows)
    pdf.to_csv(C.RES / "support_identity.csv", index=False)
    summary = {
        "crs": C.CRS_UTM,
        "crs_transform": tr,
        "buffer_m": C.POINT_BUFFER_M,
        "reducer": "ee.Reducer.mean (area-weighted default)",
        "max_abs_diff_centres": float(pdf.loc[pdf.kind == "centre", "abs_diff"].max()),
        "max_abs_diff_samples": float(pdf.loc[pdf.kind == "sample", "abs_diff"].max()),
        "x_neighbour_step": 1000,
        "note": "Centres match exactly. Sample buffers mix neighbours; abs_diff/1000 is the weight on an east-west neighbour.",
    }
    save_json(summary, C.RES / "support_identity.json")
    log.info(pdf.to_string(index=False))
    log.info(str(summary))


if __name__ == "__main__":
    main()
