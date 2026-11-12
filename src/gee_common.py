"""Earth Engine helpers shared by point extraction and map export."""
from __future__ import annotations

import ee
import geopandas as gpd
import numpy as np
from shapely.geometry import mapping

from utils import C, load_soil

_INIT = False


def init() -> None:
    global _INIT
    if not _INIT:
        ee.Initialize(project=C.GEE_PROJECT)
        _INIT = True


def roi_geometry(simplify_m: float = 0.0) -> ee.Geometry:
    g = gpd.read_file(C.ROI_SHP).to_crs(4326)
    geom = g.geometry.union_all()
    if simplify_m > 0:
        geom = gpd.GeoSeries([geom], crs=4326).to_crs(C.CRS_UTM).simplify(simplify_m) \
            .to_crs(4326).iloc[0]
    return ee.Geometry(mapping(geom), proj="EPSG:4326", geodesic=False)


def sample_points() -> ee.FeatureCollection:
    soil = load_soil()
    feats = [ee.Feature(ee.Geometry.Point([float(r.lon), float(r.lat)]),
                        {"sample_id": sid}) for sid, r in soil.iterrows()]
    return ee.FeatureCollection(feats)


def grid_transform(scale: int) -> list:
    """Fixed EPSG:32644 grid anchored at multiples of `scale` metres."""
    return [scale, 0, 0, 0, -scale, 0]
