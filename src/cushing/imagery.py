"""Accès à l'imagerie Sentinel-2 via le catalogue STAC du Planetary Computer.

Périmètre volontairement réduit : sélection d'une scène et découpe RGB pour les
figures de contrôle. Le masquage SCL, la conservation des métadonnées solaires et
le cache local relèvent de l'étape 3 et ne sont PAS ici — voir DESIGN §7. Ce
module a été extrait de scripts/02b pour être partagé, pas pour anticiper
l'étape 3.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import geopandas as gpd
import planetary_computer
import pystac_client
import rioxarray  # noqa: F401  (enregistre l'accesseur .rio)
from shapely.geometry import box, shape

logger = logging.getLogger(__name__)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"


class ImageryError(RuntimeError):
    pass


def open_catalog():
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def recent_window(days_back: int = 120) -> str:
    """Fenêtre glissante, pour que les figures restent reproductibles dans le temps."""
    today = date.today()
    return f"{(today - timedelta(days=days_back)).isoformat()}/{today.isoformat()}"


def pick_best_scene(catalog, bbox: list[float], window: str, max_cloud_pct: float = 20.0):
    """Scène la mieux couvrante sur l'AOI, puis la plus récente."""
    aoi_box = box(*bbox)
    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=window,
        query={"eo:cloud_cover": {"lt": max_cloud_pct}},
    )
    items = list(search.items())
    if not items:
        raise ImageryError(
            f"Aucune scène {COLLECTION} avec moins de {max_cloud_pct}% de nuages sur {window}."
        )

    scored = [
        (shape(it.geometry).intersection(aoi_box).area / aoi_box.area, it.datetime, it)
        for it in items
    ]
    scored.sort(key=lambda t: (t[0] > 0.9, t[1]), reverse=True)
    coverage, dt, item = scored[0]
    logger.info(
        "Scène retenue: %s (%s), couverture AOI=%.1f%%, nuages=%.1f%%",
        item.id, dt.date(), 100 * coverage, item.properties.get("eo:cloud_cover", float("nan")),
    )
    return item, coverage


def load_rgb_clip(item, bounds_4326: list[float]):
    """Asset `visual` (RGB 8 bits) découpé sur une emprise donnée en WGS84.

    Renvoie le raster et son CRS projeté. Pas de masquage nuage : cet asset sert
    aux figures, pas à la mesure radiométrique.
    """
    rgb = rioxarray.open_rasterio(item.assets["visual"].href)
    aoi_proj = gpd.GeoSeries([box(*bounds_4326)], crs="EPSG:4326").to_crs(rgb.rio.crs)
    return rgb.rio.clip_box(*aoi_proj.total_bounds), rgb.rio.crs
