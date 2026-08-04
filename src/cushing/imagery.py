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
import pandas as pd
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


def search_scenes(catalog, bbox: list[float], start: str, end: str) -> list:
    """Tous les items Sentinel-2 L2A intersectant `bbox` sur la période.

    Aucun filtre nuage ici : le taux de rejet est un résultat à mesurer, pas une
    donnée à écarter en silence. Le filtrage se fait en aval, explicitement.
    """
    search = catalog.search(
        collections=[COLLECTION], bbox=bbox, datetime=f"{start}/{end}"
    )
    items = list(search.items())
    if not items:
        raise ImageryError(f"Aucune scène {COLLECTION} sur {bbox} entre {start} et {end}.")
    return items


def eia_week_ending(ts: pd.Timestamp) -> pd.Timestamp:
    """Vendredi de fin de semaine EIA auquel une acquisition se rattache.

    Règle d'intégrité temporelle du DESIGN §9 : le chiffre EIA daté du vendredi V
    ne peut être prédit qu'avec des images acquises AU PLUS TARD le vendredi V.
    On rattache donc chaque scène au premier vendredi >= sa date d'acquisition.
    Une image du samedi appartient à la semaine suivante, jamais à celle qui
    vient de se clore.
    """
    # lundi=0 ... vendredi=4 ... dimanche=6
    days_ahead = (4 - ts.weekday()) % 7
    return (ts.normalize() + pd.Timedelta(days=days_ahead)).normalize()


def scene_metadata(items: list, bbox: list[float]) -> pd.DataFrame:
    """Métadonnées exploitables, une ligne par item STAC.

    Les angles solaires sont OBLIGATOIRES (DESIGN §7) : sans eux la correction du
    confondant saisonnier est impossible. Un item qui en manque est signalé.
    """
    frame_box = box(*bbox)
    rows = []
    for it in items:
        props = it.properties
        zenith = props.get("s2:mean_solar_zenith")
        azimuth = props.get("s2:mean_solar_azimuth")
        ts = pd.Timestamp(it.datetime).tz_localize(None)
        rows.append({
            "item_id": it.id,
            "datetime": ts,
            "acquisition_date": ts.normalize(),
            "eia_week_ending": eia_week_ending(ts),
            "mgrs_tile": props.get("s2:mgrs_tile"),
            "relative_orbit": props.get("sat:relative_orbit"),
            "cloud_cover_pct": props.get("eo:cloud_cover"),
            "sun_elevation_deg": None if zenith is None else 90.0 - zenith,
            "sun_azimuth_deg": azimuth,
            "frame_coverage": shape(it.geometry).intersection(frame_box).area / frame_box.area,
        })

    df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)

    missing = df["sun_elevation_deg"].isna()
    if missing.any():
        logger.warning(
            "%d item(s) sans métadonnées solaires — inutilisables pour l'étape 4: %s",
            int(missing.sum()), df.loc[missing, "item_id"].head(3).tolist(),
        )
    return df


def load_rgb_clip(item, bounds_4326: list[float]):
    """Asset `visual` (RGB 8 bits) découpé sur une emprise donnée en WGS84.

    Renvoie le raster et son CRS projeté. Pas de masquage nuage : cet asset sert
    aux figures, pas à la mesure radiométrique.
    """
    rgb = rioxarray.open_rasterio(item.assets["visual"].href)
    aoi_proj = gpd.GeoSeries([box(*bounds_4326)], crs="EPSG:4326").to_crs(rgb.rio.crs)
    return rgb.rio.clip_box(*aoi_proj.total_bounds), rgb.rio.crs
