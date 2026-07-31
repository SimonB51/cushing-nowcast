"""Étape 2 (partiel) — inventaire des cuves.

Pour l'instant : récupération brute des empreintes OpenStreetMap via Overpass.
La logique de rayon, hauteur estimée et type de toit sera ajoutée après
validation de la couverture OSM.
"""

from __future__ import annotations

import logging
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)

BBL_PER_M3 = 6.2898

# Instance principale overpass-api.de instable (504 "server too busy") au
# moment du développement (2026-07-27) — utilisation du mirroir lz4.
OVERPASS_URL = "https://lz4.overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT = 180
# Overpass renvoie 406 sur le User-Agent par défaut de `requests` — il en
# faut un explicite.
HEADERS = {"User-Agent": "cushing-nowcast/0.1 (research project, contact via github)"}
# L'infra publique Overpass est fréquemment surchargée (504/429) — on retente
# avant d'abandonner.
RETRYABLE_STATUS = {429, 504}
MAX_RETRIES = 5
RETRY_BACKOFF_S = 20

TANK_TAGS = [
    ("man_made", "storage_tank"),
    ("man_made", "petroleum_well"),
]


class OverpassError(RuntimeError):
    pass


def _build_query(aoi: dict) -> str:
    bbox = f"{aoi['min_lat']},{aoi['min_lon']},{aoi['max_lat']},{aoi['max_lon']}"
    clauses = []
    for key, value in TANK_TAGS:
        clauses.append(f'node["{key}"="{value}"]({bbox});')
        clauses.append(f'way["{key}"="{value}"]({bbox});')
        clauses.append(f'relation["{key}"="{value}"]({bbox});')
    body = "\n  ".join(clauses)
    return f"""
[out:json][timeout:{OVERPASS_TIMEOUT}];
(
  {body}
);
out body;
>;
out skel qt;
""".strip()


def fetch_osm_tanks(aoi: dict) -> gpd.GeoDataFrame:
    """Interroge Overpass pour man_made=storage_tank / petroleum_well sur l'AOI.

    Retourne un GeoDataFrame brut, une ligne par élément OSM taggé, avec les
    tags OSM tels quels. Aucun rayon, hauteur ni type de toit n'est calculé.
    """
    query = _build_query(aoi)

    resp = None
    for attempt in range(1, MAX_RETRIES + 1):
        resp = requests.post(
            OVERPASS_URL, data={"data": query}, headers=HEADERS, timeout=OVERPASS_TIMEOUT + 60
        )
        if resp.status_code == 200:
            break
        if resp.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES:
            logger.warning(
                "Overpass a répondu %d (tentative %d/%d), nouvelle tentative dans %ds",
                resp.status_code, attempt, MAX_RETRIES, RETRY_BACKOFF_S,
            )
            time.sleep(RETRY_BACKOFF_S)
            continue
        raise OverpassError(f"Overpass a répondu {resp.status_code}: {resp.text[:500]}")

    payload = resp.json()
    elements = payload.get("elements", [])

    nodes = {el["id"]: (el["lon"], el["lat"]) for el in elements if el["type"] == "node"}

    relations_skipped = 0
    records = []
    for el in elements:
        if el["type"] == "node":
            if "tags" not in el:
                continue  # nœud de géométrie d'un way, pas un élément taggé
            geom = Point(el["lon"], el["lat"])
        elif el["type"] == "way":
            if "tags" not in el:
                continue
            coords = [nodes[n] for n in el.get("nodes", []) if n in nodes]
            if len(coords) < 3:
                logger.warning(
                    "Way OSM %s ignoré: géométrie incomplète (%d nœuds résolus)",
                    el["id"], len(coords),
                )
                continue
            geom = Polygon(coords)
        else:
            relations_skipped += 1
            continue

        tags = el.get("tags", {})
        records.append({
            "osm_type": el["type"],
            "osm_id": el["id"],
            "man_made": tags.get("man_made"),
            "name": tags.get("name"),
            "operator": tags.get("operator"),
            "content": tags.get("content"),
            "geometry": geom,
        })

    if relations_skipped:
        logger.warning(
            "%d relation(s) OSM man_made=storage_tank/petroleum_well ignorée(s) "
            "(non gérées pour l'instant).", relations_skipped,
        )

    if not records:
        raise OverpassError(
            "Overpass n'a retourné aucun élément man_made=storage_tank/petroleum_well sur l'AOI."
        )

    gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
    return gdf


class InventoryError(RuntimeError):
    pass


def _circumscribed_radius_m(geom) -> float:
    """Rayon = distance moyenne du centroïde aux sommets.

    Les emprises OSM de Cushing sont des polygones réguliers (392/415 sont des
    16-gones exacts) : ce sont des cercles dessinés, pas des contours relevés.
    Pour un polygone régulier inscrit dans le cercle de la cuve, tous les
    sommets sont à distance R, donc leur distance moyenne au centroïde donne R
    directement. Passer par l'aire sous-estimerait R de 1,3 % sur un 16-gone.
    """
    cx, cy = geom.centroid.x, geom.centroid.y
    xs, ys = geom.exterior.coords.xy
    # le dernier point ferme l'anneau et duplique le premier
    d = np.hypot(np.asarray(xs[:-1]) - cx, np.asarray(ys[:-1]) - cy)
    return float(d.mean())


def assign_height_m(radii_m: np.ndarray, tanks_cfg: dict) -> np.ndarray:
    """Hauteur de paroi selon le modèle choisi dans settings.yaml.

    `flat`      : une hauteur unique pour toutes les cuves.
    `by_radius` : grille de hauteurs par classe de rayon.

    Les deux sont des hypothèses de domaine, pas des mesures. Voir le commentaire
    de la section `tanks` de settings.yaml.
    """
    model = tanks_cfg["height_model"]

    if model == "flat":
        return np.full(len(radii_m), float(tanks_cfg["height_m_flat"]))

    if model == "by_radius":
        grid = tanks_cfg["height_by_radius"]
        heights = np.full(len(radii_m), np.nan)
        for band in grid:
            bound = band["max_radius_m"]
            upper = np.inf if bound is None else float(bound)
            # première bande dont la borne haute n'est pas encore dépassée
            todo = np.isnan(heights) & (radii_m < upper)
            heights[todo] = float(band["height_m"])
        if np.isnan(heights).any():
            raise InventoryError(
                f"{int(np.isnan(heights).sum())} cuve(s) hors de toutes les bandes de "
                "`height_by_radius`. La dernière bande doit avoir max_radius_m: null."
            )
        return heights

    raise InventoryError(
        f"height_model inconnu: {model!r}. Valeurs acceptées: 'flat', 'by_radius'."
    )


def build_tank_inventory(gdf_raw: gpd.GeoDataFrame, tanks_cfg: dict) -> gpd.GeoDataFrame:
    """Construit l'inventaire exploitable à partir des empreintes OSM brutes.

    `roof_type` reste 'unknown' et `first_seen`/`last_seen` restent nuls : les
    deux exigent l'imagerie multi-dates de l'étape 3. Ils ne sont pas devinés.
    """
    tanks = gdf_raw[gdf_raw.geometry.geom_type == "Polygon"].copy()
    dropped_geom = len(gdf_raw) - len(tanks)

    tanks = tanks[tanks["man_made"] == "storage_tank"].copy()
    if tanks.empty:
        raise InventoryError("Aucune empreinte man_made=storage_tank exploitable.")

    # projection métrique locale, déduite des données plutôt que codée en dur
    utm = tanks.estimate_utm_crs()
    proj = tanks.to_crs(utm)

    radii = np.array([_circumscribed_radius_m(g) for g in proj.geometry])

    # contre-vérification : le rayon déduit de l'aire doit être cohérent
    r_from_area = np.sqrt(proj.area.to_numpy() / np.pi)
    spread = np.abs(radii - r_from_area) / radii
    if (spread > 0.05).any():
        n_bad = int((spread > 0.05).sum())
        logger.warning(
            "%d empreinte(s) où rayon-sommets et rayon-aire divergent de plus de 5%% "
            "(max %.1f%%) — géométries probablement non circulaires.",
            n_bad, 100 * spread.max(),
        )

    small = radii < float(tanks_cfg["min_radius_m"])
    if small.any():
        logger.warning(
            "%d empreinte(s) sous min_radius_m=%.1f m écartée(s).",
            int(small.sum()), float(tanks_cfg["min_radius_m"]),
        )
    keep = ~small
    tanks, proj, radii = tanks[keep], proj[keep], radii[keep]

    heights = assign_height_m(radii, tanks_cfg)
    volume_m3 = np.pi * radii**2 * heights
    centroids = proj.geometry.centroid.to_crs("EPSG:4326")

    inv = gpd.GeoDataFrame(
        {
            "tank_id": [f"osm_{t}_{i}" for t, i in zip(tanks["osm_type"], tanks["osm_id"])],
            "centroid_lon": centroids.x.to_numpy(),
            "centroid_lat": centroids.y.to_numpy(),
            "radius_m": radii.round(2),
            "height_m": heights.round(2),
            "roof_type": "unknown",       # étape 4 : variance temporelle intra-cuve
            "capacity_kbbl": (volume_m3 * BBL_PER_M3 / 1000).round(3),
            "first_seen": pd.NaT,         # étape 3 : première scène exploitable
            "last_seen": pd.NaT,
            "notes": f"OSM footprint, height model={tanks_cfg['height_model']}",
        },
        geometry=tanks.geometry.to_numpy(),
        crs="EPSG:4326",
    )

    if inv["tank_id"].duplicated().any():
        dup = inv.loc[inv["tank_id"].duplicated(), "tank_id"].tolist()
        raise InventoryError(f"tank_id dupliqué(s): {dup[:5]}")

    logger.info(
        "Inventaire: %d cuves retenues (%d géométries non polygonales et %d "
        "non-storage_tank écartées en amont).",
        len(inv), dropped_geom, len(gdf_raw) - dropped_geom - len(tanks),
    )
    return inv


def capacity_report(radii_m: np.ndarray, tanks_cfg: dict, top_frac: float = 0.10) -> dict:
    """Capacité totale et concentration, pour un modèle de hauteur donné.

    `top_decile_share` est la grandeur qui compte : la calibration contre l'EIA
    absorbe toute erreur multiplicative constante sur H, donc seule la
    pondération relative entre cuves influence l'indice agrégé.
    """
    heights = assign_height_m(radii_m, tanks_cfg)
    cap = np.pi * radii_m**2 * heights * BBL_PER_M3 / 1000  # kbbl

    order = np.argsort(cap)[::-1]
    n_top = max(1, int(round(top_frac * len(cap))))
    return {
        "model": tanks_cfg["height_model"],
        "n_tanks": len(cap),
        "total_kbbl": float(cap.sum()),
        "n_top": n_top,
        "top_decile_share": float(cap[order[:n_top]].sum() / cap.sum()),
    }
