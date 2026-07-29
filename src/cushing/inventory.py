"""Étape 2 (partiel) — inventaire des cuves.

Pour l'instant : récupération brute des empreintes OpenStreetMap via Overpass.
La logique de rayon, hauteur estimée et type de toit sera ajoutée après
validation de la couverture OSM.
"""

from __future__ import annotations

import logging
import time

import geopandas as gpd
import requests
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)

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
