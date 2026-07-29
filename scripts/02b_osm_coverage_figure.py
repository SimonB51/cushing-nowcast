"""Étape 2 (partiel) — figure de contrôle : empreintes OSM sur une scène Sentinel-2 récente.

Sert uniquement à juger visuellement la couverture OSM avant d'écrire la
logique de rayon / hauteur / type de toit. Ne fait pas partie du pipeline
imagerie de l'étape 3 (pas de masquage SCL, pas de métadonnées solaires
conservées ici).
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import planetary_computer
import pystac_client
import rioxarray
import yaml
from shapely.geometry import box

REPO_ROOT = Path(__file__).resolve().parents[1]
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
MAX_CLOUD_PCT = 20.0
# Fenêtre glissante : la figure doit rester reproductible dans le temps, pas
# dépendre d'une date figée dans le code.
SEARCH_DAYS_BACK = 120
SEARCH_WINDOW = f"{(date.today() - timedelta(days=SEARCH_DAYS_BACK)).isoformat()}/{date.today().isoformat()}"

# Les figures sont légendées en anglais : elles sont affichées dans le README public.
FOOTPRINT_COLOR = "#eb6834"
INK = "#0b0b0b"
INK_SOFT = "#52514e"


def pick_best_scene(catalog, bbox: list[float]):
    aoi_box = box(*bbox)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=SEARCH_WINDOW,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD_PCT}},
    )
    items = list(search.items())
    if not items:
        raise RuntimeError(f"Aucune scène Sentinel-2 avec <{MAX_CLOUD_PCT}% de nuages sur {SEARCH_WINDOW}.")

    scored = []
    for item in items:
        from shapely.geometry import shape
        geom = shape(item.geometry)
        coverage = geom.intersection(aoi_box).area / aoi_box.area
        scored.append((coverage, item.datetime, item))

    # priorité: couvre bien l'AOI, puis le plus récent
    scored.sort(key=lambda t: (t[0] > 0.9, t[1]), reverse=True)
    best_coverage, best_dt, best_item = scored[0]
    print(f"Scène choisie: {best_item.id} ({best_dt.date()}), couverture AOI={best_coverage:.1%}, "
          f"nuages={best_item.properties.get('eo:cloud_cover'):.1f}%")
    return best_item


def densest_window(tanks: gpd.GeoDataFrame, cell_m: float = 1000.0,
                   half_width_m: float = 1250.0) -> tuple[float, float, float, float]:
    """Fenêtre carrée centrée sur la cellule de grille qui contient le plus de cuves.

    Sert à l'encart de zoom : à l'échelle de l'AOI (18 km), une cuve fait ~5 px
    et on ne peut pas juger l'alignement des emprises.
    """
    cx = tanks.geometry.centroid.x
    cy = tanks.geometry.centroid.y
    keys = list(zip((cx // cell_m).astype(int), (cy // cell_m).astype(int)))
    best_key = max(set(keys), key=keys.count)
    in_cell = [k == best_key for k in keys]
    mid_x = cx[in_cell].mean()
    mid_y = cy[in_cell].mean()
    return (mid_x - half_width_m, mid_y - half_width_m,
            mid_x + half_width_m, mid_y + half_width_m)


def main() -> None:
    settings = yaml.safe_load((REPO_ROOT / "config" / "settings.yaml").read_text())
    aoi = settings["aoi"]
    bbox = [aoi["min_lon"], aoi["min_lat"], aoi["max_lon"], aoi["max_lat"]]

    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    item = pick_best_scene(catalog, bbox)

    rgb = rioxarray.open_rasterio(item.assets["visual"].href)
    aoi_proj = gpd.GeoSeries([box(*bbox)], crs="EPSG:4326").to_crs(rgb.rio.crs)
    rgb_clip = rgb.rio.clip_box(*aoi_proj.total_bounds)

    tanks_path = REPO_ROOT / "data" / "interim" / "osm_tanks_raw.geojson"
    tanks = gpd.read_file(tanks_path).to_crs(rgb.rio.crs)

    fig, ax = plt.subplots(figsize=(9, 9))
    rgb_clip.plot.imshow(ax=ax)
    tanks.boundary.plot(ax=ax, color=FOOTPRINT_COLOR, linewidth=0.8)
    ax.set_title(
        f"OSM storage-tank footprints (n={len(tanks)}) over Sentinel-2 L2A, "
        f"{item.datetime.date()} — Cushing, OK",
        fontsize=12, color=INK, pad=10,
    )
    ax.set_axis_off()

    # Encart : zoom sur le parc de cuves le plus dense, pour juger l'alignement.
    zx0, zy0, zx1, zy1 = densest_window(tanks)
    axins = ax.inset_axes([0.655, 0.655, 0.335, 0.335])
    rgb_clip.plot.imshow(ax=axins)
    tanks.boundary.plot(ax=axins, color=FOOTPRINT_COLOR, linewidth=0.9)
    axins.set(xlim=(zx0, zx1), ylim=(zy0, zy1), title="", xlabel="", ylabel="")
    axins.set_xticks([])
    axins.set_yticks([])
    for spine in axins.spines.values():
        spine.set(color="white", linewidth=1.5)
    # Fond blanc : le texte se pose sur l'imagerie et serait sinon illisible.
    axins.text(
        0.5, 0.0, "Zoom: densest tank farm, 2.5 km across",
        transform=axins.transAxes, ha="center", va="bottom", fontsize=8, color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2.5},
    )
    ax.indicate_inset_zoom(axins, edgecolor="white", linewidth=1.2, alpha=0.9)

    fig.text(
        0.5, 0.045,
        f"Orange outlines: {len(tanks)} OpenStreetMap footprints (man_made=storage_tank). "
        f"Background: Sentinel-2 true-colour, cloud cover {item.properties.get('eo:cloud_cover'):.1f}%.",
        fontsize=8, color=INK_SOFT, ha="center",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "osm_coverage_preview.png"
    fig.savefig(out_path, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"Figure écrite: {out_path}")


if __name__ == "__main__":
    main()
