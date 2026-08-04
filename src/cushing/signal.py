"""Étape 4 — extraction du signal radiométrique par cuve et par scène.

Produit, pour chaque couple (cuve, scène exploitable), les statistiques de
réflectance dans l'emprise de la cuve, plus les métadonnées solaires nécessaires
à la correction du confondant saisonnier (DESIGN §8).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  (enregistre l'accesseur .rio)
from rasterio.enums import Resampling
from rasterio.features import rasterize
from scipy.ndimage import binary_dilation

logger = logging.getLogger(__name__)

# --- Correction radiométrique de la baseline 04.00 -------------------------
#
# À partir de la baseline de traitement 04.00 (2022-01-25), les produits
# Sentinel-2 L2A portent un décalage BOA_ADD_OFFSET = -1000 sur les entiers
# stockés. Les catalogues STAC du Planetary Computer n'exposent PAS cet offset
# (`raster:bands` est absent), il faut donc l'appliquer soi-même.
#
# Vérifié empiriquement sur ce jeu de données (médiane B04 sur l'emprise des
# cuves, scènes claires d'octobre) : 897/766/709 DN avant, 2136/1710/1952 après.
# Saut de ~+1140 DN sans cause physique, au milieu de la période d'étude.
#
# Sans cette correction, la série présente une marche artificielle de +0,1 en
# réflectance en janvier 2022. La calibration l'interpréterait comme un
# changement de niveau des stocks.
BASELINE_OFFSET_FROM = "04.00"
BOA_ADD_OFFSET = -1000.0
REFLECTANCE_SCALE = 10000.0


# --- Masquage nuage : pourquoi SCL n'est PAS appliqué dans les emprises ----
#
# Le classifieur SCL confond les toits de cuve avec des nuages. Mesuré sur la
# scène 2021-10-11 (0,14 % de nuages annoncés à l'échelle de la tuile) :
#
#   dans les emprises de cuves .... 86,0 % de pixels classés nuage
#   anneau 0-20 m autour .......... 30,3 %
#   anneau 20-50 m ................  2,8 %
#   anneau 50-150 m ...............  1,4 %
#
# Le masque décroît avec la distance aux cuves : c'est la signature d'une erreur
# de classification par pixel, pas d'un nuage. Un nuage à 10 m de résolution fait
# au minimum plusieurs centaines de mètres et masquerait un bloc contigu.
# La réflectance le confirme : 4488 DN dans les cuves contre 702 sur le fond —
# les toits sont réellement brillants, et SCL lit la brillance comme un nuage.
#
# Appliquer SCL tel quel écarterait préférentiellement les cuves les PLUS
# brillantes. Le biais serait alors corrélé à la grandeur mesurée, ce qui est
# bien pire qu'une perte de données aléatoire.
#
# Solution retenue : juger la nébulosité sur un ANNEAU autour de chaque cuve,
# où SCL est fiable, et appliquer ce verdict à la cuve. On n'utilise jamais la
# brillance d'une cuve pour décider de la rejeter.
RING_INNER_M = 50.0
RING_OUTER_M = 150.0


class SignalError(RuntimeError):
    pass


def _baseline_as_tuple(baseline: str) -> tuple[int, ...]:
    """'04.00' -> (4, 0). Comparaison numérique, jamais lexicographique.

    Une comparaison de chaînes donnerait '3.00' > '04.00' si l'ESA cessait un
    jour de préfixer par zéro — et appliquerait alors l'offset à des produits
    antérieurs à la bascule.
    """
    try:
        return tuple(int(part) for part in str(baseline).split("."))
    except ValueError as exc:
        raise SignalError(f"Baseline de traitement illisible: {baseline!r}") from exc


def boa_offset(processing_baseline: str | None) -> float:
    """Offset à ajouter aux entiers bruts avant mise à l'échelle."""
    if processing_baseline is None:
        raise SignalError(
            "s2:processing_baseline absent de l'item STAC — impossible de savoir "
            "s'il faut appliquer BOA_ADD_OFFSET. Refus de deviner."
        )
    if _baseline_as_tuple(processing_baseline) >= _baseline_as_tuple(BASELINE_OFFSET_FROM):
        return BOA_ADD_OFFSET
    return 0.0


def _open_band(item, band: str):
    return rioxarray.open_rasterio(item.assets[band].href)


def _read_band(item, band: str, bounds, target=None):
    """Lecture fenêtrée d'une bande, rééchantillonnée sur `target` si fourni."""
    da = _open_band(item, band).rio.clip_box(*bounds)
    if target is not None:
        da = da.rio.reproject_match(target, resampling=Resampling.nearest)
    return da


def _tank_label_array(tanks_proj, template) -> np.ndarray:
    """Rasterise les emprises : 0 = hors cuve, i+1 = index de la cuve i."""
    transform = template.rio.transform()
    height, width = template.shape[-2:]
    shapes = ((geom, i + 1) for i, geom in enumerate(tanks_proj.geometry))
    return rasterize(
        shapes, out_shape=(height, width), transform=transform,
        fill=0, dtype="int32", all_touched=False,
    )


def extract_scene_signals(
    item,
    tanks,
    bands: list[str],
    scl_exclude: list[int],
    min_valid_frac: float,
) -> pd.DataFrame:
    """Statistiques de réflectance par cuve pour une scène.

    Les pixels masqués sont mis à NaN puis agrégés par `nanmean`/`nanpercentile` :
    jamais à zéro, ce qui contaminerait toutes les moyennes (DESIGN §7).
    """
    props = item.properties
    offset = boa_offset(props.get("s2:processing_baseline"))

    # une seule ouverture pour déterminer le CRS natif, puis emprise et fenêtre
    ref_open = _open_band(item, bands[0])
    tanks_proj = tanks.to_crs(ref_open.rio.crs)
    bounds = tuple(tanks_proj.total_bounds)

    ref = ref_open.rio.clip_box(*bounds)
    labels = _tank_label_array(tanks_proj, ref)

    scl = _read_band(item, "SCL", bounds, target=ref)
    cloudy = np.isin(scl.values[0], scl_exclude)

    # Nébulosité jugée sur un anneau autour de chaque cuve, jamais sur la cuve
    # elle-même (voir la note sur SCL plus haut).
    px = abs(float(ref.rio.resolution()[0]))
    in_tank = labels > 0
    ring = (
        binary_dilation(in_tank, iterations=int(round(RING_OUTER_M / px)))
        & ~binary_dilation(in_tank, iterations=int(round(RING_INNER_M / px)))
    )
    ring_cloud_frac = float(cloudy[ring].mean()) if ring.any() else float("nan")

    stack = {}
    for b in bands:
        da = ref if b == bands[0] else _read_band(item, b, bounds, target=ref)
        arr = da.values[0].astype("float32")
        arr[arr == 0] = np.nan               # no-data du produit
        arr = (arr + offset) / REFLECTANCE_SCALE
        # SCL n'est appliqué QUE hors des emprises : dans les cuves il masquerait
        # la brillance, c'est-à-dire le signal lui-même.
        arr[cloudy & ~in_tank] = np.nan
        stack[b] = arr

    zenith = props.get("s2:mean_solar_zenith")
    ts = pd.Timestamp(item.datetime).tz_localize(None)

    flat_labels = labels.ravel()
    order = np.argsort(flat_labels, kind="stable")
    sorted_labels = flat_labels[order]
    starts = np.searchsorted(sorted_labels, np.arange(1, len(tanks_proj) + 1), side="left")
    ends = np.searchsorted(sorted_labels, np.arange(1, len(tanks_proj) + 1), side="right")

    rows = []
    for i, (lo, hi) in enumerate(zip(starts, ends)):
        if hi <= lo:
            continue                          # cuve hors de l'emprise du raster
        idx = order[lo:hi]
        n_total = hi - lo

        primary = stack[bands[0]].ravel()[idx]
        valid = np.isfinite(primary)
        frac = float(valid.mean())
        if frac < min_valid_frac:
            continue                          # rejet explicite (DESIGN §8)
        if ring_cloud_frac > (1.0 - min_valid_frac):
            continue                          # voisinage nuageux -> cuve douteuse

        row = {
            "tank_id": tanks.iloc[i]["tank_id"],
            "item_id": item.id,
            "scene_date": ts.normalize(),
            "valid_pixel_frac": round(frac, 4),
            "ring_cloud_frac": round(ring_cloud_frac, 4),
            "n_pixels": int(n_total),
            "sun_elevation_deg": None if zenith is None else round(90.0 - zenith, 3),
            "sun_azimuth_deg": props.get("s2:mean_solar_azimuth"),
            "processing_baseline": props.get("s2:processing_baseline"),
        }
        for b in bands:
            v = stack[b].ravel()[idx]
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            p10, p90 = np.percentile(v, [10, 90])
            row[f"{b}_mean"] = round(float(v.mean()), 5)
            row[f"{b}_median"] = round(float(np.median(v)), 5)
            row[f"{b}_p10"] = round(float(p10), 5)
            row[f"{b}_p90"] = round(float(p90), 5)
            row[f"{b}_contrast"] = round(float(p90 - p10), 5)
        rows.append(row)

    return pd.DataFrame(rows)
