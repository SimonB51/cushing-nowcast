"""Contrôles de cohérence sur l'inventaire des cuves (DESIGN §6).

Contrairement à test_eia.py, ces tests ne se sautent pas sur un clone frais :
`config/tanks.geojson` est versionné.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SETTINGS = yaml.safe_load((REPO_ROOT / "config" / "settings.yaml").read_text())
TANKS_PATH = REPO_ROOT / SETTINGS["paths"]["tanks_geojson"]

EXPECTED_COLUMNS = {
    "tank_id", "centroid_lon", "centroid_lat", "radius_m", "height_m",
    "roof_type", "capacity_kbbl", "first_seen", "last_seen", "notes", "geometry",
}


@pytest.fixture(scope="module")
def tanks() -> gpd.GeoDataFrame:
    if not TANKS_PATH.exists():
        pytest.skip(f"{TANKS_PATH} absent — lance scripts/02_build_inventory.py d'abord.")
    return gpd.read_file(TANKS_PATH)


def test_schema(tanks: gpd.GeoDataFrame) -> None:
    missing = EXPECTED_COLUMNS - set(tanks.columns)
    assert not missing, f"Colonnes manquantes dans l'inventaire: {sorted(missing)}"


def test_tank_ids_unique(tanks: gpd.GeoDataFrame) -> None:
    dup = tanks["tank_id"][tanks["tank_id"].duplicated()]
    assert dup.empty, f"tank_id dupliqué(s): {dup.tolist()[:5]}"


def test_radii_plausible(tanks: gpd.GeoDataFrame) -> None:
    """Une cuve de brut à Cushing fait entre ~5 et ~60 m de rayon."""
    r = tanks["radius_m"]
    assert (r >= SETTINGS["tanks"]["min_radius_m"]).all(), "Rayon sous le seuil de filtrage"
    assert (r < 60).all(), f"Rayon aberrant: max {r.max():.1f} m"


def test_capacity_within_factor_two(tanks: gpd.GeoDataFrame) -> None:
    """Contrôle de cohérence obligatoire du DESIGN §6.

    Comparaison à la SHELL capacity, pas à la working : pi*r^2*H est un volume
    géométrique de cylindre. Comparer à la working (78,4 Mbbl) ferait apparaître
    un faux excédent de +25 %.

    Volontairement large : la hauteur reste une hypothèse. Ce test attrape une
    erreur d'ordre de grandeur (mauvaise unité, mauvais rayon, AOI trop large),
    pas une erreur fine. Il ne valide pas l'inventaire, il détecte qu'il est faux.
    """
    cfg = SETTINGS["tanks"]
    total_mbbl = tanks["capacity_kbbl"].sum() / 1000
    ref = float(cfg["capacity_reference_shell_mbbl"])
    factor = float(cfg["capacity_tolerance_factor"])

    assert ref / factor < total_mbbl < ref * factor, (
        f"Capacité totale {total_mbbl:.1f} Mbbl hors du facteur {factor} autour de "
        f"la shell capacity de référence {ref:.1f} Mbbl ({cfg['capacity_reference_asof']})."
    )


def test_shell_and_working_reference_are_distinct(tanks: gpd.GeoDataFrame) -> None:
    """Garde-fou : empêche qu'on rebranche un jour le contrôle sur la working.

    L'EIA publie les deux pour Cushing, avec ~25 % d'écart. Les confondre est
    l'erreur silencieuse la plus coûteuse de cette étape.
    """
    cfg = SETTINGS["tanks"]
    ratio = cfg["capacity_reference_shell_mbbl"] / cfg["capacity_reference_working_mbbl"]
    assert 1.15 < ratio < 1.35, (
        f"Rapport shell/working = {ratio:.3f}, hors de la plage attendue (~1,25). "
        "Une des deux références a probablement été modifiée par erreur."
    )


def test_capacity_matches_geometry(tanks: gpd.GeoDataFrame) -> None:
    """capacity_kbbl doit être reconstructible depuis radius_m et height_m."""
    expected = np.pi * tanks["radius_m"] ** 2 * tanks["height_m"] * 6.2898 / 1000
    assert np.allclose(tanks["capacity_kbbl"], expected, rtol=1e-3), (
        "capacity_kbbl incohérent avec la géométrie déclarée"
    )


def test_unresolved_fields_are_not_invented(tanks: gpd.GeoDataFrame) -> None:
    """roof_type et first/last_seen exigent l'imagerie — ils doivent rester vides.

    Ce test existe pour attraper une régression où un futur commit remplirait ces
    champs par défaut plutôt que par mesure.
    """
    assert (tanks["roof_type"] == "unknown").all(), (
        "roof_type renseigné alors que la détection par variance temporelle "
        "(étape 4) n'existe pas encore"
    )
    for col in ("first_seen", "last_seen"):
        assert tanks[col].isna().all(), f"{col} renseigné sans imagerie multi-dates"
