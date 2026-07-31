"""Étape 2 (partiel) — requête Overpass OSM uniquement, pour juger la couverture.

Ne calcule pas encore rayon / hauteur / type de toit. Sauvegarde les
empreintes brutes et affiche un décompte.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from cushing.inventory import (  # noqa: E402
    InventoryError,
    OverpassError,
    build_tank_inventory,
    capacity_report,
    fetch_osm_tanks,
)


def report_height_sensitivity(radii: np.ndarray, tanks_cfg: dict, ref_shell_mbbl: float) -> None:
    """Rejoue le contrôle de capacité sous les deux hypothèses de hauteur.

    Affiche les deux jeux de valeurs sans les départager : aucun seuil ici ne
    permet de trancher. La question ne se règle qu'à l'étape 4, en mesurant si
    grandes et petites cuves ont des dynamiques différentes (voir DESIGN §8).

    Le total est comparé à la SHELL capacity : pi*r^2*H est un volume géométrique.
    """
    reports = [
        capacity_report(radii, {**tanks_cfg, "height_model": model})
        for model in ("flat", "by_radius")
    ]

    print("\n--- Sensibilité à l'hypothèse de hauteur ---")
    for r in reports:
        print(
            f"  {r['model']:10s} | total {r['total_kbbl']/1000:7.1f} Mbbl"
            f" | ratio vs shell {r['total_kbbl']/1000/ref_shell_mbbl:5.2f}"
            f" | part du top {r['n_top']} cuves {r['top_decile_share']:6.2%}"
        )

    flat, byr = reports
    gap_total = abs(byr["total_kbbl"] - flat["total_kbbl"]) / flat["total_kbbl"]
    gap_share = abs(byr["top_decile_share"] - flat["top_decile_share"])
    print(f"  écart relatif des totaux          : {gap_total:6.2%}")
    print(f"  écart de pondération (top décile) : {gap_share:6.2%} en points de part")


def main() -> None:
    settings = yaml.safe_load((REPO_ROOT / "config" / "settings.yaml").read_text())
    aoi = settings["aoi"]

    try:
        gdf = fetch_osm_tanks(aoi)
    except OverpassError as e:
        print(f"ARRÊT: {e}")
        sys.exit(1)

    print(f"Éléments OSM trouvés: {len(gdf)}")
    print(gdf["osm_type"].value_counts().to_string())
    print(gdf["man_made"].value_counts().to_string())

    out_path = REPO_ROOT / "data" / "interim" / "osm_tanks_raw.geojson"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GeoJSON")
    print(f"Écrit: {out_path}")

    tanks_cfg = settings["tanks"]
    try:
        inv = build_tank_inventory(gdf, tanks_cfg)
    except InventoryError as e:
        print(f"ARRÊT: {e}")
        sys.exit(1)

    inv_path = REPO_ROOT / settings["paths"]["tanks_geojson"]
    inv_path.parent.mkdir(parents=True, exist_ok=True)
    inv.to_file(inv_path, driver="GeoJSON")
    print(f"\nInventaire: {len(inv)} cuves -> {inv_path}")
    print(f"  rayon    : {inv['radius_m'].min():.1f} - {inv['radius_m'].max():.1f} m")
    print(f"  modèle H : {tanks_cfg['height_model']}")

    report_height_sensitivity(
        inv["radius_m"].to_numpy(),
        tanks_cfg,
        float(tanks_cfg["capacity_reference_shell_mbbl"]),
    )


if __name__ == "__main__":
    main()
