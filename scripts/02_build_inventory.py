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

from cushing.inventory import OverpassError, fetch_osm_tanks  # noqa: E402


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


if __name__ == "__main__":
    main()
