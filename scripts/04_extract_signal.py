"""Étape 4 — extraction du signal radiométrique sur toutes les scènes exploitables.

Long : une requête HTTP par bande et par scène. Reprise sur interruption via le
parquet partiel, pour ne pas retélécharger ce qui est déjà mesuré.
"""

from __future__ import annotations

import sys
import time
from datetime import date
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cushing.imagery import ImageryError, open_catalog, scene_metadata, search_scenes  # noqa: E402
from cushing.signal import extract_scene_signals  # noqa: E402


def main() -> None:
    settings = yaml.safe_load((REPO_ROOT / "config" / "settings.yaml").read_text())
    img = settings["imagery"]

    inv_path = REPO_ROOT / settings["paths"]["tanks_geojson"]
    if not inv_path.exists():
        print(f"ARRÊT: {inv_path} absent — lance scripts/02_build_inventory.py d'abord.")
        sys.exit(1)
    tanks = gpd.read_file(inv_path)
    bbox = [float(v) for v in tanks.total_bounds]

    start = settings["period"]["start"]
    end = settings["period"]["end"] or date.today().isoformat()

    try:
        catalog = open_catalog()
        items = search_scenes(catalog, bbox, start, end)
    except ImageryError as e:
        print(f"ARRÊT: {e}")
        sys.exit(1)

    meta = scene_metadata(items, bbox)
    by_id = {it.id: it for it in items}

    # même sélection qu'à l'étape 3 : couverture complète puis dédoublonnage
    meta = meta[meta["frame_coverage"] >= 0.999]
    meta = (
        meta.sort_values(["acquisition_date", "cloud_cover_pct"])
        .drop_duplicates("acquisition_date", keep="first")
    )
    meta = meta[meta["cloud_cover_pct"] < float(img["max_cloud_pct"])]
    meta = meta.sort_values("datetime").reset_index(drop=True)

    out_path = REPO_ROOT / settings["paths"]["data_processed"] / "tank_signals.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    frames: list[pd.DataFrame] = []
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        frames.append(existing)
        done = set(existing["item_id"].unique())
        print(f"Reprise: {len(done)} scène(s) déjà extraite(s).")

    todo = [r for _, r in meta.iterrows() if r["item_id"] not in done]
    print(f"Scènes à traiter: {len(todo)} / {len(meta)} exploitables\n")

    t0 = time.time()
    failures = 0
    for n, row in enumerate(todo, 1):
        item = by_id[row["item_id"]]
        try:
            df = extract_scene_signals(
                item, tanks, img["bands"], img["scl_exclude"],
                float(img["min_valid_pixel_frac"]),
            )
        # batch long sur réseau : une scène en échec ne doit pas tout arrêter,
        # mais chaque échec est tracé et compté, jamais avalé en silence.
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"  [{n}/{len(todo)}] {row['acquisition_date'].date()} ÉCHEC: "
                  f"{type(e).__name__}: {str(e)[:90]}")
            continue

        if len(df):
            frames.append(df)

        if n % 20 == 0 or n == len(todo):
            pd.concat(frames, ignore_index=True).to_parquet(out_path, index=False)
            rate = (time.time() - t0) / n
            left = rate * (len(todo) - n) / 60
            total = sum(len(f) for f in frames)
            print(f"  [{n}/{len(todo)}] {row['acquisition_date'].date()} | "
                  f"{total} lignes | {rate:.1f}s/scène | ~{left:.0f} min restantes")

    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(out_path, index=False)

    print(f"\nParquet écrit: {out_path}")
    print(f"  {len(out)} observations (cuve x scène)")
    print(f"  {out['scene_date'].nunique()} dates, {out['tank_id'].nunique()} cuves")
    print(f"  élévation solaire {out['sun_elevation_deg'].min():.1f}° -> "
          f"{out['sun_elevation_deg'].max():.1f}°")
    if failures:
        print(f"  {failures} scène(s) en échec")


if __name__ == "__main__":
    main()
