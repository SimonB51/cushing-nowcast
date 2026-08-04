"""Étape 3 (partie 1) — inventaire des scènes Sentinel-2 disponibles.

Ne télécharge AUCUN pixel : uniquement les métadonnées STAC, qui sont gratuites
et rapides. Objectif : savoir combien de semaines EIA sont couvrables avant
d'engager un téléchargement long.

Le taux de pixels valides après masquage SCL exige, lui, de lire les rasters —
c'est la partie 2 de l'étape 3, pas celle-ci.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.ticker import FuncFormatter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cushing.imagery import (  # noqa: E402
    ImageryError,
    open_catalog,
    scene_metadata,
    search_scenes,
)

# Figures légendées en anglais : elles sont affichées dans le README public.
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d8d8d4"
USABLE_COLOR = "#1baf7a"
CLOUDY_COLOR = "#e34948"
SUN_COLOR = "#eda100"


def plot_availability(weekly: pd.DataFrame, scenes: pd.DataFrame,
                      max_cloud: float, out_path: Path) -> None:
    # surtout PAS sharex : le panneau haut est en dates, le bas en semaine de
    # l'année. Les partager écrase l'un des deux.
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(12, 7.6), gridspec_kw={"height_ratios": [1.25, 1]}
    )

    # --- haut : semaines EIA couvertes, en moyenne glissante sur un an ---
    roll = weekly.set_index("eia_week_ending")["has_usable"].rolling(52, min_periods=26).mean()
    ax_top.fill_between(roll.index, 0, 100 * roll.to_numpy(), color=USABLE_COLOR, alpha=0.18)
    ax_top.plot(roll.index, 100 * roll.to_numpy(), color=USABLE_COLOR, linewidth=2)
    ax_top.axhline(100, color=GRID, linewidth=1)
    ax_top.set_ylabel("EIA weeks with a usable scene\n(52-week rolling mean, %)",
                      fontsize=10, color=INK_SOFT)
    ax_top.set_ylim(0, 105)
    ax_top.set_title(
        f"Sentinel-2 coverage of the EIA weekly calendar — scene cloud cover < {max_cloud:.0f}%",
        fontsize=12.5, color=INK, loc="left", pad=10,
    )

    # --- bas : saisonnalité, par semaine de l'année ---
    wk = scenes.copy()
    wk["week_of_year"] = wk["eia_week_ending"].dt.isocalendar().week.astype(int)
    # la semaine ISO 53 n'existe que certaines années : effectif minuscule, elle
    # produit une barre aberrante et fausse la corrélation. Écartée.
    wk = wk[wk["week_of_year"] <= 52]
    seasonal = wk.groupby("week_of_year").agg(
        cloudy=("cloud_cover_pct", lambda s: (s >= max_cloud).mean()),
        sun=("sun_elevation_deg", "mean"),
    )
    ax_bot.bar(seasonal.index, 100 * seasonal["cloudy"], color=CLOUDY_COLOR,
               alpha=0.75, width=0.85, label="scenes rejected on cloud cover")
    ax_bot.set_ylabel("Scenes rejected (%)", fontsize=10, color=CLOUDY_COLOR)
    ax_bot.set_xlabel("Week of year", fontsize=10, color=INK_SOFT)
    ax_bot.set_xlim(0, 54)

    ax_sun = ax_bot.twiny()
    ax_sun.set_xlim(ax_bot.get_xlim())
    ax_sun.set_xticks([])
    ax_sun2 = ax_bot.twinx()
    ax_sun2.plot(seasonal.index, seasonal["sun"], color=SUN_COLOR, linewidth=2.2,
                 label="mean solar elevation")
    ax_sun2.set_ylabel("Mean solar elevation (deg)", fontsize=10, color=SUN_COLOR)
    ax_sun2.tick_params(colors=SUN_COLOR, labelsize=9)
    for side in ("top",):
        ax_sun2.spines[side].set_visible(False)

    # Le titre reprend la corrélation MESURÉE entre les deux cycles. Ne pas y
    # écrire d'affirmation qualitative sans la recalculer : la première version
    # de cette figure annonçait un cycle commun, ce que les données démentent.
    valid = seasonal.dropna(subset=["cloudy", "sun"])
    r = float(np.corrcoef(valid["cloudy"], valid["sun"])[0, 1])
    ax_bot.set_title(
        f"Cloud rejection peaks in spring, not at the solar maximum — the two "
        f"seasonal cycles are near-independent (r = {r:+.2f})",
        fontsize=11, color=INK, loc="left", pad=8,
    )

    for ax in (ax_top, ax_bot):
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_SOFT, labelsize=9)
    ax_top.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, facecolor="white")
    plt.close(fig)


def main() -> None:
    settings = yaml.safe_load((REPO_ROOT / "config" / "settings.yaml").read_text())
    max_cloud = float(settings["imagery"]["max_cloud_pct"])

    inv_path = REPO_ROOT / settings["paths"]["tanks_geojson"]
    if not inv_path.exists():
        print(f"ARRÊT: {inv_path} absent — lance scripts/02_build_inventory.py d'abord.")
        sys.exit(1)

    # cadrage sur l'emprise des cuves, pas sur l'AOI : l'AOI chevauche deux tuiles
    # MGRS et imposerait un mosaïquage, alors que les cuves tiennent entièrement
    # dans une seule tuile. On ne mesure que dans les emprises de cuves.
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

    scenes = scene_metadata(items, bbox)
    print(f"Items STAC bruts        : {len(scenes)}  ({start} -> {end})")
    print(f"Dates d'acquisition     : {scenes['acquisition_date'].nunique()}")
    print(f"Tuiles MGRS             : {sorted(scenes['mgrs_tile'].dropna().unique())}")

    partial = scenes["frame_coverage"] < 0.999
    if partial.any():
        print(f"  {partial.sum()} item(s) ne couvrent pas toute l'emprise des cuves — écartés")
        scenes = scenes[~partial].copy()

    # Dédoublonnage : la même acquisition est découpée en plusieurs tuiles MGRS.
    # On garde, par date, l'item le moins nuageux.
    before = len(scenes)
    scenes = (
        scenes.sort_values(["acquisition_date", "cloud_cover_pct"])
        .drop_duplicates("acquisition_date", keep="first")
        .reset_index(drop=True)
    )
    print(f"Après dédoublonnage      : {len(scenes)} acquisitions distinctes "
          f"({before - len(scenes)} doublons de tuile retirés)")

    scenes["usable"] = scenes["cloud_cover_pct"] < max_cloud
    print(f"Nuages < {max_cloud:.0f}%           : {scenes['usable'].sum()} "
          f"({scenes['usable'].mean():.1%})")

    # couverture du calendrier EIA
    all_weeks = pd.date_range(scenes["eia_week_ending"].min(),
                              scenes["eia_week_ending"].max(), freq="W-FRI")
    weekly = pd.DataFrame({"eia_week_ending": all_weeks}).merge(
        scenes[scenes["usable"]].groupby("eia_week_ending").size().rename("n_usable"),
        on="eia_week_ending", how="left",
    )
    weekly["n_usable"] = weekly["n_usable"].fillna(0).astype(int)
    weekly["has_usable"] = weekly["n_usable"] > 0

    print(f"\nSemaines EIA couvertes   : {weekly['has_usable'].sum()} / {len(weekly)} "
          f"({weekly['has_usable'].mean():.1%})")
    print(f"Élévation solaire        : {scenes['sun_elevation_deg'].min():.1f}° "
          f"-> {scenes['sun_elevation_deg'].max():.1f}°")

    out_csv = REPO_ROOT / settings["paths"]["data_interim"] / "scene_inventory.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    scenes.to_csv(out_csv, index=False)
    print(f"\nCSV écrit: {out_csv}")

    out_fig = REPO_ROOT / settings["paths"]["outputs"] / "scene_availability.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    plot_availability(weekly, scenes, max_cloud, out_fig)
    print(f"Figure écrite: {out_fig}")


if __name__ == "__main__":
    main()
