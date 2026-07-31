"""Étape 2 — figure de l'inventaire (definition of done du DESIGN §6).

Deux panneaux :
  - l'inventaire géolocalisé, chaque cuve colorée par sa capacité ;
  - la courbe de capacité cumulée, qui montre à quel point la distribution est
    plate — c'est le résultat analytique de l'étape, pas une décoration.

Distinct de 02b, qui juge la COUVERTURE des empreintes OSM brutes. Ici on
représente l'inventaire construit, avec rayons et capacités calculés.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cushing.imagery import (  # noqa: E402
    ImageryError,
    load_rgb_clip,
    open_catalog,
    pick_best_scene,
    recent_window,
)
from cushing.inventory import assign_height_m  # noqa: E402

# Figures légendées en anglais : elles sont affichées dans le README public.
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d8d8d4"
FLAT_COLOR = "#2a78d6"
BYRADIUS_COLOR = "#eb6834"
MARGIN_M = 600.0


def _panel_map(ax, inv_proj, rgb_clip, item, coverage) -> None:
    rgb_clip.plot.imshow(ax=ax)
    ax.set_title("")  # xarray pose son propre titre ("spatial_ref = 0")

    caps = inv_proj["capacity_kbbl"].to_numpy()
    inv_proj.plot(
        ax=ax, column="capacity_kbbl", cmap="Oranges", vmin=caps.min(), vmax=caps.max(),
        edgecolor="white", linewidth=0.4, alpha=0.92, legend=True,
        legend_kwds={
            "label": "Tank shell capacity (thousand barrels)",
            "orientation": "horizontal", "pad": 0.01, "shrink": 0.9, "aspect": 26,
            "location": "bottom",
        },
    )
    ax.set_title(
        f"A — Tank inventory: {len(inv_proj)} tanks, Sentinel-2 {item.datetime.date()}",
        fontsize=11, color=INK, loc="left", pad=8,
    )
    ax.set_axis_off()
    if coverage < 0.99:
        ax.text(
            0.01, 0.01, f"scene covers {coverage:.0%} of AOI",
            transform=ax.transAxes, fontsize=7, color="white", va="bottom",
        )


def _panel_concentration(ax, radii, tanks_cfg) -> None:
    """Capacité cumulée par rang décroissant, sous les deux modèles de hauteur."""
    for model, colour in (("flat", FLAT_COLOR), ("by_radius", BYRADIUS_COLOR)):
        h = assign_height_m(radii, {**tanks_cfg, "height_model": model})
        cap = np.sort(np.pi * radii**2 * h)[::-1]
        share = np.cumsum(cap) / cap.sum()
        x = 100 * np.arange(1, len(cap) + 1) / len(cap)
        ax.plot(x, 100 * share, linewidth=2, color=colour, label=model)

        # marqueur du top décile : la grandeur qui pilote la pondération
        i = max(1, int(round(0.10 * len(cap)))) - 1
        ax.plot([x[i]], [100 * share[i]], "o", color=colour, markersize=8,
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        ax.annotate(
            f"top 10% = {100*share[i]:.1f}%",
            xy=(x[i], 100 * share[i]), xytext=(14, -6 if model == "flat" else 10),
            textcoords="offset points", fontsize=9, color=colour,
        )

    ax.plot([0, 100], [0, 100], linestyle=(0, (4, 4)), linewidth=1,
            color=INK_SOFT, alpha=0.55)
    ax.annotate("perfectly even distribution", xy=(62, 62), xytext=(0, -16),
                textcoords="offset points", fontsize=8, color=INK_SOFT, rotation=32)

    ax.set_title(
        "B — Capacity concentration: the index is mutualised, not driven by a few giants",
        fontsize=11, color=INK, loc="left", pad=8,
    )
    ax.set_xlabel("Tanks, ranked largest first (%)", fontsize=10, color=INK_SOFT)
    ax.set_ylabel("Cumulative share of capacity (%)", fontsize=10, color=INK_SOFT)
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    ax.legend(
        handles=[Line2D([], [], color=c, linewidth=2, label=f"height_model = {m}")
                 for m, c in (("flat", FLAT_COLOR), ("by_radius", BYRADIUS_COLOR))],
        loc="lower right", frameon=False, fontsize=9, labelcolor=INK_SOFT,
    )


def main() -> None:
    settings = yaml.safe_load((REPO_ROOT / "config" / "settings.yaml").read_text())
    tanks_cfg = settings["tanks"]

    inv_path = REPO_ROOT / settings["paths"]["tanks_geojson"]
    if not inv_path.exists():
        print(f"ARRÊT: {inv_path} absent — lance scripts/02_build_inventory.py d'abord.")
        sys.exit(1)
    inv = gpd.read_file(inv_path)

    # cadrage sur les parcs de cuves, pas sur l'AOI : l'AOI est très majoritairement
    # de la campagne vide et noierait l'inventaire.
    utm = inv.estimate_utm_crs()
    inv_proj = inv.to_crs(utm)
    minx, miny, maxx, maxy = inv_proj.total_bounds
    frame = gpd.GeoSeries.from_wkt(
        [f"POLYGON(({minx-MARGIN_M} {miny-MARGIN_M}, {maxx+MARGIN_M} {miny-MARGIN_M}, "
         f"{maxx+MARGIN_M} {maxy+MARGIN_M}, {minx-MARGIN_M} {maxy+MARGIN_M}, "
         f"{minx-MARGIN_M} {miny-MARGIN_M}))"],
        crs=utm,
    ).to_crs("EPSG:4326")

    try:
        catalog = open_catalog()
        item, coverage = pick_best_scene(catalog, list(frame.total_bounds), recent_window())
        rgb_clip, raster_crs = load_rgb_clip(item, list(frame.total_bounds))
    except ImageryError as e:
        print(f"ARRÊT: {e}")
        sys.exit(1)

    print(f"Scène: {item.id} ({item.datetime.date()}), couverture={coverage:.1%}")

    # l'emprise des cuves est nettement plus haute que large : on donne au
    # panneau carte une colonne étroite plutôt que de l'étirer à vide.
    fig, (ax_map, ax_conc) = plt.subplots(
        1, 2, figsize=(13.0, 7.6), gridspec_kw={"width_ratios": [0.78, 1.22]}
    )
    _panel_map(ax_map, inv_proj.to_crs(raster_crs), rgb_clip, item, coverage)
    _panel_concentration(ax_conc, inv["radius_m"].to_numpy(), tanks_cfg)

    total = inv["capacity_kbbl"].sum() / 1000
    ref = float(tanks_cfg["capacity_reference_shell_mbbl"])
    fig.text(
        0.5, 0.015,
        f"{len(inv)} OpenStreetMap tank footprints, Cushing OK. Total shell capacity "
        f"{total:.1f} Mbbl under height_model={tanks_cfg['height_model']}, versus "
        f"{ref:.1f} Mbbl reported by the EIA ({tanks_cfg['capacity_reference_asof']}) "
        f"— ratio {total/ref:.2f}.",
        fontsize=8, color=INK_SOFT, ha="center",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out_path = REPO_ROOT / settings["paths"]["outputs"] / "tank_inventory.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, facecolor="white")
    plt.close(fig)
    print(f"Figure écrite: {out_path}")


if __name__ == "__main__":
    main()
