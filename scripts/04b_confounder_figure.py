"""Étape 4 — rendre le confondant solaire VISIBLE avant de le corriger.

Le DESIGN §8 l'exige explicitement : à remplissage constant, une cuve paraît
plus sombre quand le soleil est bas. Sans correction, le pipeline mesure la
saison et non le stock, et la corrélation obtenue avec l'EIA est fallacieuse.

Cette figure ne corrige rien. Elle montre l'ampleur du problème.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

BAND = "B04"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#d8d8d4"
SIGNAL_COLOR = "#2a78d6"
FIT_COLOR = "#e34948"
EIA_COLOR = "#1baf7a"


def main() -> None:
    settings = yaml.safe_load((REPO_ROOT / "config" / "settings.yaml").read_text())
    sig_path = REPO_ROOT / settings["paths"]["data_processed"] / "tank_signals.parquet"
    if not sig_path.exists():
        print(f"ARRÊT: {sig_path} absent — lance scripts/04_extract_signal.py d'abord.")
        sys.exit(1)

    sig = pd.read_parquet(sig_path)
    col = f"{BAND}_mean"
    sig = sig.dropna(subset=[col, "sun_elevation_deg"])
    print(f"{len(sig)} observations, {sig['tank_id'].nunique()} cuves, "
          f"{sig['scene_date'].nunique()} dates")

    # indice brut : moyenne par date, pondérée par la capacité de la cuve
    import geopandas as gpd
    tanks = gpd.read_file(REPO_ROOT / settings["paths"]["tanks_geojson"])
    caps = tanks.set_index("tank_id")["capacity_kbbl"]
    sig["w"] = sig["tank_id"].map(caps)

    daily = sig.groupby("scene_date").apply(
        lambda d: pd.Series({
            "index_raw": np.average(d[col], weights=d["w"]),
            "sun": d["sun_elevation_deg"].mean(),
            "n_tanks": len(d),
        }), include_groups=False,
    ).reset_index()

    fig, (ax_scatter, ax_time) = plt.subplots(1, 2, figsize=(13.5, 5.6))

    # --- gauche : le confondant, cuve par cuve ---
    sample = sig["tank_id"].drop_duplicates().sample(
        min(400, sig["tank_id"].nunique()), random_state=0
    )
    sub = sig[sig["tank_id"].isin(sample)]
    ax_scatter.scatter(sub["sun_elevation_deg"], sub[col], s=3, alpha=0.06,
                       color=SIGNAL_COLOR, edgecolors="none", rasterized=True)

    bins = np.arange(25, 75, 2.5)
    mid = 0.5 * (bins[:-1] + bins[1:])
    med = [sub.loc[sub["sun_elevation_deg"].between(a, b), col].median()
           for a, b in zip(bins[:-1], bins[1:])]
    ax_scatter.plot(mid, med, color=FIT_COLOR, linewidth=2.5, label="median by 2.5° bin")

    r = float(np.corrcoef(sig["sun_elevation_deg"], sig[col])[0, 1])
    slope = np.polyfit(sig["sun_elevation_deg"], sig[col], 1)[0]
    span = slope * (sig["sun_elevation_deg"].max() - sig["sun_elevation_deg"].min())
    ax_scatter.set_title(
        f"A — The confounder: tank brightness tracks the sun, not the stock\n"
        f"r = {r:+.2f} over {len(sig):,} tank-scene observations",
        fontsize=11.5, color=INK, loc="left", pad=8,
    )
    ax_scatter.set_xlabel("Solar elevation at acquisition (deg)", fontsize=10, color=INK_SOFT)
    ax_scatter.set_ylabel(f"{BAND} mean surface reflectance", fontsize=10, color=INK_SOFT)
    ax_scatter.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK_SOFT)
    ax_scatter.text(
        0.03, 0.97,
        f"Winter-to-summer swing: {span:+.3f} reflectance\n"
        "at a fill level we have no reason to think changed",
        transform=ax_scatter.transAxes, va="top", fontsize=9, color=FIT_COLOR,
    )

    # --- droite : la conséquence, indice brut contre EIA ---
    eia = pd.read_csv(REPO_ROOT / settings["paths"]["data_processed"] / "eia_cushing.csv",
                      parse_dates=["observation_date"])
    ax_time.plot(daily["scene_date"], daily["index_raw"], linewidth=0.9,
                 color=SIGNAL_COLOR, label=f"raw {BAND} index (capacity-weighted)")
    ax_time.set_ylabel(f"Raw {BAND} index", fontsize=10, color=SIGNAL_COLOR)
    ax_time.tick_params(axis="y", colors=SIGNAL_COLOR)

    ax_eia = ax_time.twinx()
    ax_eia.plot(eia["observation_date"], eia["stocks_kbbl"] / 1000,
                linewidth=1.6, color=EIA_COLOR, label="EIA stocks")
    ax_eia.set_ylabel("EIA Cushing stocks (Mbbl)", fontsize=10, color=EIA_COLOR)
    ax_eia.tick_params(axis="y", colors=EIA_COLOR)
    ax_eia.spines["top"].set_visible(False)

    ax_time.set_title(
        "B — Both series are seasonal. Any correlation here is suspect\n"
        "until the solar cycle is removed from the left panel.",
        fontsize=11.5, color=INK, loc="left", pad=8,
    )
    ax_time.set_xlabel("Date", fontsize=10, color=INK_SOFT)

    for ax in (ax_scatter, ax_time):
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(labelsize=9)

    fig.tight_layout()
    out = REPO_ROOT / settings["paths"]["outputs"] / "solar_confounder.png"
    fig.savefig(out, dpi=110, facecolor="white")
    plt.close(fig)
    print(f"Figure écrite: {out}")

    print(f"\nCorrélation réflectance <-> élévation solaire : {r:+.3f}")
    print(f"Amplitude hiver->été                          : {span:+.4f} de réflectance")
    print("Cette variation est du bruit saisonnier, pas du signal de stock.")


if __name__ == "__main__":
    main()
