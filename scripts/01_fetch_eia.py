"""Étape 1 — récupère la série EIA Cushing et produit le CSV + la figure."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from matplotlib.ticker import FuncFormatter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cushing.eia import EIA_SERIES_ID, EiaApiError, EiaApiKeyMissing, fetch_eia_cushing  # noqa: E402

# Les figures sont légendées en anglais : elles sont affichées dans le README public.
SERIES_COLOR = "#2a78d6"
INK = "#0b0b0b"
INK_SOFT = "#52514e"


def plot_series(df, out_path: Path) -> None:
    """Série EIA — une seule série, donc pas de légende : le titre la nomme."""
    fig, ax = plt.subplots(figsize=(10.5, 4.5))

    ax.plot(df["observation_date"], df["stocks_kbbl"], linewidth=1.8, color=SERIES_COLOR)

    ax.set_title(
        "Cushing, OK — Weekly Ending Stocks of Crude Oil",
        fontsize=13, color=INK, pad=12, loc="left",
    )
    ax.set_ylabel("Stocks (thousand barrels)", fontsize=10, color=INK_SOFT)
    ax.set_xlabel("Observation date (week ending Friday)", fontsize=10, color=INK_SOFT)

    ax.margins(x=0.01)
    ax.grid(axis="y", color="#d8d8d4", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d8d8d4")
    ax.tick_params(colors=INK_SOFT, labelsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    fig.text(
        0.008, 0.015,
        f"Source: U.S. EIA, series {EIA_SERIES_ID}. Unit: thousand barrels (kbbl).",
        fontsize=8, color=INK_SOFT,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(out_path, dpi=120, facecolor="white")
    plt.close(fig)


def main() -> None:
    settings = yaml.safe_load((REPO_ROOT / "config" / "settings.yaml").read_text())

    start = settings["period"]["start"]
    end = settings["period"]["end"] or date.today().isoformat()

    try:
        df = fetch_eia_cushing(start, end)
    except EiaApiKeyMissing as e:
        print(f"ARRÊT: {e}")
        sys.exit(1)
    except EiaApiError as e:
        print(f"ARRÊT: échec de récupération EIA: {e}")
        sys.exit(1)

    out_csv = REPO_ROOT / settings["paths"]["data_processed"] / "eia_cushing.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"CSV écrit: {out_csv} ({len(df)} lignes, {df['observation_date'].min().date()} -> {df['observation_date'].max().date()})")

    out_fig_dir = REPO_ROOT / settings["paths"]["outputs"]
    out_fig_dir.mkdir(parents=True, exist_ok=True)
    out_fig = out_fig_dir / "eia_cushing_series.png"

    plot_series(df, out_fig)
    print(f"Figure écrite: {out_fig}")


if __name__ == "__main__":
    main()
