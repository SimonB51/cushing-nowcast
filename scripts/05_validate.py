"""Étapes 5-6 — calibration hors-échantillon et validation honnête.

Respecte les règles d'intégrité temporelle du DESIGN §9 :
  - split strictement chronologique (TimeSeriesSplit), jamais de shuffle ;
  - une image du samedi ne peut pas servir à prédire le vendredi qui précède ;
  - la régression de nuisance solaire est ajustée sur l'ENTRAÎNEMENT seul ;
  - aucun paramètre choisi en regardant le test.

Rapporte les métriques du DESIGN §10, y compris la comparaison au baseline naïf
« la variation de cette semaine est nulle ».
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

BAND = "B04"
INK, INK_SOFT, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
POINT, NAIVE, PHYS = "#2a78d6", "#52514e", "#eb6834"


def build_weekly(settings) -> pd.DataFrame:
    """Indice hebdomadaire pondéré par capacité, apparié à la série EIA."""
    proc = REPO_ROOT / settings["paths"]["data_processed"]
    sig = pd.read_parquet(proc / "tank_signals.parquet")
    tanks = gpd.read_file(REPO_ROOT / settings["paths"]["tanks_geojson"]).set_index("tank_id")
    sig["cap"] = sig["tank_id"].map(tanks["capacity_kbbl"])

    daily = (
        sig.groupby("scene_date")
        .apply(lambda x: pd.Series({
            "index_raw": np.average(x[f"{BAND}_mean"], weights=x["cap"]),
            "sun": x["sun_elevation_deg"].mean(),
            "n_tanks": len(x),
        }), include_groups=False)
        .reset_index()
    )
    # règle d'intégrité : rattachement au premier vendredi >= acquisition
    daily["week"] = daily["scene_date"] + pd.to_timedelta(
        (4 - daily["scene_date"].dt.weekday) % 7, unit="D"
    )
    weekly = daily.groupby("week").agg(
        index_raw=("index_raw", "mean"), sun=("sun", "mean"), n_scenes=("index_raw", "size")
    ).reset_index()

    eia = pd.read_csv(proc / "eia_cushing.csv", parse_dates=["observation_date"])
    m = weekly.merge(eia, left_on="week", right_on="observation_date", how="inner")
    return m.sort_values("week").reset_index(drop=True)


def evaluate(m: pd.DataFrame, n_splits: int = 5) -> tuple[pd.DataFrame, dict]:
    """Prédit la VARIATION hebdomadaire des stocks, hors-échantillon."""
    m = m.copy()
    m["d_eia"] = m["stocks_kbbl"].diff()
    m["d_index"] = m["index_raw"].diff()
    consecutive = m["week"].diff().dt.days == 7
    m = m[consecutive & m["d_eia"].notna() & m["d_index"].notna()].reset_index(drop=True)

    preds = []
    for tr, te in TimeSeriesSplit(n_splits=n_splits).split(m):
        train, test = m.iloc[tr], m.iloc[te]

        # régression de nuisance ajustée sur l'ENTRAÎNEMENT seul
        Xs = np.sin(np.radians(train["sun"])).to_numpy().reshape(-1, 1)
        nuis = Ridge(alpha=1.0).fit(Xs, train["d_index"])

        def resid(df):
            s = np.sin(np.radians(df["sun"])).to_numpy().reshape(-1, 1)
            return df["d_index"].to_numpy() - nuis.predict(s)

        mu, sd = resid(train).mean(), resid(train).std()
        sd = sd if sd > 0 else 1.0
        model = Ridge(alpha=1.0).fit(((resid(train) - mu) / sd).reshape(-1, 1), train["d_eia"])
        yhat = model.predict(((resid(test) - mu) / sd).reshape(-1, 1))

        preds.append(pd.DataFrame({
            "week": test["week"].to_numpy(),
            "actual": test["d_eia"].to_numpy(),
            "pred": yhat,
        }))

    out = pd.concat(preds, ignore_index=True)
    naive = np.zeros(len(out))                      # « variation nulle »
    metrics = {
        "n_test_weeks": len(out),
        "corr_weekly_change": float(np.corrcoef(out["pred"], out["actual"])[0, 1]),
        "directional_accuracy": float((np.sign(out["pred"]) == np.sign(out["actual"])).mean()),
        "mae_mbbl": float(np.abs(out["pred"] - out["actual"]).mean() / 1000),
        "mae_naive_mbbl": float(np.abs(naive - out["actual"]).mean() / 1000),
    }
    metrics["mae_vs_naive"] = metrics["mae_mbbl"] / metrics["mae_naive_mbbl"]
    se = np.sqrt(0.25 / len(out))
    metrics["dir_acc_z"] = (metrics["directional_accuracy"] - 0.5) / se
    return out, metrics


def physical_limit(settings) -> dict:
    """Amplitude du signal recherché, en fraction de pixel."""
    proc = REPO_ROOT / settings["paths"]["data_processed"]
    eia = pd.read_csv(proc / "eia_cushing.csv")
    tanks = gpd.read_file(REPO_ROOT / settings["paths"]["tanks_geojson"])
    H = float(settings["tanks"]["height_m_flat"])
    d_week = float(eia["stocks_kbbl"].diff().abs().median())
    frac = d_week / float(tanks["capacity_kbbl"].sum())
    wall_m = frac * H
    return {
        "median_weekly_change_kbbl": d_week,
        "fill_change_pct": 100 * frac,
        "wall_exposed_cm": 100 * wall_m,
        "shadow_px_at_45deg": wall_m / 10.0,
    }


def main() -> None:
    settings = yaml.safe_load((REPO_ROOT / "config" / "settings.yaml").read_text())
    if not (REPO_ROOT / settings["paths"]["data_processed"] / "tank_signals.parquet").exists():
        print("ARRÊT: tank_signals.parquet absent — lance scripts/04_extract_signal.py.")
        sys.exit(1)

    m = build_weekly(settings)
    out, met = evaluate(m)
    phys = physical_limit(settings)

    print(f"Semaines appariées: {len(m)} | testées hors-échantillon: {met['n_test_weeks']}\n")
    print("--- Métriques hors-échantillon (DESIGN §10) ---")
    print(f"  corr_weekly_change   : {met['corr_weekly_change']:+.3f}   <- LA métrique")
    print(f"  directional_accuracy : {met['directional_accuracy']:.1%} "
          f"(z={met['dir_acc_z']:+.2f}, hasard=50%)")
    print(f"  mae_mbbl             : {met['mae_mbbl']:.3f}")
    print(f"  mae_naive_mbbl       : {met['mae_naive_mbbl']:.3f}  (variation nulle)")
    print(f"  mae_vs_naive         : {met['mae_vs_naive']:.3f}  "
          f"({'bat' if met['mae_vs_naive'] < 1 else 'NE BAT PAS'} le baseline naïf)")

    print("\n--- Pourquoi : amplitude du signal recherché ---")
    print(f"  variation hebdo médiane EIA : {phys['median_weekly_change_kbbl']:,.0f} kbbl")
    print(f"  soit un remplissage de      : {phys['fill_change_pct']:.2f} % par semaine")
    print(f"  paroi découverte            : {phys['wall_exposed_cm']:.1f} cm")
    print(f"  ombre à 45°                 : {phys['shadow_px_at_45deg']:.4f} pixel Sentinel-2")

    # --- figure ---
    fig, (ax_sc, ax_ph) = plt.subplots(1, 2, figsize=(13, 5.6))

    lim = np.abs(np.concatenate([out["actual"], out["pred"]])).max() / 1000 * 1.05
    ax_sc.axhline(0, color=NAIVE, linewidth=1.2, linestyle=(0, (4, 4)))
    ax_sc.scatter(out["actual"] / 1000, out["pred"] / 1000, s=26, alpha=0.6,
                  color=POINT, edgecolors="white", linewidth=0.5)
    ax_sc.set(xlim=(-lim, lim), ylim=(-lim, lim))
    ax_sc.set_xlabel("Actual weekly change (Mbbl)", fontsize=10, color=INK_SOFT)
    ax_sc.set_ylabel("Predicted weekly change (Mbbl)", fontsize=10, color=INK_SOFT)
    ax_sc.set_title(
        f"A — Out-of-sample: no relationship\n"
        f"r = {met['corr_weekly_change']:+.3f}, directional accuracy "
        f"{met['directional_accuracy']:.0%} (z = {met['dir_acc_z']:+.1f})",
        fontsize=11.5, color=INK, loc="left", pad=8,
    )
    ax_sc.text(0.03, 0.04, "dashed line = naive forecast (no change)",
               transform=ax_sc.transAxes, fontsize=8.5, color=NAIVE)

    labels = ["Median weekly\nEIA change", "10 points\nof fill", "One Sentinel-2\npixel"]
    vals = [phys["shadow_px_at_45deg"], 0.10 * float(settings["tanks"]["height_m_flat"]) / 10, 1.0]
    bars = ax_ph.barh(labels, vals, color=[PHYS, "#eda100", GRID], height=0.55)
    ax_ph.set_xscale("log")
    ax_ph.set_xlabel("Shadow displacement at 45° sun (Sentinel-2 pixels, log scale)",
                     fontsize=10, color=INK_SOFT)
    for b, v in zip(bars, vals):
        ax_ph.text(v * 1.15, b.get_y() + b.get_height() / 2,
                   f"{v:.3f} px" if v < 1 else f"{v:.0f} px",
                   va="center", fontsize=9.5, color=INK)
    ax_ph.set_title(
        "B — Why: the weekly signal is 1% of a pixel\n"
        "Geometric and radiometric methods both need it resolved",
        fontsize=11.5, color=INK, loc="left", pad=8,
    )

    for ax in (ax_sc, ax_ph):
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK_SOFT, labelsize=9)

    fig.tight_layout()
    out_fig = REPO_ROOT / settings["paths"]["outputs"] / "validation.png"
    fig.savefig(out_fig, dpi=110, facecolor="white")
    plt.close(fig)
    print(f"\nFigure écrite: {out_fig}")

    pd.DataFrame([{**met, **phys}]).to_csv(
        REPO_ROOT / settings["paths"]["outputs"] / "validation_metrics.csv", index=False
    )


if __name__ == "__main__":
    main()
