"""Vérifie l'intégrité de la série EIA produite à l'étape 1."""

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = REPO_ROOT / "data" / "processed" / "eia_cushing.csv"

MAX_GAP_DAYS = 8


@pytest.fixture()
def df() -> pd.DataFrame:
    if not CSV_PATH.exists():
        pytest.skip(f"{CSV_PATH} absent — lance scripts/01_fetch_eia.py d'abord.")
    frame = pd.read_csv(CSV_PATH, parse_dates=["observation_date", "publication_date"])
    return frame.sort_values("observation_date").reset_index(drop=True)


def test_no_duplicate_dates(df: pd.DataFrame) -> None:
    dup = df["observation_date"][df["observation_date"].duplicated()]
    assert dup.empty, f"Dates dupliquées: {dup.tolist()}"


def test_no_gap_over_threshold(df: pd.DataFrame) -> None:
    gaps = df["observation_date"].diff().dropna()
    max_gap = gaps.max()
    bad = gaps[gaps > pd.Timedelta(days=MAX_GAP_DAYS)]
    assert bad.empty, (
        f"Trou(s) de plus de {MAX_GAP_DAYS} jours dans la série EIA "
        f"(max observé: {max_gap}): indices {bad.index.tolist()}"
    )


def test_publication_lag_is_five_days(df: pd.DataFrame) -> None:
    lag = (df["publication_date"] - df["observation_date"]).dt.days
    assert (lag == 5).all(), "publication_date doit être observation_date + 5 jours"
