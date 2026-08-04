"""Verrous sur les deux corrections radiométriques critiques de l'étape 4.

Aucun accès réseau : ces tests portent sur la logique, pas sur les données.
Les deux défauts qu'ils couvrent sont silencieux — sans eux, le pipeline
produirait des nombres plausibles et faux.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cushing.signal import (  # noqa: E402
    BOA_ADD_OFFSET,
    RING_INNER_M,
    RING_OUTER_M,
    SignalError,
    boa_offset,
)


@pytest.mark.parametrize("baseline", ["02.12", "03.00", "03.01"])
def test_no_offset_before_baseline_0400(baseline: str) -> None:
    assert boa_offset(baseline) == 0.0


@pytest.mark.parametrize("baseline", ["04.00", "05.09", "05.11", "05.12"])
def test_offset_applied_from_baseline_0400(baseline: str) -> None:
    """Baseline >= 04.00 : BOA_ADD_OFFSET = -1000.

    Sans cette correction la série présente une marche artificielle de +0,1 en
    réflectance en janvier 2022, que la calibration lirait comme un changement
    de niveau des stocks. Mesuré sur ce jeu de données : médiane B04 de 897/766/709
    DN avant bascule, 2136/1710/1952 après.
    """
    assert boa_offset(baseline) == BOA_ADD_OFFSET == -1000.0


def test_missing_baseline_raises_rather_than_guessing() -> None:
    """Deviner l'offset vaudrait pire que s'arrêter : l'erreur est invisible."""
    with pytest.raises(SignalError, match="processing_baseline"):
        boa_offset(None)


def test_baseline_comparison_is_numeric_not_lexicographic() -> None:
    """La comparaison doit être numérique.

    Une comparaison de chaînes donnerait '3.00' > '04.00' (car '3' > '0') et
    appliquerait l'offset à un produit ANTÉRIEUR à la bascule, décalant de -0,1
    une partie de la série. Ce test attrape ce retour en arrière.
    """
    assert boa_offset("3.00") == 0.0, "comparaison lexicographique détectée"
    assert boa_offset("09.99") == BOA_ADD_OFFSET
    assert boa_offset("10.00") == BOA_ADD_OFFSET
    assert boa_offset("4.00") == BOA_ADD_OFFSET


def test_unreadable_baseline_raises() -> None:
    with pytest.raises(SignalError, match="illisible"):
        boa_offset("not-a-version")


def test_ring_is_outside_tank_and_ordered() -> None:
    """L'anneau de jugement nuage doit être hors des cuves, et non dégénéré.

    Il sert à décider de la nébulosité SANS regarder la brillance de la cuve :
    SCL classe les toits brillants comme nuages (86 % des pixels de cuve sur la
    scène 2021-10-11, contre 1,4 % à 150 m). Juger sur la cuve elle-même
    écarterait les cuves les plus brillantes, créant un biais corrélé à la
    grandeur mesurée.
    """
    assert 0 < RING_INNER_M < RING_OUTER_M
    assert RING_INNER_M >= 50.0, (
        "L'anneau doit commencer au-delà de la zone de contamination des bords "
        "(30 % de faux nuage mesurés à 20 m des emprises)."
    )
