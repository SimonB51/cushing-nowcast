"""Étape 1 — récupération de la série EIA de vérité terrain.

Série W_EPC0_SAX_YCUOK_MBBL : Weekly Cushing, OK Ending Stocks of Crude Oil
(milliers de barils). Publiée le mercredi, référence au vendredi précédent.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

EIA_SERIES_ID = "W_EPC0_SAX_YCUOK_MBBL"
# La route raccourci /v2/seriesid/{id} renvoie 404 "not valid" pour cette
# série (testé le 2026-07-27) alors que la route complète fonctionne. On
# utilise donc la route explicite du dataset petroleum/stoc/wstk avec un
# facet sur le series id.
EIA_DATA_URL = "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
PUBLICATION_LAG_DAYS = 5
PAGE_LENGTH = 5000


class EiaApiKeyMissing(RuntimeError):
    pass


class EiaApiError(RuntimeError):
    pass


def _get_api_key() -> str:
    key = os.environ.get("EIA_API_KEY")
    if not key:
        raise EiaApiKeyMissing(
            "EIA_API_KEY n'est pas défini. Copie .env.example en .env et "
            "renseigne ta clé (gratuite sur https://www.eia.gov/opendata/), "
            "ou exporte EIA_API_KEY dans l'environnement avant de relancer."
        )
    return key


def _redact(text: str, api_key: str) -> str:
    """Masque la clé API dans un texte destiné à être affiché.

    L'EIA peut renvoyer la requête dans le corps d'une erreur, et urllib3 met
    l'URL complète — donc la query string — dans ses messages d'exception.
    """
    return text.replace(api_key, "***REDACTED***") if api_key else text


def _fetch_page(api_key: str, start: str, end: str, offset: int) -> list[dict]:
    params = [
        ("api_key", api_key),
        ("frequency", "weekly"),
        ("data[0]", "value"),
        ("facets[series][]", EIA_SERIES_ID),
        ("start", start),
        ("end", end),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "asc"),
        ("offset", offset),
        ("length", PAGE_LENGTH),
    ]
    try:
        resp = requests.get(EIA_DATA_URL, params=params, timeout=30)
    except requests.RequestException as e:
        # `from None` est délibéré, pas une négligence : le message d'urllib3
        # contient l'URL complète, clé API comprise. Chaîner l'exception la
        # réafficherait dans la traceback, ce que cette branche vise à éviter.
        raise EiaApiError(
            f"Échec réseau vers l'API EIA (offset={offset}): {type(e).__name__}. "
            "Détail masqué: il contiendrait la clé API. Vérifie ta connexion réseau."
        ) from None

    if resp.status_code != 200:
        raise EiaApiError(
            f"EIA API a répondu {resp.status_code} pour offset={offset}: "
            f"{_redact(resp.text[:500], api_key)}"
        )
    payload = resp.json()
    if "response" not in payload or "data" not in payload["response"]:
        raise EiaApiError(f"Réponse EIA inattendue: {payload}")
    return payload["response"]["data"]


def fetch_eia_cushing(start: str, end: str) -> pd.DataFrame:
    """
    Retourne un DataFrame indexé par la DATE D'OBSERVATION (le vendredi),
    avec une colonne séparée pour la date de publication.

    Colonnes: observation_date, publication_date, stocks_kbbl
    """
    api_key = _get_api_key()

    rows: list[dict] = []
    offset = 0
    while True:
        page = _fetch_page(api_key, start, end, offset)
        if not page:
            break
        rows.extend(page)
        if len(page) < PAGE_LENGTH:
            break
        offset += PAGE_LENGTH

    if not rows:
        raise EiaApiError(
            f"Aucune donnée retournée par l'EIA pour la série {EIA_SERIES_ID} "
            f"entre {start} et {end}."
        )

    df = pd.DataFrame(rows)
    if "period" not in df.columns or "value" not in df.columns:
        raise EiaApiError(f"Colonnes inattendues dans la réponse EIA: {df.columns.tolist()}")

    df = df.rename(columns={"period": "observation_date", "value": "stocks_kbbl"})
    df["observation_date"] = pd.to_datetime(df["observation_date"])
    df["stocks_kbbl"] = pd.to_numeric(df["stocks_kbbl"], errors="coerce")

    if df["stocks_kbbl"].isna().any():
        bad = df.loc[df["stocks_kbbl"].isna(), "observation_date"].dt.strftime("%Y-%m-%d").tolist()
        raise EiaApiError(f"Valeurs non numériques renvoyées par l'EIA pour: {bad}")

    df = df[["observation_date", "stocks_kbbl"]].sort_values("observation_date").reset_index(drop=True)

    dup = df["observation_date"][df["observation_date"].duplicated()]
    if not dup.empty:
        raise EiaApiError(f"Dates dupliquées dans la série EIA: {dup.dt.strftime('%Y-%m-%d').tolist()}")

    df["publication_date"] = df["observation_date"] + pd.Timedelta(days=PUBLICATION_LAG_DAYS)

    non_wednesday = df[df["publication_date"].dt.dayofweek != 2]
    if not non_wednesday.empty:
        for _, row in non_wednesday.iterrows():
            logger.warning(
                "publication_date %s (obs %s) ne tombe pas un mercredi — "
                "probablement un décalage lié à un jour férié.",
                row["publication_date"].strftime("%Y-%m-%d"),
                row["observation_date"].strftime("%Y-%m-%d"),
            )

    return df
