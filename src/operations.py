"""Cálculos de capacidad y costo operativo del equipo de cumplimiento."""

import pandas as pd

from src.metrics import expected_counts


def team_capacity(analysts: int, alerts_per_analyst: int) -> int:
    """Devuelve el número de alertas que el equipo puede revisar por día."""
    return analysts * alerts_per_analyst


def review_cost(alerts: float, cost_per_review: float) -> float:
    """Devuelve el costo de revisar ``alerts`` alertas."""
    return alerts * cost_per_review


def detected_cases(alerts: float, confirmed_rate: float) -> float:
    """Devuelve el número de casos confirmados dentro de ``alerts`` alertas."""
    return alerts * confirmed_rate


def threshold_table(
    thresholds: pd.DataFrame,
    n: int,
    prevalence: float,
    cost_per_review: float,
    missed_case_cost: float,
) -> pd.DataFrame:
    """Proyecta cada umbral a la población diaria real.

    Args:
        thresholds: Tabla con columnas ``umbral``, ``recall`` y ``fpr``.
        n: Remitentes evaluados por día.
        prevalence: Fracción de remitentes con caso confirmado.
        cost_per_review: Costo en USD de revisar una alerta.
        missed_case_cost: Costo estimado en USD de un caso no detectado.

    Returns:
        DataFrame con alertas diarias, casos detectados y no detectados,
        precisión esperada, costo de revisión, costo de no detección y
        costo total por umbral.
    """
    rows = []
    for row in thresholds.itertuples():
        counts = expected_counts(row.recall, row.fpr, n, prevalence)
        review = review_cost(counts["alertas"], cost_per_review)
        missed = counts["fn"] * missed_case_cost
        rows.append({
            "umbral": row.umbral,
            "alertas_diarias": counts["alertas"],
            "detectados": counts["tp"],
            "no_detectados": counts["fn"],
            "precision_esperada": counts["tp"] / counts["alertas"],
            "costo_revision_usd": review,
            "costo_no_deteccion_usd": missed,
            "costo_total_usd": review + missed,
        })
    return pd.DataFrame(rows)
