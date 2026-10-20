"""Datos del caso RemesaSegura transcritos del enunciado del examen.

Cada función devuelve una tabla del enunciado como ``pandas.DataFrame``.
Las constantes de módulo contienen los parámetros operativos del caso.
"""

import pandas as pd

DAILY_SENDERS = 6_000
PREVALENCE = 0.002
EXPECTED_DAILY_CASES = 12

ANALYSTS = 8
ALERTS_PER_ANALYST = 5
REVIEW_COST_USD = 12
MISSED_CASE_COST_USD = 50_000

PILOT_DAILY_ALERTS = 250
PILOT_CONFIRMED_RATE = 0.04

AUC_RANDOM_SPLIT = 0.87
AUC_TEMPORAL_SPLIT = 0.81
AUC_TEMPORAL_DIFFUSION = 0.83


def thresholds() -> pd.DataFrame:
    """Devuelve la tabla de umbrales del Rol 1 (validación balanceada 50/50).

    Columnas: ``umbral``, ``recall``, ``fpr`` y ``precision_balanceada``.
    """
    return pd.DataFrame({
        "umbral": ["U1", "U2", "U3", "U4"],
        "recall": [0.90, 0.80, 0.65, 0.50],
        "fpr": [0.10, 0.04, 0.01, 0.005],
        "precision_balanceada": [0.90, 0.95, 0.98, 0.99],
    })


def training_losses() -> pd.DataFrame:
    """Devuelve los puntos de pérdida reportados para el entrenamiento del Rol 1.

    Columnas: ``epoca``, ``perdida_entrenamiento`` y ``perdida_validacion``.
    Los valores no reportados en el enunciado quedan como ``NaN``.
    """
    return pd.DataFrame({
        "epoca": [1, 6, 20],
        "perdida_entrenamiento": [0.42, None, 0.08],
        "perdida_validacion": [None, 0.21, 0.34],
    })


def transfer_strategies() -> pd.DataFrame:
    """Devuelve la tabla de estrategias de transferencia del Rol 2.

    Columnas: ``estrategia``, ``auc``, ``parametros_entrenables`` (fracción)
    y ``gpu_horas``.
    """
    return pd.DataFrame({
        "estrategia": [
            "Desde cero",
            "Extracción de características",
            "Ajuste fino de 2 bloques",
            "Ajuste fino completo",
            "LoRA rango 8",
        ],
        "auc": [0.78, 0.82, 0.87, 0.84, 0.86],
        "parametros_entrenables": [1.00, 0.005, 0.18, 1.00, 0.02],
        "gpu_horas": [6.0, 0.5, 1.5, 6.0, 1.0],
    })


def generators() -> pd.DataFrame:
    """Devuelve la tabla de generadores de datos sintéticos del Rol 4.

    Columnas: ``generador``, ``tipologias_cubiertas``, ``tipologias_totales``,
    ``costo_horas`` y ``observacion``.
    """
    return pd.DataFrame({
        "generador": ["GAN", "VAE", "Difusión"],
        "tipologias_cubiertas": [2, 5, 6],
        "tipologias_totales": [7, 7, 7],
        "costo_horas": [4, 2, 72],
        "observacion": [
            "Repite una sola forma de fraccionamiento",
            "Secuencias borrosas, promedia montos",
            "Mayor diversidad y nitidez",
        ],
    })


def narrative_audit() -> pd.DataFrame:
    """Devuelve la auditoría de narrativas del Rol 5.

    Columnas: ``configuracion``, ``con_hechos_inventados``, ``auditadas``
    y ``minutos_por_narrativa``. La redacción manual no tiene conteo de
    hechos inventados (``NaN``).
    """
    return pd.DataFrame({
        "configuracion": [
            "Sin contexto, T=0.7",
            "Con contexto, T=0.2",
            "Manual",
        ],
        "con_hechos_inventados": [6, 2, None],
        "auditadas": [40, 40, None],
        "minutos_por_narrativa": [1.5, 1.5, 6.0],
    })


def explainability_tests() -> pd.DataFrame:
    """Devuelve las pruebas de explicabilidad del Rol 6.

    Columnas: ``prueba``, ``metrica`` y ``valor``.
    """
    return pd.DataFrame({
        "prueba": [
            "SHAP, 100 secuencias de fondo",
            "LIME, 500 muestras",
            "LIME, 5,000 muestras",
            "Atención vs SHAP",
        ],
        "metrica": [
            "Error de eficiencia máximo (log-odds)",
            "Jaccard promedio top-5, 20 corridas",
            "Jaccard promedio top-5, 20 corridas",
            "Correlación de rangos",
        ],
        "valor": [3e-7, 0.45, 0.78, 0.35],
    })
