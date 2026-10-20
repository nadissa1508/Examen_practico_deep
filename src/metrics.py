"""Métricas de clasificación bajo una prevalencia dada."""


def expected_counts(recall: float, fpr: float, n: int, prevalence: float) -> dict:
    """Calcula los conteos esperados de la matriz de confusión.

    Args:
        recall: Tasa de verdaderos positivos del umbral.
        fpr: Tasa de falsos positivos del umbral.
        n: Número total de casos evaluados.
        prevalence: Fracción de positivos en la población.

    Returns:
        Diccionario con ``tp``, ``fp``, ``fn``, ``tn`` y ``alertas`` (tp + fp).
    """
    positives = n * prevalence
    negatives = n - positives
    tp = recall * positives
    fp = fpr * negatives
    return {
        "tp": tp,
        "fp": fp,
        "fn": positives - tp,
        "tn": negatives - fp,
        "alertas": tp + fp,
    }


def precision_at_prevalence(recall: float, fpr: float, prevalence: float) -> float:
    """Devuelve la precisión esperada de un umbral a la prevalencia indicada."""
    tp = recall * prevalence
    fp = fpr * (1 - prevalence)
    return tp / (tp + fp)


def accuracy_of_constant_classifier(prevalence: float) -> float:
    """Devuelve la exactitud de un clasificador que siempre predice la clase negativa."""
    return 1 - prevalence


def hallucination_rate(fabricated: int, audited: int) -> float:
    """Devuelve la fracción de textos auditados que contienen hechos inventados."""
    return fabricated / audited
