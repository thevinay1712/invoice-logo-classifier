"""Evaluation module: metrics calculation, confusion matrix plotting, and threshold calibration."""

from invoice_logo_classifier.evaluation.evaluate import (
    EvaluationReport,
    evaluate_model,
    evaluate_predictions,
    generate_evaluation_summary_md,
    plot_confusion_matrix,
    sweep_confidence_thresholds,
)

__all__ = [
    "EvaluationReport",
    "evaluate_model",
    "evaluate_predictions",
    "generate_evaluation_summary_md",
    "plot_confusion_matrix",
    "sweep_confidence_thresholds",
]
