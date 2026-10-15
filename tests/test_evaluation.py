"""Unit tests for evaluation metrics, confusion matrix plotting, threshold calibration, and markdown summary generation."""

import numpy as np
import pytest

from invoice_logo_classifier.evaluation import (
    EvaluationReport,
    evaluate_predictions,
    generate_evaluation_summary_md,
    plot_confusion_matrix,
    sweep_confidence_thresholds,
)


def test_evaluate_predictions_metrics_and_confused_pairs():
    """Verify evaluate_predictions correctly computes metrics and detects confused class pairs."""
    class_list = ["class_A", "class_B", "class_C"]
    y_true = [0, 0, 0, 1, 1, 2]
    y_pred = [0, 0, 1, 1, 1, 2]  # One sample of class_A misclassified as class_B

    report = evaluate_predictions(y_true, y_pred, class_list)

    assert isinstance(report, EvaluationReport)
    assert report.overall_accuracy == pytest.approx(83.33, abs=0.1)
    assert len(report.confused_pairs) == 1
    assert report.confused_pairs[0]["true_class"] == "class_A"
    assert report.confused_pairs[0]["predicted_class"] == "class_B"
    assert report.confused_pairs[0]["count"] == 1


def test_sweep_confidence_thresholds():
    """Verify sweep_confidence_thresholds calculates auto-coverage and auto-accuracy across confidence levels."""
    y_true = [0, 1, 0, 1]
    # Softmax probabilities: 2 high confidence, 2 low confidence
    probs = np.array(
        [
            [0.95, 0.05],  # Confident correct (class 0)
            [0.10, 0.90],  # Confident correct (class 1)
            [0.55, 0.45],  # Low confidence correct (class 0)
            [0.40, 0.60],  # Low confidence correct (class 1)
        ]
    )

    sweep = sweep_confidence_thresholds(y_true, probs, thresholds=[0.50, 0.85])

    assert len(sweep) == 2
    # At threshold 0.85, only 2 samples meet or exceed 0.85 -> 50% coverage
    t_85 = next(s for s in sweep if s["threshold"] == 0.85)
    assert t_85["auto_coverage_pct"] == 50.0
    assert t_85["auto_accuracy_pct"] == 100.0
    assert t_85["fallback_rate_pct"] == 50.0


def test_plot_confusion_matrix(tmp_path):
    """Verify plot_confusion_matrix creates a valid PNG file on disk."""
    cm = [[5, 1], [0, 4]]
    class_list = ["class_A", "class_B"]
    out_png = tmp_path / "test_cm.png"

    saved_path = plot_confusion_matrix(cm, class_list, output_path=out_png)

    assert saved_path.exists()
    assert saved_path.stat().st_size > 0


def test_generate_evaluation_summary_md(tmp_path):
    """Verify generate_evaluation_summary_md produces a valid summary markdown file containing explicit test split note."""
    class_list = ["class_A", "class_B"]
    report = EvaluationReport(
        overall_accuracy=90.0,
        macro_precision=90.0,
        macro_recall=90.0,
        macro_f1=90.0,
        per_class_metrics={},
        confusion_matrix=[[9, 1], [1, 9]],
        class_list=class_list,
        confused_pairs=[{"true_class": "class_A", "predicted_class": "class_B", "count": 1}],
    )
    sweep_results = [
        {
            "threshold": 0.85,
            "auto_coverage_pct": 80.0,
            "auto_accuracy_pct": 95.0,
            "fallback_rate_pct": 20.0,
            "confident_count": 16,
            "fallback_count": 4,
        }
    ]

    out_md = tmp_path / "summary.md"
    generate_evaluation_summary_md(report, sweep_results, output_path=out_md)

    assert out_md.exists()
    content = out_md.read_text(encoding="utf-8")
    assert "LogoDet-3K's held-out test split" in content
    assert "90.00%" in content
    assert "`closed_set_confidence_threshold = 0.85`" in content
