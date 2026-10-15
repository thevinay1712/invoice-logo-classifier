"""Model evaluation module: accuracy metrics, confusion matrix plotting, and confidence threshold calibration."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class EvaluationReport:
    """Dataclass holding evaluation metrics, confusion matrix, and top confused pairs."""

    overall_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class_metrics: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]
    class_list: list[str]
    confused_pairs: list[dict[str, Any]] = field(default_factory=list)


def evaluate_predictions(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_list: list[str],
) -> EvaluationReport:
    """Calculate evaluation metrics, per-class breakdown, confusion matrix, and confused pairs.

    Args:
        y_true: Array/list of true integer class targets.
        y_pred: Array/list of predicted integer class targets.
        class_list: Ordered list of class names.

    Returns:
        EvaluationReport object containing metric summaries.
    """
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    num_classes = len(class_list)

    total_samples = len(y_true_arr)
    correct = int((y_true_arr == y_pred_arr).sum())
    overall_acc = (correct / max(total_samples, 1)) * 100.0

    # Sklearn classification report
    report_dict = classification_report(
        y_true_arr,
        y_pred_arr,
        labels=list(range(num_classes)),
        target_names=class_list,
        output_dict=True,
        zero_division=0,
    )

    macro_prec = float(report_dict.get("macro avg", {}).get("precision", 0.0)) * 100.0
    macro_rec = float(report_dict.get("macro avg", {}).get("recall", 0.0)) * 100.0
    macro_f1 = float(report_dict.get("macro avg", {}).get("f1-score", 0.0)) * 100.0

    per_class = {}
    for cls_name in class_list:
        if cls_name in report_dict:
            cls_data = report_dict[cls_name]
            per_class[cls_name] = {
                "precision": float(cls_data["precision"]),
                "recall": float(cls_data["recall"]),
                "f1_score": float(cls_data["f1-score"]),
                "support": int(cls_data["support"]),
            }

    # Confusion matrix
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(num_classes)))

    # Identify top confused class pairs (true != pred)
    confused_list = []
    for r in range(num_classes):
        for c in range(num_classes):
            if r != c and cm[r, c] > 0:
                confused_list.append(
                    {
                        "true_class": class_list[r],
                        "predicted_class": class_list[c],
                        "count": int(cm[r, c]),
                    }
                )

    confused_list.sort(key=lambda x: x["count"], reverse=True)

    return EvaluationReport(
        overall_accuracy=round(overall_acc, 2),
        macro_precision=round(macro_prec, 2),
        macro_recall=round(macro_rec, 2),
        macro_f1=round(macro_f1, 2),
        per_class_metrics=per_class,
        confusion_matrix=cm.tolist(),
        class_list=class_list,
        confused_pairs=confused_list,
    )


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    class_list: list[str],
    device: torch.device | None = None,
) -> tuple[EvaluationReport, np.ndarray, np.ndarray]:
    """Run model inference over test dataloader and generate evaluation report.

    Args:
        model: Trained PyTorch classification model.
        dataloader: Test split DataLoader.
        class_list: Ordered class list matching model output indices.
        device: 'cuda' or 'cpu'. Defaults to CPU if not specified.

    Returns:
        (EvaluationReport, y_true_array, softmax_probabilities_array)
    """
    resolved_device = device if device is not None else torch.device("cpu")
    model.eval()
    model.to(resolved_device)

    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(resolved_device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_targets.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().numpy())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    probs_arr = np.array(all_probs)

    report = evaluate_predictions(y_true, y_pred, class_list)
    return report, y_true, probs_arr


def sweep_confidence_thresholds(
    y_true: np.ndarray | list[int],
    probs: np.ndarray,
    thresholds: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Sweep softmax confidence thresholds to calibrate automatic decision vs gallery fallback trigger.

    Args:
        y_true: True integer class array.
        probs: Softmax probability matrix (N, num_classes).
        thresholds: List of threshold values to evaluate (default 0.30 to 0.95, step 0.05).

    Returns:
        List of dictionary records summarizing coverage and accuracy at each threshold.
    """
    y_true_arr = np.array(y_true)
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.30, 0.96, 0.05)]

    max_probs = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    total_samples = len(y_true_arr)

    sweep_results = []
    for t in thresholds:
        confident_mask = max_probs >= t
        confident_count = int(np.sum(confident_mask))
        fallback_count = total_samples - confident_count

        auto_coverage = (confident_count / max(total_samples, 1)) * 100.0
        fallback_rate = (fallback_count / max(total_samples, 1)) * 100.0

        if confident_count > 0:
            auto_acc = (
                np.sum(y_true_arr[confident_mask] == preds[confident_mask]) / confident_count
            ) * 100.0
        else:
            auto_acc = 0.0

        sweep_results.append(
            {
                "threshold": t,
                "auto_coverage_pct": round(auto_coverage, 2),
                "auto_accuracy_pct": round(auto_acc, 2),
                "fallback_rate_pct": round(fallback_rate, 2),
                "confident_count": confident_count,
                "fallback_count": fallback_count,
            }
        )

    return sweep_results


def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    class_list: list[str],
    output_path: Path | str,
    title: str = "Logo Det-3K Test Split Confusion Matrix",
) -> Path:
    """Plot and save labeled confusion matrix heat map image.

    Args:
        cm: Confusion matrix 2D array.
        class_list: List of class labels for rows and columns.
        output_path: Path to save PNG image.
        title: Plot title header.

    Returns:
        Path to saved PNG image file.
    """
    cm_arr = np.array(cm)
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    fig_size = max(8, len(class_list) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(cm_arr, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, shrink=0.7)

    # Show all ticks and label them with class names
    ax.set(
        xticks=np.arange(cm_arr.shape[1]),
        yticks=np.arange(cm_arr.shape[0]),
        xticklabels=class_list,
        yticklabels=class_list,
        title=title,
        ylabel="True Class Label",
        xlabel="Predicted Class Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate matrix values if number of classes is reasonable
    if len(class_list) <= 25:
        thresh = cm_arr.max() / 2.0 if cm_arr.max() > 0 else 1.0
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                val = cm_arr[i, j]
                if val > 0:
                    ax.text(
                        j,
                        i,
                        format(val, "d"),
                        ha="center",
                        va="center",
                        color="white" if val > thresh else "black",
                        fontsize=8,
                    )

    fig.tight_layout()
    plt.savefig(target_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved confusion matrix plot to '{target_path}'.")
    return target_path


def generate_evaluation_summary_md(
    report: EvaluationReport,
    sweep_results: list[dict[str, Any]],
    output_path: Path | str,
) -> Path:
    """Write clean evaluation summary markdown report.

    Args:
        report: EvaluationReport instance.
        sweep_results: List of threshold calibration sweep records.
        output_path: File path for summary.md.

    Returns:
        Path to saved summary.md file.
    """
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Select recommended operating confidence threshold (e.g. highest accuracy with >= 70% coverage)
    rec_thresh = 0.85
    for s in sweep_results:
        if s["auto_accuracy_pct"] >= 85.0 and s["auto_coverage_pct"] >= 50.0:
            rec_thresh = s["threshold"]
            break

    md_lines = [
        "# Logo Classifier Evaluation & Threshold Calibration Summary",
        "",
        "> [!IMPORTANT]",
        "> **Evaluation Note**: These metrics are evaluated exclusively on **LogoDet-3K's held-out test split** logo crops. End-to-end document invoice localization and classification performance across full document scans (PDF/DOCX) is evaluated separately in Phase 15.",
        "",
        "## Overall Test Split Performance",
        "",
        f"- **Test Accuracy**: {report.overall_accuracy:.2f}%",
        f"- **Macro Precision**: {report.macro_precision:.2f}%",
        f"- **Macro Recall**: {report.macro_recall:.2f}%",
        f"- **Macro F1 Score**: {report.macro_f1:.2f}%",
        f"- **Evaluated Classes**: {len(report.class_list)} target vendor classes",
        "",
        "## Top Confusion-Prone Class Pairs",
        "",
        "| True Class Label | Confused Prediction | Misclassified Count |",
        "| :--- | :--- | :---: |",
    ]

    if report.confused_pairs:
        for pair in report.confused_pairs[:10]:
            md_lines.append(
                f"| `{pair['true_class']}` | `{pair['predicted_class']}` | {pair['count']} |"
            )
    else:
        md_lines.append("| *None* | *None* | 0 |")

    md_lines.extend(
        [
            "",
            "## Softmax Confidence Threshold Calibration Sweep",
            "",
            f"**Recommended Operating Threshold**: `closed_set_confidence_threshold = {rec_thresh}`",
            "",
            "| Confidence Threshold ($T$) | Auto-Classified Coverage (%) | Auto-Classification Accuracy (%) | Fallthrough to Gallery/Review (%) |",
            "| :---: | :---: | :---: | :---: |",
        ]
    )

    for s in sweep_results:
        md_lines.append(
            f"| `{s['threshold']:.2f}` | {s['auto_coverage_pct']:.2f}% | {s['auto_accuracy_pct']:.2f}% | {s['fallback_rate_pct']:.2f}% |"
        )

    md_lines.extend(
        [
            "",
            "---",
            "*Report generated automatically by `invoice_logo_classifier.evaluation`.*",
        ]
    )

    content = "\n".join(md_lines)
    with open(target_path, mode="w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Wrote evaluation summary report to '{target_path}'.")
    return target_path
