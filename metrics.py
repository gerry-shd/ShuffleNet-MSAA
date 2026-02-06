from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score


def compute_metrics(y_true: List[int], y_pred: List[int], class_names: List[str]) -> Dict[str, Any]:
    """Calculate standard classification summary metrics from predictions.

    Args:
        y_true: Ground-truth label indices.
        y_pred: Predicted label indices.
        class_names: Label strings used for reports/plots.
    """
    top1 = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    return {"top1": top1, "macro_f1": macro_f1, "weighted_f1": weighted_f1, "report": report, "cm": cm}


def save_confusion_matrix(cm: np.ndarray, class_names: List[str], out_png: str, out_csv: str) -> None:
    """Persist the confusion matrix both as a CSV file and a heatmap image.

    Args:
        cm: Confusion matrix with shape [num_classes, num_classes].
        class_names: Ordered class labels for axes/legend.
        out_png: Destination path for the rendered heatmap.
        out_csv: Destination path for the raw matrix values.
    """
    import csv
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([""] + class_names)
        for i, row in enumerate(cm):
            w.writerow([class_names[i]] + row.tolist())

    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)

    thresh = cm.max() * 0.6 if cm.max() > 0 else 0.0  # Switch font color for better readability.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
