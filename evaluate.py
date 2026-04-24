"""
evaluate.py — Model evaluation utilities
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def evaluate_model(y_true, y_pred, y_proba=None, model_name="Model"):
    """Print and return a full evaluation report."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec  = recall_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred)

    print(f"\n{'='*45}")
    print(f"  {model_name} — Evaluation Report")
    print(f"{'='*45}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")

    if y_proba is not None:
        auc = roc_auc_score(y_true, y_proba)
        print(f"  ROC-AUC   : {auc:.4f}")

    print(f"\n{classification_report(y_true, y_pred, target_names=['Fake', 'Real'])}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"  Confusion Matrix:\n  TN={cm[0,0]}  FP={cm[0,1]}\n  FN={cm[1,0]}  TP={cm[1,1]}")

    return {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1": round(f1 * 100, 2),
    }


def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Fake", "Real"]); ax.set_yticklabels(["Fake", "Real"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"📊 Confusion matrix saved → {save_path}")
