# src/utils_metrics.py
"""
评估指标计算与 CSV 训练日志回调。
完整保留原 notebook 逻辑，合并了 search 版和 final 版的两个 CSVLoggerCallback。
"""

import csv
import os
import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from transformers import TrainerCallback


# ---------------------------------------------------------------------------
# 评估指标（完整保留原始逻辑）
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred):
    """
    计算 accuracy / weighted-F1 / AUROC / AUPRC。
    与原 notebook 逻辑完全一致。
    """
    logits, labels = eval_pred
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    preds = np.argmax(logits, axis=-1)
    acc   = accuracy_score(labels, preds)
    f1w   = f1_score(labels, preds, average="weighted")

    # softmax 概率（数值稳定）
    if logits.ndim == 1:
        probs     = 1 / (1 + np.exp(-logits))
        n_classes = 2
    else:
        e         = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs     = e / e.sum(axis=1, keepdims=True)
        n_classes = logits.shape[1]

    auroc_macro_ovr = None
    auprc_macro     = None
    try:
        if n_classes == 2:
            y_score         = probs[:, 1] if probs.ndim > 1 else probs
            auroc_macro_ovr = roc_auc_score(labels, y_score)
            auprc_macro     = average_precision_score(labels, y_score)
        else:
            y_true          = label_binarize(labels, classes=list(range(n_classes)))
            auroc_macro_ovr = roc_auc_score(
                y_true, probs, average="macro", multi_class="ovr"
            )
            auprc_macro = average_precision_score(y_true, probs, average="macro")
    except Exception:
        pass  # 某些极端情况（单类别等）跳过

    out = {"accuracy": acc, "f1": f1w}
    if auroc_macro_ovr is not None:
        out["auroc_macro_ovr"] = auroc_macro_ovr
    if auprc_macro is not None:
        out["auprc_macro"] = auprc_macro
    return out


# ---------------------------------------------------------------------------
# CSV 训练日志回调（合并 search 版 + final 版）
# ---------------------------------------------------------------------------

class CSVLoggerCallback(TrainerCallback):
    """
    每次 on_evaluate / on_log 时追加写入 CSV 日志。
    param_id 和 phase 为可选字段（仅参数搜索时使用）。
    """

    def __init__(self, csv_path: str, param_id=None, phase=None):
        self.csv_path = csv_path
        self.param_id = param_id
        self.phase    = phase
        self._init_file()

    def _init_file(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.csv_path)), exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    "phase", "param_id", "ts", "epoch", "step",
                    "split", "loss", "accuracy", "f1",
                    "auroc_macro_ovr", "auprc_macro",
                ])

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                self.phase,
                self.param_id,
                time.strftime("%Y-%m-%d %H:%M:%S"),
                getattr(state, "epoch", None),
                state.global_step,
                "eval",
                metrics.get("eval_loss"),
                metrics.get("eval_accuracy"),
                metrics.get("eval_f1"),
                metrics.get("eval_auroc_macro_ovr"),
                metrics.get("eval_auprc_macro"),
            ])

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                self.phase,
                self.param_id,
                time.strftime("%Y-%m-%d %H:%M:%S"),
                getattr(state, "epoch", None),
                state.global_step,
                "train_log",
                logs.get("loss"),
                logs.get("accuracy"),
                logs.get("f1"),
                logs.get("auroc_macro_ovr"),
                logs.get("auprc_macro"),
            ])
