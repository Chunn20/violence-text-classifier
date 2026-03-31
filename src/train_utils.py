# src/train_utils.py
"""
训练工具函数。
完整保留原 notebook 中的所有训练逻辑，包括：
  - 本地模型加载（local_files_only=True）
  - TrainingArguments 版本兼容性处理
  - 两阶段分层参数搜索（仅 author_gender 使用）
  - 最终模型训练 + 测试集评估
"""

import inspect
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .data_utils import prepare_data, tokenize_data
from .utils_io import save_json
from .utils_metrics import CSVLoggerCallback, compute_metrics


# ---------------------------------------------------------------------------
# 模型加载（local_files_only=True，保留原始逻辑）
# ---------------------------------------------------------------------------

def setup_tokenizer_and_model(
    model_path: str,
    num_labels: int,
) -> tuple:
    """
    从本地路径加载 tokenizer 和 model。
    local_files_only=True 确保离线运行。
    返回: (tokenizer, model, pad_token_id)
    """
    print(f"从本地路径加载模型和分词器: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型路径不存在: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    # LLaMA/DeepSeek 系：若无 pad_token，用 eos_token 兜底（保留原始逻辑）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("[Info] tokenizer.pad_token was None → set to eos_token")
    pad_token_id = tokenizer.pad_token_id

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=num_labels, local_files_only=True
    )
    try:
        model.config.pad_token_id = pad_token_id
    except Exception:
        pass

    print(f"成功从本地加载模型: {model_path}")
    return tokenizer, model, pad_token_id


def _load_fresh_model(
    model_path: str,
    num_labels: int,
    pad_token_id,
):
    """每次训练前重新从 base model 加载（参数搜索时每个 combo 独立加载）。"""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=num_labels, local_files_only=True
    )
    try:
        model.config.pad_token_id = pad_token_id
    except Exception:
        pass
    return model


# ---------------------------------------------------------------------------
# TrainingArguments 版本兼容性处理（完整保留原始逻辑）
# ---------------------------------------------------------------------------

def make_training_args(
    param_combo: dict,
    output_dir: str,
    **extra_kwargs,
) -> TrainingArguments:
    """
    根据本机 transformers 版本智能映射/过滤不支持的参数。
    完整保留原 notebook 的兼容性处理逻辑。
    """
    sig       = inspect.signature(TrainingArguments.__init__)
    supported = set(sig.parameters.keys())

    kwargs = {
        "output_dir":                    output_dir,
        "per_device_train_batch_size":   param_combo.get("per_device_train_batch_size", 1),
        "per_device_eval_batch_size":    1,
        "num_train_epochs":              param_combo.get("num_train_epochs", 3),
        "learning_rate":                 param_combo.get("learning_rate", 2e-5),
        "weight_decay":                  param_combo.get("weight_decay", 0.0),
    }
    kwargs.update(extra_kwargs)

    # evaluation_strategy 兼容旧版本
    eval_strategy = kwargs.pop("evaluation_strategy", None)
    if "evaluation_strategy" not in supported and "evaluate_during_training" in supported:
        if eval_strategy in ("steps", "epoch"):
            kwargs["evaluate_during_training"] = True
        if (
            "eval_steps" in kwargs
            and "eval_steps" not in supported
            and "evaluate_during_training_steps" in supported
        ):
            kwargs["evaluate_during_training_steps"] = kwargs.pop("eval_steps")

    # 去掉旧版本不支持的参数
    for k in ["report_to", "logging_strategy", "save_strategy"]:
        if k in kwargs and k not in supported:
            kwargs.pop(k)

    # per_device_eval_batch_size 旧版本兼容
    if (
        "per_device_eval_batch_size" in kwargs
        and "per_device_eval_batch_size" not in supported
        and "per_gpu_eval_batch_size" in supported
    ):
        kwargs["per_gpu_eval_batch_size"] = kwargs.pop("per_device_eval_batch_size")

    # 仅保留当前版本支持的参数
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    return TrainingArguments(**filtered)


# ---------------------------------------------------------------------------
# 单次训练（供参数搜索循环调用）
# ---------------------------------------------------------------------------

def _run_single_training(
    param_combo: dict,
    param_id: int,
    tokenized_dataset,
    tokenizer,
    num_labels: int,
    base_model_path: str,
    pad_token_id,
    base_output_dir: str,
    fixed_args: dict = None,
    phase: int = 1,
) -> tuple:
    """
    运行单次训练，返回 (final_metrics, model_output_dir)。
    每次都从 base_model_path 重新加载模型（不复用上一次的权重）。
    """
    print(f"\n{'='*60}")
    print(f"阶段{phase} - 参数组合 {param_id}: {param_combo}")
    print(f"{'='*60}")

    if fixed_args is None:
        fixed_args = {}

    model = _load_fresh_model(base_model_path, num_labels, pad_token_id)

    param_output_dir = os.path.join(base_output_dir, f"phase{phase}_param_{param_id}")
    logs_dir         = os.path.join(base_output_dir, "logs")
    csv_log_path     = os.path.join(logs_dir, f"training_metrics_phase{phase}.csv")

    combined = {**fixed_args, **param_combo}
    training_args = make_training_args(
        combined,
        param_output_dir,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        save_strategy="no",
        save_total_limit=1,
        do_train=True,
        do_eval=True,
        seed=42,
    )

    csv_logger = CSVLoggerCallback(csv_log_path, param_id=param_id, phase=phase)

    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[csv_logger],
        )
    except TypeError:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        try:
            trainer.add_callback(csv_logger)
        except Exception:
            pass

    trainer.train()
    final_metrics = trainer.evaluate()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_metrics, param_output_dir


# ---------------------------------------------------------------------------
# 两阶段分层参数搜索（仅 author_gender --mode search 使用）
# ---------------------------------------------------------------------------

def hierarchical_parameter_search(
    df_trainval: pd.DataFrame,
    task_cfg: dict,
    base_model_path: str,
    search_dir: str,
) -> dict:
    """
    两阶段分层参数搜索（保留原 notebook 完整逻辑）。

    阶段1：快速筛选 lr x wd，固定 epochs=2
    阶段2：在最佳 lr/wd 附近精细搜索 + epochs=[3,4]

    df_trainval: 已合并且 dropna 的 train+val 数据
    search_dir:  搜索输出目录，例如 runs/author_gender/search/
    """
    label_col  = task_cfg["label_col"]
    text_col   = task_cfg["text_col"]
    num_labels = task_cfg["num_labels"]
    max_length = task_cfg.get("max_length", 512)

    os.makedirs(search_dir, exist_ok=True)
    results_dir = os.path.join(search_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # 数据准备
    dataset, mapping_df = prepare_data(df_trainval, label_col, text_col, search_dir)
    tokenizer, _, pad_token_id = setup_tokenizer_and_model(base_model_path, num_labels)
    tokenized_dataset = tokenize_data(dataset, tokenizer, text_col, max_length)

    # ── 阶段1：快速筛选 ──────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("第一阶段：快速筛选")
    print("="*80)

    param_grid_phase1 = {
        "learning_rate": [1e-5, 5e-5],
        "weight_decay":  [0.0,  0.01],
    }
    fixed_args_phase1 = {
        "per_device_train_batch_size": 1,
        "num_train_epochs":            2,
    }
    combos_phase1 = list(ParameterGrid(param_grid_phase1))
    print(f"第一阶段: {len(combos_phase1)} 种参数组合")

    phase1_results      = []
    best_score_phase1   = -1.0
    best_params_phase1  = None

    for i, params in enumerate(combos_phase1):
        metrics, _ = _run_single_training(
            params, i, tokenized_dataset, tokenizer, num_labels,
            base_model_path, pad_token_id, search_dir, fixed_args_phase1, phase=1,
        )
        result = {"phase": 1, "param_id": i, "params": params, "metrics": metrics}
        phase1_results.append(result)

        score = metrics.get("eval_f1", 0)
        if score > best_score_phase1:
            best_score_phase1  = score
            best_params_phase1 = params
            print(f"  [阶段1 新最佳] F1={best_score_phase1:.4f}, params={params}")

    save_json(phase1_results, os.path.join(results_dir, "phase1_results.json"))

    # ── 阶段2：精细调整 ──────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("第二阶段：精细调整")
    print("="*80)

    best_lr = best_params_phase1["learning_rate"]
    best_wd = best_params_phase1["weight_decay"]
    print(f"第一阶段最佳: lr={best_lr}, wd={best_wd}, F1={best_score_phase1:.4f}")

    # 在最佳值附近生成候选
    if best_lr == 1e-5:
        lr_candidates = [5e-6, 1e-5, 2e-5]
    elif best_lr == 5e-5:
        lr_candidates = [3e-5, 5e-5, 7e-5]
    else:
        lr_candidates = [best_lr * 0.5, best_lr, best_lr * 1.5]

    if best_wd == 0.0:
        wd_candidates = [0.0, 0.005, 0.01]
    else:
        wd_candidates = [best_wd * 0.5, best_wd, best_wd * 1.5]

    param_grid_phase2 = {
        "learning_rate":               lr_candidates,
        "weight_decay":                wd_candidates,
        "per_device_train_batch_size": [1],
        "num_train_epochs":            [3, 4],
    }
    combos_phase2 = list(ParameterGrid(param_grid_phase2))
    print(f"第二阶段: {len(combos_phase2)} 种参数组合")

    phase2_results     = []
    best_score_phase2  = -1.0
    best_params_phase2 = None

    for i, params in enumerate(combos_phase2):
        metrics, _ = _run_single_training(
            params, i, tokenized_dataset, tokenizer, num_labels,
            base_model_path, pad_token_id, search_dir, {}, phase=2,
        )
        result = {"phase": 2, "param_id": i, "params": params, "metrics": metrics}
        phase2_results.append(result)

        score = metrics.get("eval_f1", 0)
        if score > best_score_phase2:
            best_score_phase2  = score
            best_params_phase2 = params
            print(f"  [阶段2 新最佳] F1={best_score_phase2:.4f}, params={params}")

    save_json(phase2_results, os.path.join(results_dir, "phase2_results.json"))

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    summary = {
        "best_score_phase1":  best_score_phase1,
        "best_params_phase1": best_params_phase1,
        "best_score_phase2":  best_score_phase2,
        "best_params_phase2": best_params_phase2,
        "param_grid_phase1":  param_grid_phase1,
        "param_grid_phase2":  param_grid_phase2,
        "created_at":         time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(summary, os.path.join(results_dir, "hierarchical_search_results.json"))

    print(f"\n搜索完成！")
    print(f"  第一阶段最佳: F1={best_score_phase1:.4f}, params={best_params_phase1}")
    print(f"  第二阶段最佳: F1={best_score_phase2:.4f}, params={best_params_phase2}")
    print(f"\n请将 best_params_phase2 更新到 configs/tasks.yaml 的 author_gender 配置。")
    return summary


# ---------------------------------------------------------------------------
# 最终模型训练（--mode train）
# ---------------------------------------------------------------------------

def train_final_model(
    df_trainval: pd.DataFrame,
    task_cfg: dict,
    base_model_path: str,
    run_dir: str,
    df_test: pd.DataFrame = None,
) -> dict:
    """
    使用 best_params 训练最终模型（保留原 notebook 完整逻辑）。

    df_trainval:     已合并且 dropna 的 train+val 数据
    task_cfg:        通过 get_task_config() 获取的任务配置
    base_model_path: base model 本地路径
    run_dir:         任务运行目录，例如 runs/author_gender/
    df_test:         独立测试集（可选），用于最终 test set 评估

    saved_model_dir = task_cfg["saved_model_dir"]（唯一路径来源）
    """
    label_col       = task_cfg["label_col"]
    text_col        = task_cfg["text_col"]
    num_labels      = task_cfg["num_labels"]
    max_length      = task_cfg.get("max_length", 512)
    best_params     = task_cfg["best_params"]
    saved_model_dir = task_cfg["saved_model_dir"]   # 训练保存 & 推理加载，唯一来源
    task_name       = task_cfg["task_name"]

    # 子目录
    logs_dir        = os.path.join(run_dir, "logs")
    results_dir     = os.path.join(run_dir, "results")
    checkpoints_dir = os.path.join(run_dir, "checkpoints")   # 训练中间 checkpoint
    os.makedirs(logs_dir,        exist_ok=True)
    os.makedirs(results_dir,     exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"开始训练最终模型：{task_name}")
    print(f"使用参数: {best_params}")
    print(f"saved_model_dir: {saved_model_dir}")
    print(f"{'='*80}")

    # 数据准备（内部 0.9/0.1 切分，seed=42）
    dataset, mapping_df = prepare_data(df_trainval, label_col, text_col, run_dir)

    # 加载 tokenizer + model
    tokenizer, model, pad_token_id = setup_tokenizer_and_model(base_model_path, num_labels)
    tokenized_dataset = tokenize_data(dataset, tokenizer, text_col, max_length)

    # 训练参数（checkpoints 写到 checkpoints_dir，最终模型单独保存到 saved_model_dir）
    csv_log_path  = os.path.join(logs_dir, "final_training_metrics.csv")
    training_args = make_training_args(
        best_params,
        checkpoints_dir,            # 训练过程中间 checkpoint 目录
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        metric_for_best_model="f1",
        greater_is_better=True,
        do_train=True,
        do_eval=True,
        seed=42,
        logging_first_step=True,
        logging_dir=logs_dir,
        label_names=["labels"],
    )

    csv_logger = CSVLoggerCallback(csv_log_path)

    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[csv_logger],
        )
    except TypeError:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        try:
            trainer.add_callback(csv_logger)
        except Exception:
            pass

    print("\n开始训练...")
    train_result  = trainer.train()

    print("\n评估模型（内部 eval 集）...")
    final_metrics = trainer.evaluate()

    # 保存最终模型到 saved_model_dir（训练和推理的唯一路径来源）
    print(f"\n保存最终模型到: {saved_model_dir}")
    trainer.save_model(saved_model_dir)

    # 保存 config snapshot（训练参数快照）
    config_snapshot = {
        "task_name":       task_name,
        "label_col":       label_col,
        "num_labels":      num_labels,
        "best_params":     best_params,
        "base_model_path": base_model_path,
        "saved_model_dir": saved_model_dir,
        "label_mapping":   mapping_df.to_dict("records"),
        "final_metrics":   final_metrics,
        "train_loss":      getattr(train_result, "training_loss", None),
        "total_steps":     getattr(train_result, "global_step", None),
        "saved_at":        time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(config_snapshot, os.path.join(results_dir, "final_model_results.json"))

    # 清理显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # 测试集评估（可选）
    if df_test is not None and len(df_test) > 0:
        print("\n在独立测试集上评估...")
        test_results = _evaluate_on_test(
            df_test, label_col, text_col,
            task_cfg["id_to_label"], mapping_df,
            saved_model_dir, max_length, results_dir,
        )
        config_snapshot["test_results"] = test_results

    _print_metrics_summary(final_metrics, saved_model_dir, run_dir)
    return config_snapshot


# ---------------------------------------------------------------------------
# 测试集评估（内部函数，保留原 notebook 逻辑）
# ---------------------------------------------------------------------------

def _evaluate_on_test(
    df_test: pd.DataFrame,
    label_col: str,
    text_col: str,
    id_to_label: dict,
    mapping_df: pd.DataFrame,
    saved_model_dir: str,
    max_length: int,
    results_dir: str,
) -> dict:
    """
    在独立 test sheet 上评估已保存模型（保留原 notebook 逻辑）。
    从 saved_model_dir 加载模型，与推理使用完全相同的路径。
    """
    df_test_clean = df_test.dropna(subset=[text_col, label_col]).copy()
    if len(df_test_clean) == 0:
        print("测试集为空或全为 NaN，跳过测试集评估。")
        return {}

    # 构建 label_str → int 映射（基于训练时的 mapping_df）
    label_to_int = {
        row[label_col]: int(row["label"])
        for _, row in mapping_df.iterrows()
    }

    test_texts      = df_test_clean[text_col].tolist()
    raw_labels      = df_test_clean[label_col].map(label_to_int)
    valid_mask      = raw_labels.notna()
    test_texts      = [t for t, v in zip(test_texts, valid_mask) if v]
    test_labels_int = [int(l) for l, v in zip(raw_labels, valid_mask) if v]

    if not test_texts:
        print("测试集中无有效标签样本，跳过。")
        return {}

    print(f"有效测试样本: {len(test_texts)}")

    device = (
        torch.device("cuda")  if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    # 从 saved_model_dir 加载（与推理使用同一路径）
    model     = AutoModelForSequenceClassification.from_pretrained(
        saved_model_dir, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(saved_model_dir, local_files_only=True)
    model.to(device).eval()

    predictions = []
    print("开始测试集推理...")
    for i in range(0, len(test_texts)):
        inputs = tokenizer(
            [test_texts[i]],
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            predictions.append(int(torch.argmax(logits, dim=-1).cpu()))
        if i % 10 == 0:
            print(f"  已处理 {i}/{len(test_texts)}")

    num_labels   = len(id_to_label)
    id_to_label_int = {int(k): v for k, v in id_to_label.items()}
    target_names = [id_to_label_int.get(i, str(i)) for i in range(num_labels)]

    acc = accuracy_score(test_labels_int, predictions)
    f1  = f1_score(test_labels_int, predictions, average="weighted")

    print(f"\n测试集结果: 准确率={acc:.4f}, 加权F1={f1:.4f}")
    print(classification_report(test_labels_int, predictions, target_names=target_names))

    results = {
        "test_accuracy":    float(acc),
        "test_f1_weighted": float(f1),
        "test_size":        len(test_texts),
        "predictions":      [int(p) for p in predictions],
        "true_labels":      test_labels_int,
        "sample_texts":     test_texts[:10],
    }
    save_json(results, os.path.join(results_dir, "test_results.json"))
    print(f"[Saved] 测试结果: {os.path.join(results_dir, 'test_results.json')}")
    return results


def _print_metrics_summary(metrics: dict, saved_model_dir: str, run_dir: str):
    print(f"\n{'='*80}")
    print("训练完成！")
    print(f"{'='*80}")
    print("最终 eval 指标:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
    print(f"\n保存路径:")
    print(f"  saved_model:    {saved_model_dir}")
    print(f"  训练日志:       {os.path.join(run_dir, 'logs', 'final_training_metrics.csv')}")
    print(f"  label_mapping:  {os.path.join(run_dir, 'label_mapping.csv')}")
    print(f"  训练结果:       {os.path.join(run_dir, 'results', 'final_model_results.json')}")
