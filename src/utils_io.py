# src/utils_io.py
"""
配置加载、JSON 保存、概率列名推导等 I/O 工具函数。

关键设计原则：
  - saved_model_dir 是唯一路径来源（训练保存 & 推理加载）
  - text_col 始终来自 config["global"]["text_col"]（= "new_text"）
  - 概率列名由 id_to_label + pred_col 动态推导，不硬编码
"""

import os
import json
from typing import Dict, List, Tuple

import yaml


# ---------------------------------------------------------------------------
# 配置加载
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """加载 tasks.yaml 配置文件。"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_task_config(config: dict, task_name: str) -> dict:
    """
    获取指定任务的合并配置（global + task-specific）。
    返回的 dict 中会注入 task_name，global 字段作为默认值。
    """
    tasks = config.get("tasks", {})
    if task_name not in tasks:
        raise ValueError(
            f"Task '{task_name}' 不在配置中。"
            f"可用任务: {list(tasks.keys())}"
        )
    # task-specific 字段优先于 global 字段
    task_cfg = dict(config.get("global", {}))
    task_cfg.update(config["tasks"][task_name])
    task_cfg["task_name"] = task_name

    # 确保 id_to_label 的 key 为 int（yaml 整数 key 已是 int，但防御性处理）
    task_cfg["id_to_label"] = {
        int(k): str(v) for k, v in task_cfg["id_to_label"].items()
    }
    return task_cfg


# ---------------------------------------------------------------------------
# JSON 保存
# ---------------------------------------------------------------------------

def save_json(data: dict, path: str) -> None:
    """将 dict 保存为 UTF-8 JSON，自动创建父目录。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# 概率列名推导（不硬编码，由 id_to_label + pred_col 决定）
# ---------------------------------------------------------------------------

def label_to_prob_col(pred_col: str, label: str) -> str:
    """
    由 pred_col 和 label 名称推导概率列名。

    规则（与原 notebook 完全一致）：
      pred_col 去掉 "_finetune" 后缀作为前缀
      label 中 " & " → "_"，空格 → "_"

    示例：
      ("author_gender_finetune", "female")       → "author_gender_prob_female"
      ("victim_gender_finetune", "female & male") → "victim_gender_prob_female_male"
      ("author_is_victim_finetune", "False")      → "author_is_victim_prob_False"
      ("relationship_finetune", "dad")            → "relationship_prob_dad"
    """
    prefix = pred_col.replace("_finetune", "")
    safe_label = label.replace(" & ", "_").replace(" ", "_")
    return f"{prefix}_prob_{safe_label}"


def get_prob_cols(task_cfg: dict) -> List[Tuple[str, str]]:
    """
    返回按 id 升序排列的 [(label, prob_col_name), ...] 列表。
    顺序与模型输出的 softmax 概率维度严格对应。
    """
    id_to_label = task_cfg["id_to_label"]          # {int: str}
    pred_col    = task_cfg["pred_col"]
    return [
        (id_to_label[i], label_to_prob_col(pred_col, id_to_label[i]))
        for i in sorted(id_to_label.keys())
    ]
