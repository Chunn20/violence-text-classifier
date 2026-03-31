# src/data_utils.py
"""
数据加载与预处理工具。
完整保留原 notebook 的数据切分逻辑（0.9/0.1, seed=42）。
"""

import os

import pandas as pd
from datasets import Dataset


# ---------------------------------------------------------------------------
# Gold 数据加载
# ---------------------------------------------------------------------------

def load_gold_data(
    gold_file: str,
    label_col: str,
    text_col: str,
) -> tuple:
    """
    从 Excel 中加载三个 sheet（train / validation / test）。

    返回: (df_train, df_val, df_test)
    """
    if not os.path.exists(gold_file):
        raise FileNotFoundError(f"Gold 数据文件不存在: {gold_file}")

    df_train = pd.read_excel(gold_file, sheet_name="train",      engine="openpyxl")
    df_val   = pd.read_excel(gold_file, sheet_name="validation", engine="openpyxl")
    df_test  = pd.read_excel(gold_file, sheet_name="test",       engine="openpyxl")

    print(f"[Gold] Train={df_train.shape}, Validation={df_val.shape}, Test={df_test.shape}")

    # 列存在性检查
    for name, df in [("train", df_train), ("validation", df_val), ("test", df_test)]:
        for col in [text_col, label_col]:
            if col not in df.columns:
                raise ValueError(
                    f"Sheet '{name}' 缺少列 '{col}'。当前列: {list(df.columns)}"
                )

    return df_train, df_val, df_test


# ---------------------------------------------------------------------------
# 数据预处理（保留原始逻辑）
# ---------------------------------------------------------------------------

def prepare_data(
    df: pd.DataFrame,
    label_col: str,
    text_col: str,
    output_dir: str,
) -> tuple:
    """
    数据预处理：
      1. 生成数值 label（astype category → codes，与原 notebook 完全一致）
      2. 保存 label_mapping.csv 到 output_dir
      3. 内部做 0.9/0.1 切分（seed=42），返回 HuggingFace Dataset

    返回: (dataset, mapping_df)
      dataset["train"] / dataset["test"] 对应内部 90% / 10%
    """
    print("开始数据预处理...")
    os.makedirs(output_dir, exist_ok=True)

    df = df.copy().reset_index(drop=True)

    # 数值标签（categorical codes，与原始逻辑一致）
    df["label"] = df[label_col].astype("category").cat.codes
    df = df[[text_col, "label", label_col]]

    # 保存 label mapping
    mapping_df = (
        df[["label", label_col]]
        .drop_duplicates()
        .sort_values("label")
        .reset_index(drop=True)
    )
    print("Label mapping:\n", mapping_df)

    mapping_path = os.path.join(output_dir, "label_mapping.csv")
    mapping_df.to_csv(mapping_path, index=False)
    print(f"[Saved] {mapping_path}")

    # 0.9/0.1 内部切分（seed=42，与原始逻辑一致）
    dataset = Dataset.from_pandas(df)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    print(f"内部训练集大小: {len(dataset['train'])}")
    print(f"内部验证集大小: {len(dataset['test'])}")

    return dataset, mapping_df


# ---------------------------------------------------------------------------
# 分词处理（保留原始逻辑）
# ---------------------------------------------------------------------------

def tokenize_data(
    dataset,
    tokenizer,
    text_col: str,
    max_length: int = 512,
):
    """
    对 HuggingFace Dataset 做 tokenize。
    text_col 由外部传入（来自 task_cfg["text_col"]），不硬编码。
    """
    def _tokenize(example):
        return tokenizer(
            example[text_col],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    print("开始分词处理...")
    tokenized = dataset.map(_tokenize, batched=True)
    print("分词完成")
    return tokenized
