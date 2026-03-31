# src/predict_utils.py
"""
推理工具函数。

关键设计原则（严格遵守）：
  1. 模型路径完全由 task_cfg["saved_model_dir"] 决定，不存在任何硬编码路径
  2. 输入文本列完全由 task_cfg["text_col"] 决定（= "new_text"），不存在任何硬编码列名
  3. 输出列名（pred_col / conf_col / prob_cols）完全由 task_cfg 驱动
  4. 概率列名由 get_prob_cols() 从 id_to_label + pred_col 动态推导

完整保留原 notebook 的推理逻辑：
  - 自适应 OOM（自动降低 batch 重试）
  - 断点续跑（每 save_every 条写一次 ckpt）
  - Ctrl+C 安全中断
  - JSON stats 生成
"""

import logging
import os
import time

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .utils_io import get_prob_cols, save_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 设备 & 类型工具
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """优先 MPS > CUDA > CPU（与原 notebook 一致）。"""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_torch_dtype(device: torch.device):
    """CUDA 用 float16，其余用 float32（与原 notebook 一致）。"""
    return torch.float16 if device.type == "cuda" else torch.float32


def get_batch_size_start(device: torch.device) -> int:
    """CUDA 初始 batch=64，其余 batch=16（与原 notebook 一致）。"""
    return 64 if device.type == "cuda" else 16


def empty_cache(device: torch.device) -> None:
    """释放显存（与原 notebook 一致）。"""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_path: str,
    torch_dtype=None,
    low_cpu_mem_usage: bool = True,
) -> tuple:
    """
    从 saved_model_dir 加载 tokenizer 和 model。
    model_path 必须来自 task_cfg["saved_model_dir"]，由调用方保证。
    """
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"未找到模型目录: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token if tokenizer.eos_token is not None
            else tokenizer.unk_token
        )
    tokenizer.padding_side = "right"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model


# ---------------------------------------------------------------------------
# 推理主循环（完全由 task_cfg 驱动）
# ---------------------------------------------------------------------------

def run_inference(
    df: pd.DataFrame,
    model,
    tokenizer,
    task_cfg: dict,
    device: torch.device,
    out_dir: str,
    ckpt_file: str,
    output_file: str,
    save_every: int = 2000,
) -> pd.DataFrame:
    """
    对 df 进行批量推理，支持断点续跑。

    所有列名均来自 task_cfg，无任何硬编码：
      text_col  = task_cfg["text_col"]          (= "new_text")
      pred_col  = task_cfg["pred_col"]
      conf_col  = task_cfg["conf_col"]
      prob_cols = get_prob_cols(task_cfg)        (由 id_to_label + pred_col 推导)
    """
    # ── 从 task_cfg 读取所有列名配置 ─────────────────────────────────────────
    text_col    = task_cfg["text_col"]          # 来自 global.text_col，= "new_text"
    pred_col    = task_cfg["pred_col"]
    conf_col    = task_cfg["conf_col"]
    id_to_label = task_cfg["id_to_label"]       # {int: str}，已在 get_task_config 转换
    max_length  = task_cfg.get("max_length", 128)
    task_name   = task_cfg.get("task_name", "")

    # 概率列：[(label_str, col_name), ...]，顺序与 softmax 维度严格对应
    prob_col_pairs  = get_prob_cols(task_cfg)
    all_output_cols = [pred_col, conf_col] + [col for _, col in prob_col_pairs]

    # ── 输入验证 ──────────────────────────────────────────────────────────────
    if text_col not in df.columns:
        raise ValueError(
            f"输入文件缺少 '{text_col}' 列（来自 config.text_col）。"
            f"当前列: {list(df.columns)}"
        )
    df[text_col] = df[text_col].fillna("").astype(str)

    # ── 断点恢复 ──────────────────────────────────────────────────────────────
    if os.path.exists(ckpt_file):
        logger.info(f"发现断点文件，正在恢复：{ckpt_file}")
        ckpt = pd.read_csv(ckpt_file, low_memory=False)

        for c in all_output_cols:
            if c not in ckpt.columns:
                raise ValueError(
                    f"断点文件缺少列 '{c}'，无法恢复。"
                    "如需重新推理，请删除断点文件。"
                )
        if len(ckpt) != len(df):
            raise ValueError(
                f"断点文件行数({len(ckpt)})与输入行数({len(df)})不一致。"
                "请确认未更换或重排输入数据。"
            )
        for c in all_output_cols:
            df[c] = ckpt[c]
    else:
        for c in all_output_cols:
            df[c] = pd.NA

    # ── 确定待推理行 ──────────────────────────────────────────────────────────
    mask_todo    = df[pred_col].isna()
    todo_indices = df.index[mask_todo].tolist()
    todo_texts   = df.loc[mask_todo, text_col].tolist()

    n_total = len(df)
    n_todo  = len(todo_texts)
    logger.info(f"总数据: {n_total}；已完成: {n_total - n_todo}；待预测: {n_todo}")

    if n_todo == 0:
        logger.info("断点文件显示已全部预测完成，直接输出最终文件。")
        os.makedirs(out_dir, exist_ok=True)
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        logger.info(f"已保存结果到: {output_file}")
        return df

    # ── 推理循环（保留原 notebook 完整逻辑） ──────────────────────────────────
    BATCH_SIZE_START = get_batch_size_start(device)
    current_bs = BATCH_SIZE_START
    processed  = 0
    t0         = time.time()

    bar_format = "{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    logger.info(f"开始 {task_name} 推理...（支持 Ctrl+C 安全中断）")

    try:
        with tqdm(
            total=n_todo,
            desc=f"{task_name} 推理",
            unit="条",
            bar_format=bar_format,
            dynamic_ncols=True,
        ) as pbar:
            i = 0
            while i < n_todo:
                bs            = min(current_bs, n_todo - i)
                batch_texts   = todo_texts[i : i + bs]
                batch_indices = todo_indices[i : i + bs]

                try:
                    enc = tokenizer(
                        batch_texts,
                        truncation=True,
                        max_length=max_length,
                        padding=True,
                        return_tensors="pt",
                    )
                    enc = {k: v.to(device) for k, v in enc.items()}

                    with torch.inference_mode():
                        if device.type == "cuda":
                            with torch.cuda.amp.autocast(dtype=torch.float16):
                                logits = model(**enc).logits
                        else:
                            logits = model(**enc).logits

                    probs    = torch.softmax(logits, dim=1)
                    pred_ids = torch.argmax(probs, dim=1).detach().cpu().tolist()
                    confs    = torch.max(probs, dim=1).values.detach().cpu().tolist()
                    probs_np = probs.detach().cpu().numpy()

                    # 写回 df
                    for row_i, idx in enumerate(batch_indices):
                        pid   = int(pred_ids[row_i])
                        label = id_to_label.get(pid, "unknown")

                        df.at[idx, pred_col] = label
                        df.at[idx, conf_col] = float(confs[row_i])

                        # 概率列：按 id_to_label 升序写入，与 softmax 维度对应
                        for k, (_, col_name) in enumerate(prob_col_pairs):
                            df.at[idx, col_name] = float(probs_np[row_i, k])

                    i         += bs
                    processed += bs
                    pbar.update(bs)
                    empty_cache(device)

                    # OOM 恢复后缓慢回升 batch size
                    if current_bs < BATCH_SIZE_START:
                        current_bs = min(BATCH_SIZE_START, current_bs + 1)

                    # 定期保存断点
                    if processed % save_every == 0:
                        df.to_csv(ckpt_file, index=False, encoding="utf-8-sig")
                        logger.info(
                            f"已保存断点：累计完成 {n_total - n_todo + processed} / {n_total}"
                        )

                except RuntimeError as e:
                    msg = str(e).lower()
                    if "out of memory" in msg or "mps backend out of memory" in msg:
                        empty_cache(device)
                        if current_bs == 1:
                            logger.error(
                                "batch_size=1 仍然 OOM。"
                                "请降低 max_length 或改用 CPU。"
                            )
                            raise
                        current_bs = max(1, current_bs // 2)
                        logger.warning(f"显存不足，降低 batch_size 至 {current_bs}，重试该批。")
                    else:
                        raise

    except KeyboardInterrupt:
        logger.warning("检测到中断（Ctrl+C），正在保存断点...")
        df.to_csv(ckpt_file, index=False, encoding="utf-8-sig")
        logger.info(f"已保存断点到: {ckpt_file}")
        logger.info("下次运行同一命令即可从断点继续。")
        raise SystemExit(0)

    # 最终保存
    df.to_csv(ckpt_file, index=False, encoding="utf-8-sig")
    elapsed = time.time() - t0
    logger.info(
        f"本次运行新增预测: {processed} 条，"
        f"用时 {elapsed:.2f} 秒，"
        f"平均速度 {processed / max(elapsed, 1e-9):.2f} 条/秒。"
    )
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    logger.info(f"已保存最终结果到: {output_file}")
    return df


# ---------------------------------------------------------------------------
# JSON stats 生成（统一格式，不区分任务）
# ---------------------------------------------------------------------------

def generate_stats(
    df: pd.DataFrame,
    task_cfg: dict,
    stats_file: str,
    model_path: str,
    input_file: str,
    output_file: str,
) -> None:
    """
    生成推理统计 JSON。
    统一使用 label_distribution（不再区分 gender_distribution），
    格式��原 notebook 一致，但字段名已标准化。
    """
    pred_col    = task_cfg["pred_col"]
    conf_col    = task_cfg["conf_col"]
    task_name   = task_cfg.get("task_name", "")
    id_to_label = task_cfg["id_to_label"]
    labels_ordered = [id_to_label[i] for i in sorted(id_to_label.keys())]
    prob_col_pairs = get_prob_cols(task_cfg)

    try:
        conf_series = pd.to_numeric(df[conf_col], errors="coerce")

        stats = {
            "task":             task_name,
            "model_path":       model_path,
            "input_file":       os.path.basename(input_file),
            "output_file":      os.path.basename(output_file),
            "created_at":       time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_records":    int(len(df)),
            "avg_confidence":   float(conf_series.mean()),
            "label_distribution": df[pred_col].value_counts(dropna=False).to_dict(),
            "avg_confidence_by_label": {},
            "confidence_distribution": {
                "high(>=0.8)":    int((conf_series >= 0.8).sum()),
                "medium(0.6-0.8)": int(((conf_series >= 0.6) & (conf_series < 0.8)).sum()),
                "low(<0.6)":      int((conf_series < 0.6).sum()),
                "missing":        int(conf_series.isna().sum()),
            },
            "prob_summary": {},
        }

        for lab in labels_ordered:
            mask = df[pred_col] == lab
            if mask.any():
                stats["avg_confidence_by_label"][lab] = float(
                    pd.to_numeric(df.loc[mask, conf_col], errors="coerce").mean()
                )

        for lab, col in prob_col_pairs:
            s = pd.to_numeric(df[col], errors="coerce")
            stats["prob_summary"][col] = {
                "mean": float(s.mean()),
                "min":  float(s.min()),
                "max":  float(s.max()),
            }

        save_json(stats, stats_file)
        logger.info(f"已生成统计文件: {stats_file}")

    except Exception as e:
        logger.warning(f"生成 JSON stats 失败: {e}")
