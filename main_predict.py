# main_predict.py
"""
统一推理入口。

用法：
  python main_predict.py --task author_gender --input data/full_dataset.csv
  python main_predict.py --task relationship  --input data/full_dataset.csv --output_dir outputs/relationship/

关键设计：
  - 模型路径由 task_cfg["saved_model_dir"] 唯一决定（不接受 --model_path 参数）
  - 输入文本列由 task_cfg["text_col"] 决定（= "new_text"），不硬编码
  - 输出列名（pred_col / conf_col / prob_cols）完全由 task_cfg 驱动
"""

import argparse
import logging
import os
import sys

import pandas as pd
import torch

from src.predict_utils import (
    generate_stats,
    get_device,
    get_torch_dtype,
    load_model_and_tokenizer,
    run_inference,
)
from src.utils_io   import get_task_config, load_config
from src.utils_seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("main_predict")

VALID_TASKS = [
    "author_gender",
    "victim_gender",
    "perpetrator_gender",
    "relationship",
    "author_is_victim",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="统一推理入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python main_predict.py --task author_gender --input data/reddit_2026.csv
  python main_predict.py --task relationship  --input data/reddit_2026.csv --output_dir outputs/rel/
        """,
    )
    parser.add_argument(
        "--task",   required=True, choices=VALID_TASKS, help="任务名称"
    )
    parser.add_argument(
        "--input",  required=True, help="输入 CSV 文件路径（需包含 new_text 列）"
    )
    parser.add_argument(
        "--config", default="configs/tasks.yaml", help="配置文件路径（默认: configs/tasks.yaml）"
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="推理输出目录（默认: outputs/<task_name>/）",
    )
    parser.add_argument(
        "--save_every", type=int, default=2000,
        help="断点保存频率（每 N 条写一次 ckpt，默认 2000）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 加载配置 ──────────────────────────────────────────────────────────────
    config   = load_config(args.config)
    task_cfg = get_task_config(config, args.task)

    set_seed(task_cfg["seed"])

    # ── 检查 saved_model_dir（训练保存 & 推理加载的唯一路径来源）────────────────
    model_path = task_cfg["saved_model_dir"]
    if not os.path.isdir(model_path):
        logger.error(f"未找到模型目录: {model_path}")
        logger.error(
            f"请先运行: python main_train.py --task {args.task} --mode train"
        )
        sys.exit(1)

    # ── 检查输入文件 ──────────────────────────────────────────────────────────
    input_file = args.input
    if not os.path.exists(input_file):
        logger.error(f"输入文件不存在: {input_file}")
        sys.exit(1)

    # ── 确定输出路径 ──────────────────────────────────────────────────────────
    out_dir = args.output_dir or os.path.join("outputs", args.task)
    os.makedirs(out_dir, exist_ok=True)

    in_base     = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(out_dir, f"{in_base}_with_{args.task}.csv")
    ckpt_file   = os.path.join(out_dir, f"{args.task}_ckpt.csv")
    stats_file  = os.path.join(out_dir, f"{args.task}_stats.json")

    logger.info("=" * 60)
    logger.info(f"Task:             {args.task}")
    logger.info(f"saved_model_dir:  {model_path}")   # 唯一来源，训练和推理共用
    logger.info(f"text_col:         {task_cfg['text_col']}")  # = "new_text"
    logger.info(f"pred_col:         {task_cfg['pred_col']}")
    logger.info(f"conf_col:         {task_cfg['conf_col']}")
    logger.info(f"Input:            {input_file}")
    logger.info(f"Output:           {output_file}")
    logger.info(f"Checkpoint:       {ckpt_file}")
    logger.info(f"Stats:            {stats_file}")
    logger.info("=" * 60)

    # ── 设备与 dtype ──────────────────────────────────────────────────────────
    device      = get_device()
    torch_dtype = get_torch_dtype(device)
    logger.info(f"Device: {device}  |  dtype: {torch_dtype}")

    # ── 加载模型（路径完全由 saved_model_dir 决定）────────────────────────────
    logger.info(f"加载模型和分词器：{model_path}")
    tokenizer, model = load_model_and_tokenizer(model_path, torch_dtype=torch_dtype)

    # 验证 num_labels 是否与配置一致
    expected = task_cfg["num_labels"]
    actual   = getattr(model.config, "num_labels", None)
    if actual != expected:
        logger.warning(
            f"模型 num_labels={actual}，configs 期望 {expected}。"
            "请确认 saved_model_dir 指向正确的任务模型。"
        )

    model.to(device).eval()
    logger.info("模型加载成功。")

    # ── 读取输入数据 ──────────────────────────────────────────────────────────
    logger.info(f"读取输入文件：{input_file}")
    df = pd.read_csv(input_file, low_memory=False)
    logger.info(f"数据量: {len(df)} 行 × {len(df.columns)} 列")

    # ── 推理（text_col 由 task_cfg 决定，= "new_text"）────────────────────────
    df = run_inference(
        df,
        model,
        tokenizer,
        task_cfg,
        device,
        out_dir=out_dir,
        ckpt_file=ckpt_file,
        output_file=output_file,
        save_every=args.save_every,
    )

    # ── 生成 stats ────────────────────────────────────────────────────────────
    generate_stats(
        df,
        task_cfg,
        stats_file,
        model_path=model_path,
        input_file=input_file,
        output_file=output_file,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
