# main_train.py
"""
统一训练入口。

用法：
  # 参数搜索（仅 author_gender 支持）
  python main_train.py --task author_gender --mode search

  # 最终训练（使用 configs/tasks.yaml 中的 best_params）
  python main_train.py --task author_gender --mode train
  python main_train.py --task victim_gender --mode train

  # 覆盖训练超参数
  python main_train.py --task author_gender --mode train --lr 5e-6 --epochs 4

训练输出目录：runs/<task_name>/
  saved_model_dir（由 yaml 配置，唯一路径来源）：runs/<task_name>/saved_model/
"""

import argparse
import os
import sys

# 强制离线模式（与原 notebook 一致）
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"

import pandas as pd

from src.data_utils   import load_gold_data
from src.train_utils  import hierarchical_parameter_search, train_final_model
from src.utils_io     import get_task_config, load_config
from src.utils_seed   import set_seed

VALID_TASKS = [
    "author_gender",
    "victim_gender",
    "perpetrator_gender",
    "relationship",
    "author_is_victim",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="统一训练入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python main_train.py --task author_gender --mode search
  python main_train.py --task author_gender --mode train
  python main_train.py --task victim_gender --mode train --lr 1e-5 --epochs 4
        """,
    )
    parser.add_argument(
        "--task",   required=True, choices=VALID_TASKS, help="任务名称"
    )
    parser.add_argument(
        "--mode",   required=True, choices=["search", "train"],
        help="search=两阶段参数搜索（仅 author_gender）; train=使用 best_params 训练",
    )
    parser.add_argument(
        "--config", default="configs/tasks.yaml", help="配置文件路径（默认: configs/tasks.yaml）"
    )
    # 可选超参数覆盖（仅 --mode train 时生效）
    parser.add_argument("--lr",     type=float, default=None, help="覆盖 best_params.learning_rate")
    parser.add_argument("--wd",     type=float, default=None, help="覆盖 best_params.weight_decay")
    parser.add_argument("--epochs", type=int,   default=None, help="覆盖 best_params.num_train_epochs")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 加载配置 ──────────────────────────────────────────────────────────────
    config   = load_config(args.config)
    task_cfg = get_task_config(config, args.task)

    set_seed(task_cfg["seed"])

    # ── 约束：search 仅支持 author_gender ────────────────────────────────────
    if args.mode == "search" and args.task != "author_gender":
        print(f"[ERROR] --mode search 仅支持 author_gender（收到 '{args.task}'）。")
        print("其他任务的 best_params 已记录在 configs/tasks.yaml，请直接使用 --mode train。")
        sys.exit(1)

    # ── 检查 base model 路径 ─────────────────────────────────────────────────
    base_model_path = config["global"]["base_model_path"]
    if not os.path.exists(base_model_path):
        print(f"[ERROR] base_model_path 不存在: {base_model_path}")
        print("请在 configs/tasks.yaml 的 global.base_model_path 中填写正确的本地路径。")
        sys.exit(1)

    # ── 加载 gold 数据 ────────────────────────────────────────────────────────
    gold_file = task_cfg["gold_file"]
    label_col = task_cfg["label_col"]
    text_col  = task_cfg["text_col"]

    print(f"\n任务:       {args.task}")
    print(f"模式:       {args.mode}")
    print(f"Gold 文件:  {gold_file}")
    print(f"标签列:     {label_col}")
    print(f"文本列:     {text_col}")
    print(f"Base model: {base_model_path}")
    print(f"saved_model_dir: {task_cfg['saved_model_dir']}")

    df_train, df_val, df_test = load_gold_data(gold_file, label_col, text_col)

    # 合并 train + validation，dropna（与原 notebook 完全一致）
    df_trainval = pd.concat([df_train, df_val], ignore_index=True)
    df_trainval = df_trainval.dropna(subset=[text_col, label_col])
    print(f"合并 train+val（dropna 后）: {df_trainval.shape}")

    # ── 分支：search vs train ─────────────────────────────────────────────────
    if args.mode == "search":
        search_dir = os.path.join("runs", args.task, "search")
        print(f"\n[mode=search] 两阶段参数搜索，输出目录: {search_dir}")

        summary = hierarchical_parameter_search(
            df_trainval, task_cfg, base_model_path, search_dir
        )

        print(f"\n建议将以下 best_params 更新到 configs/tasks.yaml 的 [{args.task}] 配置：")
        bp = summary.get("best_params_phase2") or summary.get("best_params_phase1", {})
        for k, v in bp.items():
            print(f"  {k}: {v}")

    else:  # mode == "train"
        # 命令行覆盖超参数（可选）
        if args.lr     is not None: task_cfg["best_params"]["learning_rate"]    = args.lr
        if args.wd     is not None: task_cfg["best_params"]["weight_decay"]     = args.wd
        if args.epochs is not None: task_cfg["best_params"]["num_train_epochs"] = args.epochs

        run_dir = os.path.join("runs", args.task)
        print(f"\n[mode=train] 最终训练，输出目录: {run_dir}")
        print(f"使用 best_params: {task_cfg['best_params']}")

        train_final_model(
            df_trainval,
            task_cfg,
            base_model_path,
            run_dir,
            df_test=df_test,
        )


if __name__ == "__main__":
    main()
