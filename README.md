# Reddit Violence Text Classification — Unified Framework

## 项目概述

本项目将原有的 5 套分散 notebook 重构为**统一可配置的工程化 Python 项目**，支持：

- 通过命令行参数切换任务（5 个分类任务）
- 统一训练入口（参数搜索 / 最终训练）
- 统一推理入口（支持断点续跑）
- 固定随机种子、保存日志 / 指标 / label mapping
- 完全离线运行（`local_files_only=True`）

---

## 目录结构

```
violence-text-classifier/
├── README.md
├── requirements.txt
├── configs/
│   └── tasks.yaml              # 5 个任务的全部差异配置 + 全局配置
├── src/
│   ├── __init__.py
│   ├── utils_seed.py           # 随机种子固定
│   ├── utils_io.py             # 配置加载、JSON 保存、概率列名推导
│   ├── utils_metrics.py        # compute_metrics、CSVLoggerCallback
│   ├── data_utils.py           # load_gold_data、prepare_data、tokenize_data
│   ├── train_utils.py          # 模型加载、训练参数、参数搜索、最终训练
│   └── predict_utils.py        # 推理循环、断点续跑、stats 生成
├── main_train.py               # 统一训练入口
├── main_predict.py             # 统一推理入口
├── runs/                       # 训练输出（按任务分子目录，自动创建）
│   └── <task_name>/
│       ├── saved_model/        # ← saved_model_dir，训练保存 & 推理加载的唯一路径
│       ├── checkpoints/        # 训练中间 checkpoint（不用于推理）
│       ├── logs/
│       │   └── final_training_metrics.csv
│       ├── results/
│       │   ├── final_model_results.json
│       │   └── test_results.json
│       └── label_mapping.csv
├── outputs/                    # 推理输出（按任务分子目录，自动创建）
│   └── <task_name>/
│       ├── <input_basename>_with_<task>.csv
│       ├── <task>_ckpt.csv     # 断点文件
│       └── <task>_stats.json
└── outputs_summary/            # 可选汇总目录
```

---

## 快速上手（完整流程）

### 第一步：克隆 / 下载项目

```bash
git clone https://github.com/Chunn20/violence-text-classifier.git
cd violence-text-classifier
```

### 第二步：安装依赖

建议使用虚拟环境：

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 第三步：放置本地模型

将 DeepSeek-7B base model 文件夹放在项目根目录，命名为 `deepseek-llm-7b-base`：

```
violence-text-classifier/
└── deepseek-llm-7b-base/       # ← 将模型文件夹放在这里
    ├── config.json
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── pytorch_model.bin (或 model.safetensors)
```

若模型放在其他位置，修改 `configs/tasks.yaml` 中的 `global.base_model_path` 为对应路径：

```yaml
global:
  base_model_path: "/absolute/path/to/your/deepseek-llm-7b-base"
```

### 第四步：放置 gold 数据文件

将 5 个 Excel 标注文件放在项目根目录（与 `main_train.py` 同级）：

```
violence-text-classifier/
├── split_author_gender_final8_1_1_bool.xlsx
├── split_victim_gender_final8_1_1_bool.xlsx
├── split_perpetrator_gender_final8_1_1_bool.xlsx
├── split_relationship_final8_1_1_bool.xlsx
└── split_Is the author the victim?_final8_1_1_bool.xlsx
```

每个文件需包含三个 sheet：`train` / `validation` / `test`，且都有 `new_text` 列和对应标签列。

### 第五步：训练模型

**所有命令在项目根目录（`violence-text-classifier/`）下运行。**

依次训练 5 个任务：

```bash
python main_train.py --task author_gender      --mode train
python main_train.py --task victim_gender      --mode train
python main_train.py --task perpetrator_gender --mode train
python main_train.py --task relationship       --mode train
python main_train.py --task author_is_victim   --mode train
```

训练完成后，各任务的微调模型自动保存到 `runs/<task_name>/saved_model/`。

### 第六步：推理

准备好输入 CSV 文件（需包含 `new_text` 列），然后运行：

```bash
python main_predict.py --task author_gender      --input data/reddit_2026.csv
python main_predict.py --task victim_gender      --input data/reddit_2026.csv
python main_predict.py --task perpetrator_gender --input data/reddit_2026.csv
python main_predict.py --task relationship       --input data/reddit_2026.csv
python main_predict.py --task author_is_victim   --input data/reddit_2026.csv
```

输出结果保存到 `outputs/<task_name>/`，文件名格式为 `<输入文件名>_with_<task>.csv`。

---

## 环境配置

```bash
pip install -r requirements.txt
```

**requirements.txt 依赖：**

```
torch>=1.10.0
numpy>=1.21.0
pandas>=1.5.0
scikit-learn>=1.0.0
transformers>=4.30.0
datasets>=2.0.0
openpyxl>=3.0.0
tqdm>=4.60.0
pyyaml>=6.0
```

---

## 配置说明（configs/tasks.yaml）

```yaml
global:
  base_model_path: "./deepseek-llm-7b-base"   # 修改为你的本地 base model 路径
  text_col: "new_text"                          # 输入文本列，训练/推理统一使用
  max_length: 128
  seed: 42

tasks:
  author_gender:
    saved_model_dir: "runs/author_gender/saved_model"  # 训练保存 & 推理加载的唯一路径
    ...
```

**关键设计：`saved_model_dir` 是唯一路径来源**
训练脚本将模型保存到此路径，推理脚本从此路径加载，两者完全一致。

---

## 训练

> 所有命令需在项目根目录（`violence-text-classifier/`）下运行。

### 最终训练（使用 yaml 中已记录的 best_params）

```bash
python main_train.py --task author_gender      --mode train
python main_train.py --task victim_gender      --mode train
python main_train.py --task perpetrator_gender --mode train
python main_train.py --task relationship       --mode train
python main_train.py --task author_is_victim   --mode train
```

训练完成后模型保存到 `runs/<task_name>/saved_model/`，之后推理直接从这里加载。

### 覆盖超参数（可选）

如需临时调整超参数而不修改 yaml，使用 `--lr / --wd / --epochs`：

```bash
python main_train.py --task author_gender --mode train --lr 5e-6 --epochs 4 --wd 0.01
```

### 参数搜索（仅 author_gender）

如果想重新搜索 `author_gender` 的最优超参数：

```bash
python main_train.py --task author_gender --mode search
```

搜索分两阶段自动运行：
1. 快速筛选：lr × wd 组合，固定 epochs=2
2. 精细调整：在第一阶段最优值附近细化，epochs=[3,4]

搜索结束后终端会打印建议的 `best_params`，将其手动填入 `configs/tasks.yaml` 的 `author_gender.best_params`，再运行 `--mode train`。

### 训练输出说明

```
runs/author_gender/
├── saved_model/                    # 最终微调模型（推理直接从这里加载）
├── checkpoints/                    # 训练中间 checkpoint（可忽略）
├── logs/
│   └── final_training_metrics.csv  # 每 epoch 的 loss / accuracy / f1
├── results/
│   ├── final_model_results.json    # 训练参数快照 + eval 指标
│   └── test_results.json           # 独立 test sheet 的评估结果
└── label_mapping.csv               # int ↔ 标签字符串 映射
```

---

## 推理

> 运行推理前，必须先完成对应任务的训练（模型需存在于 `runs/<task>/saved_model/`）。

### 基本用法

```bash
python main_predict.py --task <任务名> --input <输入CSV路径>
```

示例：

```bash
python main_predict.py --task author_gender      --input data/reddit_2026.csv
python main_predict.py --task victim_gender      --input data/reddit_2026.csv
python main_predict.py --task perpetrator_gender --input data/reddit_2026.csv
python main_predict.py --task relationship       --input data/reddit_2026.csv
python main_predict.py --task author_is_victim   --input data/reddit_2026.csv
```

### 可选参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--output_dir` | `outputs/<task>/` | 指定推理结果保存目录 |
| `--save_every` | `2000` | 每处理 N 条保存一次断点 |
| `--config` | `configs/tasks.yaml` | 指定配置文件路径 |

示例：

```bash
python main_predict.py --task relationship --input data/reddit_2026.csv \
    --output_dir outputs/rel_run2/ --save_every 500
```

### 推理输出说明

每次推理生成 3 个文件：

| 文件 | 说明 |
|---|---|
| `<输入名>_with_<task>.csv` | 在原始 CSV 上追加预测列的最终结果 |
| `<task>_ckpt.csv` | 断点文件（中断后用于续跑，完成后可删除） |
| `<task>_stats.json` | 预测分布统计（置信度分布、各标签占比等） |

### 输出新增列名（与原 notebook 完全一致）

| 任务 | pred_col | conf_col | prob_cols |
|---|---|---|---|
| author_gender | `author_gender_finetune` | `author_gender_conf` | `author_gender_prob_female`, `_prob_male`, `_prob_unknown` |
| victim_gender | `victim_gender_finetune` | `victim_gender_conf` | `victim_gender_prob_female`, `_prob_female_male`, `_prob_male`, `_prob_unknown` |
| perpetrator_gender | `perpetrator_gender_finetune` | `perpetrator_gender_conf` | `perpetrator_gender_prob_female`, `_prob_female_male`, `_prob_male`, `_prob_unknown` |
| relationship | `relationship_finetune` | `relationship_conf` | `relationship_prob_dad`, `_prob_family`, `_prob_friend`, `_prob_lover`, `_prob_mom`, `_prob_other`, `_prob_parents`, `_prob_unknown` |
| author_is_victim | `author_is_victim_finetune` | `author_is_victim_conf` | `author_is_victim_prob_False`, `_prob_True` |

### 断点续跑

推理过程支持 `Ctrl+C` 安全中断，重新运行相同命令即可从断点继续：

```bash
# 第一次运行（中途中断）
python main_predict.py --task author_gender --input data/reddit_2026.csv

# 重新运行，自动从断点继续
python main_predict.py --task author_gender --input data/reddit_2026.csv
```

---

## 任务说明

| 任务 key | 标签列 | 类别数 | 标签 |
|---|---|---|---|
| `author_gender` | `author_gender_final` | 3 | female / male / unknown |
| `victim_gender` | `victim_gender_final` | 4 | female / female & male / male / unknown |
| `perpetrator_gender` | `perpetrator_gender_final` | 4 | female / female & male / male / unknown |
| `relationship` | `relationship_final` | 8 | dad / family / friend / lover / mom / other / parents / unknown |
| `author_is_victim` | `Is the author the victim?_final` | 2 | False / True |

---

## 从旧 Notebook 迁移说明

| 原始 Notebook | 迁移到 |
|---|---|
| `Try-DS-fine-tuning-v8_authorGender.ipynb` | `main_train.py --task author_gender --mode search` |
| `Train final model-20260101.ipynb`（各任务 section） | `main_train.py --task <任务名> --mode train` |
| `3.predict_author_gender.ipynb` | `main_predict.py --task author_gender` |
| `4.predict_victim_gender.ipynb` | `main_predict.py --task victim_gender` |
| `5.predict_perpetrator_gender.ipynb` | `main_predict.py --task perpetrator_gender` |
| `6.predict_author_is_victim.ipynb` | `main_predict.py --task author_is_victim` |
| `7.predict_relationship.ipynb` | `main_predict.py --task relationship` |

**原有逻辑均已完整保留，未改动研究设计。**

---

## 可复现性保障

- 固定随机种子：`seed=42`（Python / NumPy / PyTorch）
- 固定数据切分：`Dataset.train_test_split(test_size=0.1, seed=42)`
- 保存训练参数快照：`runs/<task>/results/final_model_results.json`
- 保存 label mapping：`runs/<task>/label_mapping.csv`
- 保存训练日志：`runs/<task>/logs/final_training_metrics.csv`
- 强制离线模式：`TRANSFORMERS_OFFLINE=1`，`local_files_only=True`
