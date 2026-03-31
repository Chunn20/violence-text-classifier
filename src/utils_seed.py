# src/utils_seed.py
"""固定随机种子，确保实验可复现。"""

import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """固定 Python / NumPy / PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"[Seed] 随机种子已固定为 {seed}")
