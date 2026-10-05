"""Загрузка данных (CSV) и сохранение обученных весов сети (JSON)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
TRAIN_PATH = DATA_DIR / "mnist_train.csv"
TEST_PATH = DATA_DIR / "mnist_test.csv"
WEIGHTS_PATH = DATA_DIR / "weights.json"


def load_dataset(path):
    """CSV вида label + 784 столбца пикселей -> (метки, матрица пикселей 0..255)."""
    df = pd.read_csv(path)
    return df["label"].to_numpy(), df.drop(columns="label").to_numpy(dtype=np.uint8)


def save_weights(weights, meta, path=WEIGHTS_PATH):
    """Веса (последний - смещение) и параметры обучения в JSON."""
    path.write_text(json.dumps({**meta, "weights": weights}, ensure_ascii=False), encoding="utf-8")


def load_weights(path=WEIGHTS_PATH):
    """-> (веса, параметры обучения)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.pop("weights"), data
