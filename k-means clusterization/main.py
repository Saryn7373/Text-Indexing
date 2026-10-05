"""Практическая работа 3. Кластеризация K-means.

PDF-статьи лежат в папках data/computer_vision (тема A) и data/astrophysics (тема B).
"""
import hashlib
import re
import warnings
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pdfminer.high_level import extract_text as pdfminer_extract_text
from pypdf import PdfReader
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from unidecode import unidecode

DATA_DIR = Path(__file__).resolve().parent / "data"
TOPICS = [("computer_vision", "A"), ("astrophysics", "B")]


# --- Загрузка PDF и извлечение текста ---------------------------------------
def read_pdf_text(path: Path) -> str:
    """Извлечение текста."""
    try:
        txt = pdfminer_extract_text(str(path))
        if txt and len(txt.strip()) > 100:
            return txt
    except Exception:
        pass
    try:
        reader = PdfReader(str(path))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n".join(pages)
    except Exception as e:
        warnings.warn(f"PDF read failed: {path.name} ({e})")
        return ""


def cut_references(text: str) -> str:
    """Удаляем список литературы (по первому заголовку после 40% текста)."""
    pattern = r"(?:\n|\r|\r\n)\s*(references|литература|список литературы|appendix)\b"
    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        if m.start() > len(text) * 0.4:
            return text[:m.start()]
    return text


def extract_title_abstract(text: str) -> Tuple[str, str]:
    """Базовое разделение статьи на логические блоки."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title = lines[0][:300] if lines else ""
    m = re.search(r"(abstract|аннотация)\s*[:\-\–]?\s*(.+?)(\n[A-Z][A-Za-z ]{3,}|$)",
                  text, flags=re.IGNORECASE | re.DOTALL)
    abstract = (m.group(2).strip() if m else "")[:3000]
    return title, abstract


def hash_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()


rows = []
for folder, lbl in TOPICS:
    for pdf in sorted((DATA_DIR / folder).glob("*.pdf")):
        raw = unidecode(read_pdf_text(pdf))  # нормализуем юникод к латинице
        raw = cut_references(raw)
        title, abstract = extract_title_abstract(raw)  # до схлопывания переводов строк
        raw = re.sub(r"\s+", " ", raw)
        if len(raw) < 500:  # отсеиваем совсем короткие/битые
            continue
        keep = title + "\n" + abstract + "\n" + raw[:15000]  # ограничим размер
        rows.append({"id": pdf.stem, "topic": lbl, "title": title, "text": keep})

df = pd.DataFrame(rows)
df["hash"] = df["text"].apply(hash_text)
df = df.drop_duplicates(subset="hash").drop(columns=["hash"]).reset_index(drop=True)

# Балансировка классов: одинаковое число статей по темам
n_min = df["topic"].value_counts().min()
df_bal = pd.concat([g.sample(n=n_min, random_state=42) for _, g in df.groupby("topic")]
                   ).reset_index(drop=True)
print(f"Docs: {len(df_bal)} | A={sum(df_bal.topic == 'A')} | B={sum(df_bal.topic == 'B')}")

# --- Предобработка ----------------------------------------------------------
RU_STOP = {"это", "также", "который", "которая", "которые", "данный", "далее", "например", "таким", "образом",
           "рисунок", "таблица", "метод", "работа", "результат", "вывод", "согласно", "проведен", "представлен"}
EN_STOP = {"the", "and", "for", "with", "from", "that", "this", "these", "those", "into", "based", "using",
           "results", "conclusion", "figure", "table", "method", "study", "paper", "approach", "analysis"}
COMMON_STOP = RU_STOP | EN_STOP | {"et", "al", "eg", "ie"}
REF_PAT = re.compile(r"\[\d{1,3}\]|\([^\)]*\d{4}[^\)]*\)")  # ссылки вида [12] или (Author, 2020)
URL_PAT = re.compile(r"https?://\S+|doi:\S+", re.IGNORECASE)


def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = URL_PAT.sub(" ", text)
    text = REF_PAT.sub(" ", text)
    text = re.sub(r"[^a-zа-яё0-9\- ]", " ", text)
    text = re.sub(r"\d+[\-–]\d+", " ", text)  # диапазоны цифр
    toks = [t for t in text.split() if len(t) > 2 and t not in COMMON_STOP]
    return [t for t in toks if not t.startswith("-") and not t.endswith("-")]


df_bal["text_clean"] = df_bal["text"].apply(lambda t: " ".join(simple_tokenize(t)))

# --- Векторизация -----------------------------------------------------------
tfidf = TfidfVectorizer(min_df=5, max_df=0.7, ngram_range=(1, 2), max_features=100_000)
X = tfidf.fit_transform(df_bal["text_clean"])


n_comp = min(200, X.shape[0] - 1, X.shape[1] - 1)
lsa = make_pipeline(TruncatedSVD(n_components=n_comp, random_state=42), Normalizer(copy=False))
Xm = lsa.fit_transform(X)

# --- K-means (две исходные темы -> k=2) -------------------------------------
k = 2
kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
labels_pred = kmeans.fit_predict(Xm)
df_bal["cluster"] = labels_pred

# --- Интерпретация: топ-20 терминов на кластер ------------------------------
terms = np.array(tfidf.get_feature_names_out())
for c in range(k):
    mean_vec = np.asarray(X[labels_pred == c].mean(axis=0)).ravel()
    print(f"Cluster {c}: {', '.join(terms[np.argsort(mean_vec)[::-1][:20]])}")

for c in range(k):
    print(f"\n=== Cluster {c} examples ===")
    for _, r in df_bal[df_bal["cluster"] == c].head(5).iterrows():
        print(f"[{r['id']}] {r['title'][:120]}")

# --- Визуализация -----------------------------------------------------------
pca2 = PCA(n_components=2, random_state=42).fit_transform(Xm)
plt.figure()
plt.scatter(pca2[:, 0], pca2[:, 1], c=labels_pred, s=30)
plt.title(f"K-means, k={k} (PCA 2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()
