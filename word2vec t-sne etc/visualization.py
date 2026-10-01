"""Снижение размерности векторов слов до 2D (t-SNE) и визуализация на плоскости."""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def get_word_vectors(model, words):
    """Отбирает из модели только те слова, что есть в её словаре (разные
    модели обучены на разных корпусах, поэтому список доступных слов может
    отличаться). Возвращает (доступные_слова, матрица_векторов)."""
    wv = model.wv if hasattr(model, "wv") else model
    available = [w for w in words if w in wv.key_to_index]
    missing = [w for w in words if w not in wv.key_to_index]
    if missing:
        print(f"Нет в словаре модели: {missing}")
    vectors = np.array([wv[w] for w in available])
    return available, vectors


def reduce_to_2d(vectors, random_state=42, perplexity=5):
    """t-SNE: нелинейное снижение размерности векторов до 2D для визуализации.
    perplexity должен быть меньше числа точек - при малом наборе слов он
    автоматически уменьшается, иначе sklearn выбросит ошибку."""
    n_samples = len(vectors)
    perplexity = max(1, min(perplexity, n_samples - 1))
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, init="pca")
    return tsne.fit_transform(vectors)


def plot_words_2d(words, vectors_2d, groups=None, title=""):
    """Scatter-plot слов на 2D-плоскости с подписями.
    groups - опциональный список меток тем (для раскраски точек по группам)."""
    plt.figure(figsize=(10, 8))
    if groups is not None:
        unique_groups = sorted(set(groups))
        colors = plt.cm.tab10.colors
        for i, group in enumerate(unique_groups):
            idx = [j for j, g in enumerate(groups) if g == group]
            plt.scatter(
                vectors_2d[idx, 0], vectors_2d[idx, 1],
                label=group, color=colors[i % len(colors)],
            )
        plt.legend()
    else:
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1])

    for i, word in enumerate(words):
        plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]))

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
