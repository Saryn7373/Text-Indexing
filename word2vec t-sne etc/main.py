"""Лабораторная работа 1. word2vec: подготовка корпуса, обучение моделей
Skip-gram/CBOW, загрузка предобученной модели, поиск похожих слов, решение
аналогии и визуализация векторов на 2D-плоскости.

Корпус - английский перевод "Война и мир" (data/war_peace_plain.txt).
"""
from pathlib import Path

from tokenization import preprocess_corpus
from word2vec import train_models, load_pretrained
from similarity import most_similar_across_models, solve_analogy
from visualization import get_word_vectors, reduce_to_2d, plot_words_2d

DATA_DIR = Path(__file__).resolve().parent / "data"
CORPUS_PATH = DATA_DIR / "war_peace_plain.txt"

# Параметры обучения (подобраны под небольшой корпус в одну книгу)
VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 5

# 5 ключевых слов для поиска похожих терминов
KEYWORDS = ["war", "peace", "napoleon", "moscow", "love"]

# Аналогия: king - man + woman = ? (ожидаемо queen)
ANALOGY_POSITIVE = ["king", "woman"]
ANALOGY_NEGATIVE = ["man"]

# 20 слов для визуализации, разделённых на две темы
GEO_WORDS = ["russia", "france", "austria", "italy", "poland",
             "england", "turkey", "moscow", "paris"]
TITLE_WORDS = ["general", "soldier", "officer", "doctor", "priest",
               "count", "prince", "emperor", "colonel", "captain"]
VIS_WORDS = GEO_WORDS + TITLE_WORDS
VIS_GROUP_BY_WORD = {w: "страны/города" for w in GEO_WORDS}
VIS_GROUP_BY_WORD.update({w: "титулы/профессии" for w in TITLE_WORDS})


def load_corpus(path):
    """Проверка: файл корпуса должен существовать и быть непустым."""
    if not path.exists():
        raise FileNotFoundError(f"Файл корпуса не найден: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Файл корпуса пуст: {path}")
    return text


def print_similarity_section(models):
    print("\n=== Похожие слова по ключевым словам (косинусное расстояние) ===")
    for word in KEYWORDS:
        print(f"\n--- '{word}' ---")
        results = most_similar_across_models(word, models, topn=10)
        for name, res in results.items():
            if res is None:
                continue
            top = ", ".join(f"{w} ({score:.2f})" for w, score in res[:5])
            print(f"[{name}] {top}")


def print_analogy_section(models):
    print(f"\n=== Аналогия: {ANALOGY_POSITIVE[0]} - {ANALOGY_NEGATIVE[0]} "
          f"+ {ANALOGY_POSITIVE[1]} = ? ===")
    for name, model in models.items():
        result = solve_analogy(model, ANALOGY_POSITIVE, ANALOGY_NEGATIVE, topn=1)
        if result:
            word, score = result[0]
            print(f"[{name}] -> {word} ({score:.2f})")


def visualize_models(models):
    print("\n=== Визуализация векторов (t-SNE, 2D-плоскость) ===")
    for name, model in models.items():
        words, vectors = get_word_vectors(model, VIS_WORDS)
        # Проверка: для t-SNE нужно хотя бы несколько точек
        if len(words) < 3:
            print(f"[{name}] недостаточно слов в словаре модели "
                  f"({len(words)} из {len(VIS_WORDS)}) - визуализация пропущена")
            continue
        groups = [VIS_GROUP_BY_WORD[w] for w in words]
        vectors_2d = reduce_to_2d(vectors, perplexity=min(5, len(words) - 1))
        plot_words_2d(
            words, vectors_2d, groups=groups,
            title=f"{name}: страны/города vs титулы/профессии (t-SNE)",
        )


def main():
    # --- 1. Подготовка корпуса --------------------------------------------
    print("=== Загрузка и предобработка корпуса ===")
    text = load_corpus(CORPUS_PATH)
    sentences = preprocess_corpus(text)
    flat_tokens = [token for sentence in sentences for token in sentence]
    print(f"Предложений: {len(sentences)}, токенов всего: {len(flat_tokens)}, "
          f"уникальных слов: {len(set(flat_tokens))}")
    if not sentences:
        raise RuntimeError("Корпус пуст после предобработки - проверьте файл данных")

    # --- 2. Обучение собственных моделей ------------------------------------
    print("\n=== Обучение моделей Skip-gram и CBOW ===")
    model_sg, model_cbow = train_models(
        sentences, vector_size=VECTOR_SIZE, window=WINDOW, min_count=MIN_COUNT,
    )
    print(f"Skip-gram: словарь {len(model_sg.wv.key_to_index)} слов")
    print(f"CBOW: словарь {len(model_cbow.wv.key_to_index)} слов")
    # Проверка: обучение не должно давать пустой словарь
    if len(model_sg.wv.key_to_index) == 0 or len(model_cbow.wv.key_to_index) == 0:
        raise RuntimeError("Словарь обученной модели пуст - уменьшите min_count")

    # --- 3. Загрузка предобученной модели -----------------------------------
    print("\n=== Загрузка предобученной модели word2vec-google-news-300 ===")
    pretrained_model = load_pretrained("word2vec-google-news-300")
    print(f"Pretrained: словарь {len(pretrained_model.key_to_index)} слов")

    models = {
        "skip-gram": model_sg,
        "cbow": model_cbow,
        "pretrained": pretrained_model,
    }

    # --- 4. Поиск похожих слов -----------------------------------------------
    print_similarity_section(models)

    # --- 5. Решение аналогии --------------------------------------------------
    print_analogy_section(models)

    # --- 6. Визуализация 2D -----------------------------------------------------
    visualize_models(models)


if __name__ == "__main__":
    main()
