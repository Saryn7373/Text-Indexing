"""Поиск похожих слов и решение словесных аналогий по нескольким моделям."""


def get_wv(model):
    """Возвращает KeyedVectors независимо от типа модели: у Word2Vec это
    model.wv, у готовой (уже загруженной) модели - сам объект."""
    return model.wv if hasattr(model, "wv") else model


def in_vocab(model, word):
    return word in get_wv(model).key_to_index


def most_similar_across_models(word, models, topn=10):
    """Для слова ищет ближайшие по косинусному расстоянию термины в каждой
    из моделей. models - словарь {название_модели: модель}.

    Возвращает {название_модели: список (слово, схожесть) или None, если
    слово отсутствует в словаре этой модели}."""
    results = {}
    for name, model in models.items():
        wv = get_wv(model)
        if word not in wv.key_to_index:
            print(f"[{name}] слово '{word}' отсутствует в словаре модели - пропуск")
            results[name] = None
            continue
        results[name] = wv.most_similar(word, topn=topn)
    return results


def solve_analogy(model, positive, negative, topn=1):
    """Решает аналогию positive[0] - negative[0] + positive[1] = ?
    (например king - man + woman = queen).

    Возвращает None, если хотя бы одно слово отсутствует в словаре модели."""
    wv = get_wv(model)
    missing = [w for w in positive + negative if w not in wv.key_to_index]
    if missing:
        print(f"Нет в словаре модели: {missing} - аналогию решить нельзя")
        return None
    return wv.most_similar(positive=positive, negative=negative, topn=topn)
