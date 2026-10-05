"""Тезаурусный поиск ответа по множеству лемм."""


def concepts_of(lemma, lemma_to_concepts):
    """Смыслы леммы; у возвратных глаголов (-ся) добавляются смыслы базового глагола."""
    concepts = set(lemma_to_concepts.get(lemma, set()))
    if lemma.endswith("СЯ"):
        concepts |= lemma_to_concepts.get(lemma[:-2], set())
    return concepts


def lemmas_match(a, b, lemma_to_concepts):
    """Леммы совпадают или являются синонимами (общий смысл в тезаурусе)."""
    return a == b or bool(concepts_of(a, lemma_to_concepts) & concepts_of(b, lemma_to_concepts))


def covers(source, target, lemma_to_concepts):
    """Каждая лемма из source имеет пару в target."""
    return all(any(lemmas_match(s, t, lemma_to_concepts) for t in target) for s in source)


def find_answer(user_lemmas, queries, lemma_to_concepts):
    """Ответ на запрос или None. Сначала точное совпадение множеств лемм,
    затем совпадение с учётом синонимов тезауруса."""
    for lemmas, answer in queries:
        if lemmas == user_lemmas:
            return answer
    for lemmas, answer in queries:
        if (covers(lemmas, user_lemmas, lemma_to_concepts)
                and covers(user_lemmas, lemmas, lemma_to_concepts)):
            return answer
    return None
