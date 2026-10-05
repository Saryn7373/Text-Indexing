"""Лемматизация слов и двусловий по тезаурусу."""
import pymorphy3

_morph = pymorphy3.MorphAnalyzer()


def lemmatize_word(word, name_to_lemma):
    """Лемма слова: сначала морфологический разбор (тезаурус хранит только
    начальные формы), затем нормализованная форма ищется в тезаурусе."""
    normal = _morph.parse(word.lower())[0].normal_form.upper().replace("Ё", "Е")
    return name_to_lemma.get(normal, normal)


def extract_lemmas(words, bigrams, name_to_lemma):
    """Множество лемм запроса (дубли удаляются).

    Для двусловия берётся лемма из тезауруса; если её нет - леммы
    каждого отдельного слова (лемматизируются слова, не покрытые двусловиями).
    """
    lemmas = set()
    covered = set()
    for i, bigram in enumerate(bigrams):
        if bigram in name_to_lemma:
            lemmas.update(name_to_lemma[bigram].split())
            covered.update((i, i + 1))
    for i, word in enumerate(words):
        if i not in covered:
            lemmas.add(lemmatize_word(word, name_to_lemma))
    return lemmas
