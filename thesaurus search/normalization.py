"""Нормализация запроса и выделение двусловий."""
import re

_NON_WORD = re.compile(r"[^\w\s-]|_")


def normalize(text, service_words):
    """Верхний регистр, удаление пунктуации и сервисных слов."""
    text = _NON_WORD.sub(" ", text.upper().replace("Ё", "Е"))
    return [w for w in text.split() if w not in service_words]


def make_bigrams(words):
    """Двусловия из соседних слов: [a, b, c] -> ["a b", "b c"]."""
    return [f"{a} {b}" for a, b in zip(words, words[1:])]
