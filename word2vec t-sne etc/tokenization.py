"""Предобработка текста для корпуса (Лабораторная работа 1, word2vec).

Корпус - англоязычный текст ("Война и мир" в переводе), поэтому стоп-слова и
токенизация настроены на английский язык.
"""
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from unidecode import unidecode

# nltk.download('punkt')      # Раскомментировать при первом запуске
# nltk.download('punkt_tab')  # Раскомментировать при первом запуске
# nltk.download('stopwords')  # Раскомментировать при первом запуске

STOPWORDS_EN = set(stopwords.words('english'))


def _clean_tokens(tokens):
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in STOPWORDS_EN]
    return tokens


def preprocess_text(text):
    """Токенизация текста в один плоский список слов: нижний регистр,
    без пунктуации и стоп-слов. unidecode снимает диакритику (Natásha -> natasha),
    чтобы слова совпадали с написанием в словаре предобученной модели."""
    text = unidecode(text)
    tokens = word_tokenize(text)
    return _clean_tokens(tokens)


def preprocess_corpus(text):
    """Токенизация с сохранением границ предложений: список предложений,
    каждое - список токенов. Нужен для обучения word2vec - без этого
    контекстное окно "перетекало" бы через точки и границы главы."""
    text = unidecode(text)
    sentences = []
    for sent in sent_tokenize(text):
        tokens = _clean_tokens(word_tokenize(sent))
        if tokens:
            sentences.append(tokens)
    return sentences


if __name__ == "__main__":
    with open(Path(__file__).parent / 'data' / 'test.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    print(preprocess_text(text))
    print(preprocess_corpus(text))
