"""Обучение собственных word2vec-моделей (skip-gram, CBOW) и загрузка
готовой предобученной модели."""
from gensim.models import Word2Vec
import gensim.downloader as api


def train_models(sentences, vector_size=100, window=5, min_count=5):
    model_sg = Word2Vec(
        sentences=sentences, 
        vector_size=vector_size, # Размерность
        window=window,           # Контекстное окно
        min_count=min_count,     # Мин кол-во слов
        sg=1,                    # 1 - skipgram, 0 - cbow
        workers=4,               # Потоки
    )
    model_cbow = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=0,
        workers=4,
    )
    return model_sg, model_cbow


def load_pretrained(name="word2vec-google-news-300"):
    """Загружает готовую предобученную модель через gensim-data."""
    return api.load(name)
