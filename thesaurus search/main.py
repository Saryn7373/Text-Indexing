"""Лабораторная работа 2. Тезаурусный поиск (продвинутый уровень).

Запрос пользователя -> нормализация -> двусловия -> леммы (тезаурус senses.xml)
-> сравнение с таблицей запросов -> ответ.
"""
from lemmatization import extract_lemmas
from normalization import make_bigrams, normalize
from search import find_answer
from storage import load_queries, load_service_words, load_thesaurus

# Примеры для защиты: что и почему обрабатывается корректно
EXAMPLES = [
    "как пройти в библиотеку?",           # служебные слова отброшены, "библиотеку" -> БИБЛИОТЕКА
    "Пройти в библиотеку",                # то же множество лемм
    "когда работает столовая",            # "когда" - сервисное слово, остаются РАБОТАТЬ, СТОЛОВАЯ
    "где можно купить билеты?",           # множественное число -> лемма БИЛЕТ
    "мне нужно записаться к врачу",       # "врачу" -> ВРАЧ, служебные слова удалены
    "хочу получить справку",              # "справку" -> СПРАВКА
    "как сдать экзамены",                 # "экзамены" -> ЭКЗАМЕН
    "как построить космический корабль",  # нет в базе -> ответа нет
]


def answer_query(text, service_words, name_to_lemma, lemma_to_concepts, queries, verbose=False):
    words = normalize(text, service_words)
    bigrams = make_bigrams(words)
    lemmas = extract_lemmas(words, bigrams, name_to_lemma)
    if verbose:
        print(f"  нормализация: {' '.join(words)}")
        print(f"  двусловия:    {bigrams}")
        print(f"  леммы:        {sorted(lemmas)}")
    return find_answer(lemmas, queries, lemma_to_concepts)


def main():
    service_words = load_service_words()
    name_to_lemma, lemma_to_concepts = load_thesaurus()
    queries = load_queries()

    def ask(text):
        return answer_query(text, service_words, name_to_lemma, lemma_to_concepts,
                            queries, verbose=True) or "Ответ не найден."

    print("=== Примеры ===")
    for text in EXAMPLES:
        print(f"\nЗапрос: {text}")
        print(f"  Ответ: {ask(text)}")

    print("\n=== Интерактивный режим (пустая строка - выход) ===")
    while True:
        try:
            text = input("\nЗапрос: ").strip()
        except EOFError:
            break
        if not text:
            break
        print(f"  Ответ: {ask(text)}")


if __name__ == "__main__":
    main()
