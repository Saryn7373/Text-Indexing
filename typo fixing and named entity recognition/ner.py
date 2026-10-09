"""Распознавание сущностей: справочник с правилами и готовая модель Natasha.

Справочник: формы названий сравниваются по токенам без учёта регистра (ё = е), выбирается
самое длинное совпадение. Контекстное правило: неизвестное слово с заглавной буквы после
маркера («город», «профессор», «компания»...) получает тип маркера. Ограничения правила:
нужен маркер; срабатывает на любое неизвестное слово с заглавной буквы; не отличает
опечатку от нового имени.

Модель: slovnet_ner_news_v1 из пакета natasha 1.6.0 (https://github.com/natasha/natasha),
CPU, веса входят в пакет. Метки PER -> PERSON, ORG -> ORG, LOC -> LOC; других модель не выдаёт.
"""
from correction import normalize_word, tokenize

TRIGGERS = {  # основы слов-маркеров и максимальная длина сущности после маркера
    "PERSON": (("профессор", "академик", "доктор", "писател", "поэт", "художник",
                "композитор", "физик", "химик", "режиссер", "господин", "конструктор"), 2),
    "LOC": (("город", "реке", "река", "реку", "озер", "селе", "деревн", "поселк",
             "поселен", "станици"), 1),
    "ORG": (("компани", "корпораци", "фирм", "банк", "холдинг", "завод", "агентств"), 3),
}
MODEL_LABELS = {"PER": "PERSON", "ORG": "ORG", "LOC": "LOC"}


class DictionaryRecognizer:
    name = "dictionary"

    def __init__(self, entity_dict, known_words):
        self.known_words = known_words
        self.index = {}
        for entry in entity_dict:
            for form in entry["forms"]:
                key = tuple(normalize_word(t) for _, _, t in tokenize(form))
                self.index.setdefault(key, entry["type"])
        self.max_len = max(map(len, self.index))
        self.single = {k[0] for k in self.index if len(k) == 1}

    def recognize(self, text):
        tokens = tokenize(text)
        norm = [normalize_word(t) for _, _, t in tokens]
        entities, i = [], 0
        while i < len(tokens):
            match = self._dictionary(norm, i) or self._context(tokens, norm, i)
            if not match:
                i += 1
                continue
            length, etype = match
            start, end = tokens[i][0], tokens[i + length - 1][1]
            entities.append({"start": start, "end": end, "text": text[start:end], "type": etype})
            i += length
        return entities

    def _dictionary(self, norm, i):
        for n in range(min(self.max_len, len(norm) - i), 0, -1):
            if tuple(norm[i:i + n]) in self.index:
                return n, self.index[tuple(norm[i:i + n])]
        return None

    def _context(self, tokens, norm, i):
        if i == 0:
            return None
        for etype, (stems, max_words) in TRIGGERS.items():
            if norm[i - 1].startswith(stems):
                n = 0
                while n < max_words and i + n < len(tokens) and tokens[i + n][2][:1].isupper() \
                        and norm[i + n] not in self.known_words and norm[i + n] not in self.single:
                    n += 1
                if n:
                    return n, etype
        return None


class ModelRecognizer:
    name = "model"

    def __init__(self):
        try:
            from natasha import Doc, NewsEmbedding, NewsNERTagger, Segmenter
            self._doc, self._segmenter = Doc, Segmenter()
            self._tagger = NewsNERTagger(NewsEmbedding())
        except ImportError:
            raise SystemExit("Не установлена natasha: pip install -r requirements.txt")
        except Exception as e:
            raise SystemExit(f"Не удалось загрузить модель NER: {e}")

    def recognize(self, text):
        if not text.strip():
            return []
        doc = self._doc(text)
        doc.segment(self._segmenter)
        doc.tag_ner(self._tagger)
        return [{"start": s.start, "end": s.stop, "text": s.text, "type": MODEL_LABELS[s.type]}
                for s in doc.spans if s.type in MODEL_LABELS]


def make_recognizer(name, res):
    if name == "dictionary":
        return DictionaryRecognizer(res.entity_dict, res.word_freq)
    return ModelRecognizer()
