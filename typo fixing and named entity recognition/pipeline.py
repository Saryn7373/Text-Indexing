"""Конвейеры обработки запроса.

A: коррекция по общему словарю -> распознавание в исправленном тексте.
B: распознавание в исходном тексте -> защита найденных диапазонов от замен ->
   коррекция остальных слов -> повторное распознавание в исправленном тексте.
   Для неизвестных слов внутри защищённых диапазонов выдаются рекомендации из слов
   справочника сущностей; автоматически они не применяются.
"""
from correction import (apply_case, correct_text, find_candidates, is_checkable,
                        normalize_word, replace_yo, tokenize)
from storage import load_json

DEFAULT_CONFIG = {"threshold": 1, "use_freq": True, "protect": True}


class Resources:
    """Общий словарь, справочник и словарь слов из справочника."""

    def __init__(self):
        self.word_freq = load_json("word_dictionary.json")
        self.entity_dict = load_json("entity_dictionary.json")
        self.entity_freq = {}
        for entry in self.entity_dict:
            for form in entry["forms"]:
                for _, _, token in tokenize(form):
                    key = normalize_word(token)
                    self.entity_freq[key] = self.entity_freq.get(key, 0) + 1


def recommend(text, protected, res, config):
    result = []
    for start, end, token in tokenize(text):
        key = normalize_word(token)
        if any(s <= start and end <= e for s, e in protected) and is_checkable(token) \
                and key not in res.word_freq and key not in res.entity_freq:
            found = find_candidates(token, res.entity_freq, config["threshold"], config["use_freq"])
            if found:
                result.append({"start": start, "end": end, "original": token,
                               "candidates": [apply_case(token, c["word"]) for c in found]})
    return result


def process(text, recognizer, order, res, config=None):
    """order: "A", "B" или None (без исправлений). Позиции entities - в corrected_text."""
    config = {**DEFAULT_CONFIG, **(config or {})}
    text = replace_yo(text)  # «ё» читается как «е»; длина текста не меняется
    out = {"original_text": text, "order": order, "recognizer": recognizer.name,
           "entities_original": None, "entity_recommendations": []}
    if order is None:
        out.update(corrected_text=text, changes=[], entities=recognizer.recognize(text))
        return out
    protected = []
    if order == "B":
        out["entities_original"] = recognizer.recognize(text)
        if config["protect"]:
            protected = [(e["start"], e["end"]) for e in out["entities_original"]]
            out["entity_recommendations"] = recommend(text, protected, res, config)
    corrected, changes = correct_text(text, res.word_freq, config["threshold"], config["use_freq"], protected)
    out.update(corrected_text=corrected, changes=changes, entities=recognizer.recognize(corrected))
    return out
