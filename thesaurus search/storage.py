"""Загрузка данных: сервисные слова, тезаурус (senses.xml) и таблица запросов."""
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
SERVICE_WORDS_PATH = DATA_DIR / "service_words.xml"
SENSES_PATH = DATA_DIR / "senses.xml"
QUERIES_PATH = DATA_DIR / "queries.xml"


def load_service_words(path=SERVICE_WORDS_PATH):
    root = ET.parse(path).getroot()
    return {item.text.strip().upper() for item in root.iter("Item") if item.text}


def load_thesaurus(path=SENSES_PATH):
    """Возвращает (name_to_lemma, lemma_to_concepts).

    name_to_lemma: имя (слово или двусловие) -> лемма;
    lemma_to_concepts: лемма -> множество concept_id (смыслов) для поиска синонимов.
    """
    name_to_lemma = {}
    lemma_to_concepts = defaultdict(set)
    for _, item in ET.iterparse(path):
        if item.tag != "Item":
            continue
        name, lemma = item.get("name"), item.get("lemma")
        name_to_lemma.setdefault(name, lemma)
        lemma_to_concepts[lemma].add(item.get("concept_id"))
        item.clear()
    return name_to_lemma, lemma_to_concepts


def load_queries(path=QUERIES_PATH):
    """Таблица запросов: список (множество лемм, ответ)."""
    root = ET.parse(path).getroot()
    return [(set(i.get("lemmas").upper().split()), i.get("answer"))
            for i in root.iter("Item")]
