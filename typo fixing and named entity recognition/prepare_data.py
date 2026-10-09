"""Подготовка данных: эталон запросов, словарь словоформ, справочник сущностей и их контроль.

Вход:  data/corpus.json, data/queries_markup.json (ручная разметка), data/entity_seed.json.
Выход: data/queries_dev.json, queries_test.json, queries_demo.json,
       word_dictionary.json, entity_dictionary.json. Буква «ё» везде заменяется на «е».
Запуск: python prepare_data.py

Нотация разметки: {TYPE|текст} - сущность (PERSON, ORG, LOC), <ошибка=верно> - опечатка,
опечатка может стоять внутри сущности: {LOC|<Маскве=Москве>}.
В entity_seed.json source = "corpus" (название есть в корпусе) или "dev" (только в запросах
для настройки); названий из проверочного набора там нет.
"""
from collections import Counter

import pymorphy3

from correction import normalize_word, replace_yo, tokenize
from storage import DATA_DIR, load_json, save_json

CASES = ["nomn", "gent", "datv", "accs", "loct", "ablt"]
GROUPS = ["clean", "word_typo", "entity_typo", "rare_name"]
morph = pymorphy3.MorphAnalyzer()


# ---------- разметка запросов ----------

def classify_typo(wrong, right):
    if len(wrong) != len(right):
        return "deletion" if len(wrong) < len(right) else "insertion"
    diff = [i for i in range(len(wrong)) if wrong[i] != right[i]]
    if len(diff) == 2 and diff[1] == diff[0] + 1 \
            and wrong[diff[0]] == right[diff[1]] and wrong[diff[1]] == right[diff[0]]:
        return "transposition"
    return "substitution"


def parse_markup(qid, group, markup):
    """Строит эталонную запись: тексты, опечатки и сущности с границами [start, end)."""
    original, correct = "", ""
    typos, orig_ents, corr_ents, stack = [], [], [], []
    i = 0
    while i < len(markup):
        ch = markup[i]
        if ch == "{":
            bar = markup.index("|", i)
            stack.append((markup[i + 1:bar], len(original), len(correct)))
            i = bar + 1
        elif ch == "}":
            etype, so, sc = stack.pop()
            orig_ents.append({"start": so, "end": len(original), "text": original[so:], "type": etype})
            corr_ents.append({"start": sc, "end": len(correct), "text": correct[sc:], "type": etype})
            i += 1
        elif ch == "<":
            end = markup.index(">", i)
            wrong, right = markup[i + 1:end].split("=")
            typos.append({"start": len(original), "end": len(original) + len(wrong),
                          "wrong": wrong, "correct": right, "kind": classify_typo(wrong, right)})
            original += wrong
            correct += right
            i = end + 1
        else:
            original += ch
            correct += ch
            i += 1
    return {"id": qid, "group": group, "original_text": original, "correct_text": correct,
            "typos": typos, "original_entities": sorted(orig_ents, key=lambda e: e["start"]),
            "correct_entities": sorted(corr_ents, key=lambda e: e["start"])}


# ---------- словарь словоформ ----------

def build_word_dictionary(corpus):
    """Словоформа (нижний регистр, ё->е) -> частота; леммы не используются."""
    freq = Counter(normalize_word(t) for doc in corpus for _, _, t in tokenize(doc["text"]))
    return dict(sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])))


# ---------- справочник сущностей ----------

def inflect(parse, grammemes, source):
    if "plur" in parse.tag:
        grammemes = (grammemes - {"sing"}) | {"plur"}
    form = parse.inflect(grammemes)
    word = form.word if form else parse.word
    return word[:1].upper() + word[1:] if source[:1].isupper() else word


def pick_parse(word, prefer):
    parses = morph.parse(word)
    return next((p for p in parses if prefer & set(p.tag.grammemes)), parses[0])


def generate_forms(name, etype):
    """Падежные формы (pymorphy3 как вспомогательный инструмент); прилагательные
    согласуются с последним словом; для PERSON добавляются формы фамилии."""
    words = name.split()
    if etype == "PERSON":
        parses = [pick_parse(w, {"Name", "Surn", "Patr"}) for w in words]
        forms = [" ".join(inflect(p, {c, "sing"}, w) for w, p in zip(words, parses)) for c in CASES]
        if len(words) > 1:
            forms += [inflect(parses[-1], {c, "sing"}, words[-1]) for c in CASES]
        return forms
    parses = [pick_parse(w, {"NOUN", "ADJF"}) for w in words]
    head = parses[-1]
    gender = next((g for g in ("masc", "femn", "neut") if g in head.tag), None)
    forms = []
    for case in CASES:
        if case == "accs" and gender in ("masc", "neut") and "inan" in head.tag:
            case = "nomn"
        forms.append(" ".join(
            inflect(p, {case, "sing"} | ({gender} if gender and "ADJF" in p.tag else set()), w)
            for w, p in zip(words, parses)))
    return forms


def build_entity_dictionary(seed):
    entries, counters = [], Counter()
    for item in seed:
        forms = [item["name"]]
        if item.get("inflect", True):
            forms += generate_forms(item["name"], item["type"])
        forms += item.get("manual", [])
        prefix = item["type"].lower()
        counters[prefix] += 1
        entries.append({"id": f"{prefix}_{counters[prefix]:03d}", "name": replace_yo(item["name"]),
                        "type": item["type"], "source": item["source"],
                        "forms": list(dict.fromkeys(map(replace_yo, forms)))})
    return entries


# ---------- контроль данных ----------

def validate(corpus, dev, test, entities):
    assert len(corpus) >= 20 and all(d["id"] and d["source"] for d in corpus)
    for rec in dev + test:
        for key, text in (("original_entities", rec["original_text"]),
                          ("correct_entities", rec["correct_text"])):
            ents = rec[key]
            assert all(text[e["start"]:e["end"]] == e["text"] for e in ents), rec["id"]
            assert all(a["end"] <= b["start"] for a, b in zip(ents, ents[1:])), rec["id"]
        assert all(rec["original_text"][t["start"]:t["end"]] == t["wrong"] for t in rec["typos"])
    texts = [normalize_word(r["original_text"]) for r in dev + test]
    assert len(set(texts)) == len(texts), "есть повторы запросов"
    for part in (dev, test):
        assert Counter(r["group"] for r in part) == Counter({g: 5 for g in GROUPS})
    types = Counter(e["type"] for r in dev + test for e in r["correct_entities"])
    kinds = Counter(t["kind"] for r in dev + test for t in r["typos"])
    assert set(types) == {"PERSON", "ORG", "LOC"} and len(kinds) == 4
    corpus_text = normalize_word(" ".join(d["text"] for d in corpus))
    dev_text = normalize_word(" ".join(r["original_text"] for r in dev))
    for e in entities:
        source = corpus_text if e["source"] == "corpus" else dev_text
        assert any(normalize_word(f) in source for f in e["forms"]), e["name"]
    return types, kinds


def main():
    corpus = load_json("corpus.json")
    markup = load_json("queries_markup.json")
    dev = [parse_markup(f"q{i + 1:02d}", g, m) for i, (g, m) in enumerate(markup["dev"])]
    test = [parse_markup(f"q{i + 21:02d}", g, m) for i, (g, m) in enumerate(markup["test"])]
    demo = [parse_markup(f"demo{i + 1:02d}", g, m) for i, (g, m) in enumerate(markup["demo"])]
    freq = build_word_dictionary(corpus)
    entities = build_entity_dictionary(load_json("entity_seed.json"))
    types, kinds = validate(corpus, dev, test, entities)

    for name, obj in (("queries_dev.json", dev), ("queries_test.json", test),
                      ("queries_demo.json", demo), ("word_dictionary.json", freq),
                      ("entity_dictionary.json", entities)):
        save_json(DATA_DIR / name, obj)
    print(f"документов: {len(corpus)}, словоформ: {len(freq)}, названий: {len(entities)}")
    print(f"запросов: настройка {len(dev)}, проверка {len(test)}, демо {len(demo)}")
    print("контроль пройден; типы сущностей:", dict(types), "виды опечаток:", dict(kinds))


if __name__ == "__main__":
    main()
