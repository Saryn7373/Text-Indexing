"""Оценка и эксперименты.

    python evaluation.py tune    # сравнение параметров на наборе для настройки -> results/chosen_config.json
    python evaluation.py final   # единственный итоговый прогон на проверочном наборе
    python evaluation.py errors  # разбор ошибок по итоговому прогону -> results/error_analysis.md

Метрики коррекции: доля точно исправленных опечаток (top-1; top-3 - отдельно) и доля
изменённых корректных токенов. Сущности: precision, recall, F1 по точному совпадению границ
и типа; TP/FP/FN суммируются по набору; при нулевом знаменателе метрика равна 0.
Эталонные границы исходного запроса переносятся в обработанный текст по журналу замен;
если граница попадает внутрь заменённого токена, сущность считается пропущенной (FN),
а случай записывается в unmapped.
"""
import sys

from correction import normalize_word, tokenize
from ner import make_recognizer
from pipeline import DEFAULT_CONFIG, Resources, process
from storage import RESULTS_DIR, load_json, save_csv, save_json

TYPES = ("PERSON", "ORG", "LOC")
COMBOS = [(r, o) for r in ("dictionary", "model") for o in ("A", "B")]
FIELDS = ["condition", "method", "changes", "fix_rate", "top3_rate", "false_change_rate",
          "precision", "recall", "f1", "tp", "fp", "fn"]
CONFIG_PATH = RESULTS_DIR / "chosen_config.json"


def map_position(pos, changes):
    shift = 0
    for ch in changes:
        if ch["end"] <= pos:
            shift += len(ch["replacement"]) - (ch["end"] - ch["start"])
        elif ch["start"] < pos:
            return None
    return pos + shift


def div(a, b):
    return a / b if b else 0.0


def prf(tp, fp, fn):
    p, r = div(tp, tp + fp), div(tp, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": div(2 * p * r, p + r)}


def score_record(rec, result):
    """Счётчики коррекции и TP/FP/FN по типам для одного запроса."""
    changes = result["changes"]
    by_range = {(c["start"], c["end"]): c for c in changes}
    typo_ranges = {(t["start"], t["end"]) for t in rec["typos"]}
    fixed = top3 = 0
    for t in rec["typos"]:
        c = by_range.get((t["start"], t["end"]))
        if c:
            fixed += normalize_word(c["replacement"]) == normalize_word(t["correct"])
            top3 += any(normalize_word(a) == normalize_word(t["correct"]) for a in c["alternatives"])
    tokens = [(s, e) for s, e, _ in tokenize(rec["original_text"]) if (s, e) not in typo_ranges]
    typo = {"typos": len(rec["typos"]), "fixed": fixed, "top3": top3,
            "correct_tokens": len(tokens), "damaged": sum(t in by_range for t in tokens)}

    stats = {t: {"tp": 0, "fp": 0, "fn": 0} for t in TYPES}
    pred = {(e["start"], e["end"], e["type"]) for e in result["entities"]}
    matched, unmapped = set(), []
    for e in rec["original_entities"]:
        start, end = map_position(e["start"], changes), map_position(e["end"], changes)
        if start is None or end is None:
            unmapped.append({"id": rec["id"], "entity": e["text"],
                             "reason": "граница внутри заменённого токена"})
        gold = (start, end, e["type"])
        if None not in gold and gold in pred:
            stats[e["type"]]["tp"] += 1
            matched.add(gold)
        else:
            stats[e["type"]]["fn"] += 1
    for p in pred - matched:
        stats[p[2]]["fp"] += 1
    return typo, stats, unmapped


def run(records, recognizer, order, res, config=None):
    details, typo_sum, unmapped = [], {}, []
    stats_sum = {t: {"tp": 0, "fp": 0, "fn": 0} for t in TYPES}
    for rec in records:
        result = process(rec["original_text"], recognizer, order, res, config)
        typo, stats, um = score_record(rec, result)
        for k, v in typo.items():
            typo_sum[k] = typo_sum.get(k, 0) + v
        for t in TYPES:
            for k in stats[t]:
                stats_sum[t][k] += stats[t][k]
        unmapped += um
        details.append({"id": rec["id"], "group": rec["group"], "result": result,
                        "tp": sum(s["tp"] for s in stats.values())})
    total = {k: sum(stats_sum[t][k] for t in TYPES) for k in ("tp", "fp", "fn")}
    return {"typo": typo_sum, "entity": prf(**total),
            "per_type": {t: prf(**stats_sum[t]) for t in TYPES},
            "unmapped": unmapped, "details": details}


def summary_row(condition, method, r):
    t, e = r["typo"], r["entity"]
    row = {"condition": condition, "method": method,
           "changes": sum(len(d["result"]["changes"]) for d in r["details"]),
           "fix_rate": div(t["fixed"], t["typos"]), "top3_rate": div(t["top3"], t["typos"]),
           "false_change_rate": div(t["damaged"], t["correct_tokens"]),
           "precision": e["precision"], "recall": e["recall"], "f1": e["f1"],
           "tp": e["tp"], "fp": e["fp"], "fn": e["fn"]}
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}


def print_rows(rows):
    for r in rows:
        print(f"{r['condition']:<16}{r['method']:<22}исправлено {r['fix_rate']:.2f} "
              f"(top3 {r['top3_rate']:.2f}) повреждено {r['false_change_rate']:.3f} "
              f"замен {r['changes']:>2}  P {r['precision']:.2f} R {r['recall']:.2f} F1 {r['f1']:.2f}")


def tune():
    """Меняется один фактор относительно DEFAULT_CONFIG; выбор - по набору для настройки."""
    res, dev = Resources(), load_json("queries_dev.json")
    recognizers = {name: make_recognizer(name, res) for name in ("dictionary", "model")}
    rows, by_factor = [], {}
    for factor, values in (("threshold", [1, 2]), ("use_freq", [True, False]),
                           ("protect", [True, False])):
        for value in values:
            config = {**DEFAULT_CONFIG, factor: value}
            combos = [c for c in COMBOS if factor != "protect" or c[1] == "B"]
            part = [summary_row(f"{factor}={value}", f"{r}/{o}",
                                run(dev, recognizers[r], o, res, config)) for r, o in combos]
            by_factor.setdefault(factor, {})[value] = part
            rows += part

    def typo_score(part):  # коррекция не зависит от распознавателя в варианте A
        return part[0]["fix_rate"] - part[0]["false_change_rate"]

    chosen = {
        "threshold": max([1, 2], key=lambda v: (typo_score(by_factor["threshold"][v]), -v)),
        "use_freq": max([True, False], key=lambda v: (typo_score(by_factor["use_freq"][v]), v)),
        "protect": max([True, False], key=lambda v: (
            sum(r["f1"] for r in by_factor["protect"][v]), v)),
    }
    print_rows(rows)
    print("Выбранная конфигурация:", chosen)
    save_json(CONFIG_PATH, chosen)
    save_csv(RESULTS_DIR / "tuning_dev.csv", rows, FIELDS)


def final():
    if not CONFIG_PATH.exists():
        raise SystemExit("Сначала выполните: python evaluation.py tune")
    config = load_json(CONFIG_PATH)
    res, test = Resources(), load_json("queries_test.json")
    recognizers = {name: make_recognizer(name, res) for name in ("dictionary", "model")}
    runs = {}
    for r in recognizers:
        runs[f"{r}/baseline"] = run(test, recognizers[r], None, res)
    for r, o in COMBOS:
        runs[f"{r}/{o}"] = run(test, recognizers[r], o, res, config)
    rows = [summary_row("baseline" if k.endswith("baseline") else "final", k, v)
            for k, v in runs.items()]
    print("Конфигурация:", config)
    print_rows(rows)
    print("\nF1 по типам:")
    for k, v in runs.items():
        print(f"  {k:<20}" + "  ".join(f"{t} {v['per_type'][t]['f1']:.2f}" for t in TYPES))
    print("Непереносимых границ:", sum(len(v["unmapped"]) for v in runs.values()))
    save_csv(RESULTS_DIR / "final_test.csv", rows, FIELDS)
    save_json(RESULTS_DIR / "final_test_details.json", runs)


def describe(rec, d, note):
    r = d["result"]
    changes = "; ".join(f"[{c['start']}, {c['end']}) «{c['original']}» -> «{c['replacement']}» "
                        f"(альтернативы: {', '.join(c['alternatives'])})" for c in r["changes"])
    ents = "; ".join(f"«{e['text']}» {e['type']} [{e['start']}, {e['end']})" for e in r["entities"])
    gold = "; ".join(f"«{e['text']}» {e['type']}" for e in rec["original_entities"])
    return (f"{note}\n- `{rec['id']}` «{rec['original_text']}» -> «{r['corrected_text']}»\n"
            f"  - замены: {changes or 'нет'}\n  - сущности: {ents or 'нет'}\n  - эталон: {gold}\n")


def errors():
    """Примеры шести категорий из итогового прогона; категория 6 - демонстрационный запрос."""
    runs = load_json(RESULTS_DIR / "final_test_details.json")
    records = {r["id"]: r for r in load_json("queries_test.json")}
    found = {}

    def add(category, note, rec, d):
        if len(found.setdefault(category, [])) < 2:
            found[category].append(describe(rec, d, note))

    for r, o in COMBOS:
        key = f"{r}/{o}"
        for base, d in zip(runs[f"{r}/baseline"]["details"], runs[key]["details"]):
            rec = records[d["id"]]
            typo_ranges = {(t["start"], t["end"]) for t in rec["typos"]}
            changes = d["result"]["changes"]
            if d["tp"] > base["tp"]:
                add("1. Коррекция помогла распознаванию", f"**{key}**: без исправлений "
                    "сущность не находилась, после - найдена.", rec, d)
            for c in changes:
                inside = [e for e in rec["original_entities"]
                          if c["start"] < e["end"] and e["start"] < c["end"]]
                if inside and (c["start"], c["end"]) not in typo_ranges:
                    add("2. Название испорчено", f"**{key}**: корректное «{c['original']}» "
                        f"внутри «{inside[0]['text']}» заменено.", rec, d)
                if c["ambiguous"]:
                    add("5. Исправление неоднозначно", f"**{key}**: для «{c['original']}» "
                        "несколько кандидатов с одним расстоянием.", rec, d)
            untouched = not any(c["start"] < e["end"] and e["start"] < c["end"]
                                for c in changes for e in rec["original_entities"])
            if rec["group"] == "rare_name" and untouched and d["tp"] == len(rec["original_entities"]):
                add("3. Редкое корректное имя сохранено", f"**{key}**: имя не изменено и найдено.",
                    rec, d)
            if d["tp"] < len(rec["original_entities"]):
                add("4. Сущность пропущена", f"**{key}**: часть эталонных сущностей не найдена.",
                    rec, d)

    res = Resources()
    demo = load_json("queries_demo.json")[0]
    out = process(demo["original_text"], make_recognizer("dictionary", res), "A", res,
                  load_json(CONFIG_PATH))
    found["6. Опечатка образовала существующее слово"] = [
        f"Демонстрационный пример (в метрики не входит): «{demo['original_text']}» "
        f"(ошибка «{demo['typos'][0]['wrong']}» вместо «{demo['typos'][0]['correct']}»). "
        f"Слово есть в словаре, поэтому не проверяется. Предложено: «{out['corrected_text']}».\n"]
    text = "# Разбор ошибок (проверочный набор)\n\n" + "\n".join(
        f"## {c}\n\n" + "\n".join(found[c]) for c in sorted(found))
    (RESULTS_DIR / "error_analysis.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    modes = {"tune": tune, "final": final, "errors": errors}
    if len(sys.argv) != 2 or sys.argv[1] not in modes:
        raise SystemExit(__doc__)
    modes[sys.argv[1]]()
