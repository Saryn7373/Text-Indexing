"""Исправление опечаток и распознавание сущностей в русскоязычных запросах.

    python main.py --query "Найти статьи о работах Ломоносова в Маскве" --order B
    python main.py --file data/queries_test.json --recognizer model --threshold 2 --output results/run.csv
    python main.py --demo     # демонстрация для защиты
    python main.py            # интерактивный режим (пустая строка - выход)
"""
import argparse
import sys
from pathlib import Path

from correction import levenshtein, levenshtein_table
from evaluation import map_position
from ner import make_recognizer
from pipeline import Resources, process
from storage import load_json, save_csv, save_json


def format_result(r):
    lines = [f"Исходный текст: {r['original_text']}", f"Предлагаемый текст: {r['corrected_text']}"]
    for c in r["changes"]:
        flag = ", неоднозначно" if c["ambiguous"] else ""
        lines.append(f"Замена: [{c['start']}, {c['end']}) \"{c['original']}\" -> \"{c['replacement']}\", "
                     f"расстояние {c['distance']}{flag}; альтернативы: {', '.join(c['alternatives'])}")
    if not r["changes"]:
        lines.append("Замены: нет")
    if r["entities_original"] is not None:
        lines.append("Сущности в исходном тексте: " + ("; ".join(
            f"\"{e['text']}\", {e['type']}, [{e['start']}, {e['end']})"
            for e in r["entities_original"]) or "нет"))
    for e in r["entities"]:
        lines.append(f"Сущность: \"{e['text']}\", {e['type']}, [{e['start']}, {e['end']}), "
                     "исправленный текст")
    if not r["entities"]:
        lines.append("Сущности в исправленном тексте: не найдены")
    for rec in r["entity_recommendations"]:
        lines.append(f"Рекомендация из справочника (не применена): [{rec['start']}, {rec['end']}) "
                     f"\"{rec['original']}\" -> {', '.join(rec['candidates'])}")
    return "\n".join(lines)


def demo(res):
    query = "Найти стаьи о Лермонтове и Московксом государственном университете."
    for name in ("dictionary", "model"):
        recognizer = make_recognizer(name, res)
        for order in ("A", "B"):
            print(f"\n=== {name}, порядок {order} ===")
            print(format_result(process(query, recognizer, order, res, {"threshold": 2})))

    print("\nПроверки расстояния Левенштейна:")
    for a, b in (("", ""), ("", "дом"), ("москва", "москва"), ("москв", "москва"),
                 ("москвва", "москва"), ("маскве", "москве"), ("оснвоан", "основан")):
        print(f"  «{a}» -> «{b}»: {levenshtein(a, b)}")
    a, b = "маскве", "москве"
    print(f"\nТаблица D для «{a}» -> «{b}»:\n      " + "  ".join("∅" + b))
    for ch, row in zip("∅" + a, levenshtein_table(a, b)):
        print(f"  {ch}   " + "  ".join(map(str, row)))

    text = "Я живу в Москв и работаю в Томск."
    r = process(text, make_recognizer("dictionary", res), "A", res)
    print(f"\nГраницы после изменения длины текста:\n  {text}\n  {r['corrected_text']}")
    for c in r["changes"]:
        print(f"  [{c['start']}, {c['end']}) «{c['original']}» -> [{c['new_start']}, {c['new_end']}) "
              f"«{c['replacement']}»")
    tomsk = text.index("Томск")
    print(f"  «Томск» [{tomsk}, {tomsk + 5}) -> [{map_position(tomsk, r['changes'])}, "
          f"{map_position(tomsk + 5, r['changes'])})")
    for e in r["entities"]:
        assert r["corrected_text"][e["start"]:e["end"]] == e["text"]
        print(f"  сущность «{e['text']}» {e['type']} [{e['start']}, {e['end']}) - границы верны")


def read_queries(path):
    if path.lower().endswith(".json"):
        return [q["original_text"] if isinstance(q, dict) else str(q) for q in load_json(Path(path))]
    try:
        with open(path, encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f]
    except OSError as e:
        raise SystemExit(f"Не удалось прочитать файл запросов: {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--query", help="один запрос")
    p.add_argument("--file", help="запросы: .json (строки или записи с original_text) или .txt")
    p.add_argument("--recognizer", choices=("dictionary", "model"), default="dictionary")
    p.add_argument("--order", choices=("A", "B"), default="A")
    p.add_argument("--threshold", type=int, choices=(1, 2), default=1)
    p.add_argument("--no-freq", action="store_true", help="ранжировать без частот")
    p.add_argument("--no-protect", action="store_true", help="порядок B без защиты сущностей")
    p.add_argument("--output", help="сохранить конфигурацию и результаты (.json или .csv)")
    p.add_argument("--demo", action="store_true", help="демонстрация для защиты")
    args = p.parse_args()

    res = Resources()
    if args.demo:
        return demo(res)
    config = {"threshold": args.threshold, "use_freq": not args.no_freq,
              "protect": not args.no_protect}
    recognizer = make_recognizer(args.recognizer, res)
    results = []

    def handle(text):
        if not text.strip():
            print("Пустой запрос: нечего обрабатывать.")
            return
        results.append(process(text, recognizer, args.order, res, config))
        print(format_result(results[-1]) + "\n")

    if args.query is not None or args.file:
        for q in [args.query] if args.query is not None else read_queries(args.file):
            handle(q)
    else:
        print("Интерактивный режим (пустая строка - выход)")
        try:
            while (text := input("Запрос: ").strip()):
                handle(text)
        except EOFError:
            pass

    if args.output and results:
        if args.output.lower().endswith(".csv"):
            save_csv(args.output, [{
                "original_text": r["original_text"], "corrected_text": r["corrected_text"],
                "changes": "; ".join(f"{c['original']}->{c['replacement']}" for c in r["changes"]),
                "entities": "; ".join(f"{e['text']}|{e['type']}|{e['start']}-{e['end']}"
                                      for e in r["entities"])} for r in results],
                ["original_text", "corrected_text", "changes", "entities"])
        else:
            save_json(args.output, {"config": {"recognizer": args.recognizer, "order": args.order,
                                               **config}, "results": results})
        print(f"Результаты сохранены: {args.output}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
