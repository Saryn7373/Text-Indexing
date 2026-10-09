# Лабораторная работа № 4. Исправление опечаток и распознавание сущностей

Python 3.11.9, `pip install -r requirements.txt`. Модель NER: Natasha `slovnet_ner_news_v1`
(natasha 1.6.0), CPU, веса внутри пакета.

| Файл | Назначение |
|---|---|
| `prepare_data.py` | разметка запросов, словарь словоформ, справочник сущностей, контроль данных |
| `correction.py` | нормализация, Левенштейн, кандидаты, исправление с журналом замен |
| `ner.py` | распознавание: справочник с правилами и модель Natasha |
| `pipeline.py` | порядки обработки A/B и защита сущностей |
| `evaluation.py` | метрики, настройка (dev), итоговый прогон (test), разбор ошибок |
| `main.py` | консольная программа и демонстрация для защиты |
| `storage.py` | чтение/запись JSON и CSV |
| `data/` | корпус, ручная разметка (`queries_markup.json`, `entity_seed.json`) и собранные файлы |

```
python prepare_data.py
python evaluation.py tune      # выбор параметров на dev
python evaluation.py final     # итог на test (один раз)
python evaluation.py errors    # results/error_analysis.md
python main.py --demo
python main.py --query "Я живу в Маскве." --recognizer model --order B --threshold 2 --output results/run.json
```

Параметры `main.py`: `--recognizer {dictionary,model}`, `--order {A,B}`, `--threshold {1,2}`,
`--no-freq`, `--no-protect`, `--file`, `--output`. Правила нормализации описаны в `correction.py`.
