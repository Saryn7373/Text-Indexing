"""Нормализация, расстояние Левенштейна и исправление опечаток по словарю словоформ.

Правила нормализации (одинаковы при поиске кандидатов и оценке):
- сравнение в нижнем регистре; «ё» везде читается и выводится как «е»;
- слово с дефисом - один токен («Санкт-Петербург»), тире «—» - разделитель;
- не исправляются сокращения из заглавных букв (МГУ), токены короче 3 символов и с цифрами;
- стоп-слова из запроса не удаляются, они есть в словаре корпуса;
- регистр замены повторяет регистр исходного слова (в слове с дефисом - по частям).
"""
import re

TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+(?:-[A-Za-zА-Яа-яЁё0-9]+)*")
MIN_LEN = 3
MAX_CANDIDATES = 3


def replace_yo(text):
    return text.replace("ё", "е").replace("Ё", "Е")


def normalize_word(word):
    return replace_yo(word.lower())


def tokenize(text):
    """Список токенов (start, end, текст)."""
    return [(m.start(), m.end(), m.group()) for m in TOKEN_RE.finditer(text)]


def is_checkable(token):
    return len(token) >= MIN_LEN and not token.isupper() \
        and not any(ch.isdigit() for ch in token)


def apply_case(original, replacement):
    o_parts, r_parts = original.split("-"), replacement.split("-")
    if len(o_parts) > 1 and len(o_parts) == len(r_parts):
        return "-".join(apply_case(o, r) for o, r in zip(o_parts, r_parts))
    if original.isupper() and len(original) > 1:
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def levenshtein_table(a, b):
    """D(i,0)=i; D(0,j)=j; D(i,j)=min(D(i-1,j)+1, D(i,j-1)+1, D(i-1,j-1)+c)."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
    return d


def levenshtein(a, b):
    return levenshtein_table(a, b)[-1][-1]


def find_candidates(word, freq, threshold=1, use_freq=True):
    """Не более трёх кандидатов: по расстоянию, затем по частоте (если use_freq), затем по алфавиту."""
    key = normalize_word(word)
    found = []
    for other, count in freq.items():
        if abs(len(other) - len(key)) <= threshold:
            distance = levenshtein(key, other)
            if 0 < distance <= threshold:
                found.append((distance, -count if use_freq else 0, other))
    found.sort()
    return [{"word": other, "distance": d}
            for d, _, other in found[:MAX_CANDIDATES]]


def correct_text(text, freq, threshold=1, use_freq=True, protected=()):
    """Заменяет неизвестные слова первым кандидатом. Возвращает (новый текст, журнал замен).

    Запись журнала: исходный диапазон [start, end), исходное слово, замена, расстояние,
    альтернативы, признак неоднозначности (несколько кандидатов с лучшим расстоянием)
    и диапазон в новом тексте.
    """
    pieces, changes, cursor, new_len = [], [], 0, 0
    for start, end, token in tokenize(text):
        if not is_checkable(token) or normalize_word(token) in freq \
                or any(start < e and s < end for s, e in protected):
            continue
        candidates = find_candidates(token, freq, threshold, use_freq)
        if not candidates:
            continue
        replacement = apply_case(token, candidates[0]["word"])
        new_len += start - cursor
        pieces += [text[cursor:start], replacement]
        changes.append({
            "start": start, "end": end, "original": token, "replacement": replacement,
            "distance": candidates[0]["distance"],
            "alternatives": [apply_case(token, c["word"]) for c in candidates],
            "ambiguous": sum(c["distance"] == candidates[0]["distance"] for c in candidates) > 1,
            "new_start": new_len, "new_end": new_len + len(replacement)})
        new_len += len(replacement)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces), changes
