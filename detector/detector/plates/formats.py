"""Philippine plate formats, used to fix characters an OCR engine commonly confuses.

L = a letter, D = a digit. The list follows the Wikipedia article "Vehicle registration plates of
the Philippines" and may be incomplete: the LTO has not said its current formats are final. A plate
that matches none of these is still accepted by the voting step, just with lower confidence.
"""
import re

PATTERNS = [
    "LLLDDDD",   # cars, trucks, buses, 2014 series and later:  ABC 1234
    "LLLDDD",    # cars, 1981 series:                           ABC 123
    "LLDDDDD",   # motorcycles, 2014 series:                    AB 12345
    "LLDDDD",    # motorcycles, 1981 series and NCR:            AB 1234
    "DDDDLL",    # motorcycles, 1981 series and NCR:            1234 AB
    "DDDLLL",    # motorcycles, 2018 onward:                    123 ABC
    "DDDDDDDDDD",  # temporary plate seen on a new vehicle, printed 1201-401686 (only one example is known)
    "LDDDLL", "LLDDDL", "DLLLDD", "LDDDDL", "LDLDDD", "LDDLDD",  # newer six-character motorcycle plates
]

# What an OCR engine mixes up. A digit read where a letter belongs, and the reverse.
LETTER_FOR_DIGIT = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}
DIGIT_FOR_LETTER = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "G": "6", "B": "8"}

_NOT_ALNUM = re.compile(r"[^A-Z0-9]")


def clean(text):
    return _NOT_ALNUM.sub("", (text or "").upper())


def matches(text, pattern):
    if len(text) != len(pattern):
        return False
    for ch, kind in zip(text, pattern):
        if kind == "L" and not ch.isalpha():
            return False
        if kind == "D" and not ch.isdigit():
            return False
    return True


def is_valid_format(text):
    return any(matches(text, p) for p in PATTERNS)


# The edge of a plate, its frame and its screws are often read as an extra upright stroke at either
# end of the number: "I", "1", "|", "J", or sometimes "7".
EDGE_NOISE = set("I1|J7")


def _trim_edges(text, max_removed=2):
    """Drop stray strokes at the ends when that turns the text into exactly one valid plate.

    Tries removing 1 character, then 2, from the ends, only characters in EDGE_NOISE. Returns the
    plate, or None if nothing fits or if two different plates fit (then guessing would be worse).
    """
    for removed in range(1, max_removed + 1):
        found = set()
        for lead in range(removed + 1):
            trail = removed - lead
            if lead + trail >= len(text):
                continue
            if not all(c in EDGE_NOISE for c in text[:lead]):
                continue
            if trail and not all(c in EDGE_NOISE for c in text[len(text) - trail:]):
                continue
            candidate = text[lead:len(text) - trail]
            if is_valid_format(candidate):
                found.add(candidate)
        if len(found) == 1:
            return found.pop()
        if len(found) > 1:
            return None
    return None


def correct(text, max_changes=2):
    """Clean up an OCR reading using the Philippine plate formats.

    Fixes letter/digit confusions when exactly one format explains the text best, and drops a stray
    stroke at either end (see _trim_edges). Returns (text, changed). Text that already fits a format
    is left alone. If two readings are equally likely, nothing is changed: guessing would be worse
    than leaving it.
    """
    text = clean(text)
    if is_valid_format(text):
        return text, False
    fixed, changed = _swap_fix(text, max_changes)
    if changed and is_valid_format(fixed):
        return fixed, True
    trimmed = _trim_edges(text)
    if trimmed:
        return trimmed, True
    return fixed, changed


def _swap_fix(text, max_changes):
    best, best_cost, tied = None, None, False
    for pattern in PATTERNS:
        if len(pattern) != len(text):
            continue
        fixed, cost, ok = [], 0, True
        for ch, kind in zip(text, pattern):
            if kind == "L" and ch.isdigit():
                swap = LETTER_FOR_DIGIT.get(ch)
            elif kind == "D" and ch.isalpha():
                swap = DIGIT_FOR_LETTER.get(ch)
            else:
                swap = ch
            if swap is None:
                ok = False
                break
            cost += swap != ch
            fixed.append(swap)
        if not ok or cost > max_changes:
            continue
        if best_cost is None or cost < best_cost:
            best, best_cost, tied = "".join(fixed), cost, False
        elif cost == best_cost and "".join(fixed) != best:
            tied = True
    if best is None or tied:
        return text, False
    return best, True
