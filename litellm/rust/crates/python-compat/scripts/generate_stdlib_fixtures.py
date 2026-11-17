"""Regenerate generated/stdlib.json: `urllib.parse.quote`, `datetime`, float formatting and
`str.isspace` in CPython.

    python scripts/generate_stdlib_fixtures.py > generated/stdlib.json

`quote` rows record `urllib.parse.quote(text, safe=safe)`. `utc` rows record the fields of
`datetime.fromtimestamp(seconds, timezone.utc).replace(microsecond=micros)`. `fromisoformat`
rows record `datetime.fromisoformat(text)` as microseconds since the epoch, `aware` when it
has an offset and `naive` (read as UTC) when not, or neither when it raises `ValueError`.
`fixed` rows record `format(value, f".{precision}f")`, `number_str` rows `str(value)` for an
`int` or `float`, and `isspace` lists every code point whose `str.isspace()` is true.
"""

import json
import math
import sys
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

QUOTE_TEXTS = [
    "",
    "plain",
    "a/b/c",
    "a b",
    "a+b=c&d",
    "~_.-",
    "100%",
    "a/b:c d%#?+=&~é\"q",
    "id-é-\U0001f600",
    "日本",
    "\x00\x1f\x7f",
    "!*'();:@&=+$,/?#[]",
    "<>\\^`{|}",
]

SAFES = ["/", "", ":/"]

TIMESTAMPS = [
    (0, 0),
    (1, 999999),
    (59, 0),
    (86399, 123456),
    (86400, 0),
    (951782400, 0),
    (951868800, 0),
    (1700000000, 42),
    (1757840707, 42),
    (4102444800, 0),
    (253402300799, 999999),
]

ISO_TEXTS = [
    "2025-11-13T12:00:00+00:00",
    "2025-11-13T12:00:00Z",
    "2025-11-13T12:00:00.000001+00:00",
    "2025-11-13T12:00:00.5+00:00",
    "2025-11-13T12:00:00.123+00:00",
    "2025-11-13T12:00:00.1234567+00:00",
    "2025-11-13T12:00:00,5+00:00",
    "2025-11-13T12:00:00.+00:00",
    "2025-11-13T13:30:00+01:30",
    "2025-11-13T11:00:00-01:00",
    "2025-11-13T12:00:00-00:00",
    "2025-11-13T12:00:00+0130",
    "2025-11-13T12:00:00+01",
    "2025-11-13T12:00:00+01:30:15",
    "2025-11-13T12:00:00+01:30:15.5",
    "2025-11-13T12:00:00+23:59",
    "2025-11-13T12:00:00+24:00",
    "2025-11-13 12:00+00:00",
    "2025-11-13X12:00:00+00:00",
    "2025-11-13\u00e912:00:00+00:00",
    "20251113T120000+0000",
    "20251113T120000",
    "20251113",
    "2025-1113",
    "202511-13",
    "2025-11-13",
    "2025-11-13T12",
    "2025-11-13T12:00",
    "2025-11-13T1200",
    "2025-11-13T12:00:00",
    "2025-11-13T12:00:00.5",
    "2025-11-13T12.5",
    "2025-11-13T12:00.5",
    "2024-02-29T00:00:00+00:00",
    "0001-01-01T00:00:00+00:00",
    "9999-12-31T23:59:59.999999+00:00",
    "1969-12-31T23:59:59.999999+00:00",
    "soon",
    "",
    " 2025-11-13",
    "2025-02-30T12:00:00+00:00",
    "2025-13-01T12:00:00+00:00",
    "2025-00-01T12:00:00+00:00",
    "0000-01-01T00:00:00+00:00",
    "2025-11-13T24:00:00+00:00",
    "2025-11-13T12:60:00+00:00",
    "2025-11-13T12:00:60+00:00",
    "2025-11-13T12:00:00+00:00 ",
    "2025-11-13T12:00:00 +00:00",
    "2025-11-13T12:00:00ZZ",
    "2025-11-13T12:00:00+00:00Z",
    "2025-11",
    "2025",
    "+002025-11-13",
    "2025-317",
    "2025-W46",
    "2025-W46-4",
    "2025W464",
    "2025-W53-1",
    "2025-W46-4T12:00:00+00:00",
    "2025-11-13T12:00:00+1:00",
    "2025-11-13T1:00:00+00:00",
    "2025-11-13T12:00:00 ",
    "2025-11-13T12:00:00  +00:00",
    "2025-11-13T12:00:00\t+00:00",
    "2025-11-13T12:00 +00:00",
    "2025-11-13T12 +00:00",
    "2025-11-13T12:00:00.5 +00:00",
    "2025-11-13T12:00:00x+00:00",
    "2025-11-13T12:00:00 Z",
    "2025-11-13T12:00:00+0",
    "2025-11-13T12:00:00+000",
    "2025-11-13T12:00:00+01:30:15.",
    "2025-11-13T12:00:00+013015",
    "2025-11-13T12:00:00+013015.5",
    "2025-11-13T12:00:00+23:59:59.999999",
    "2025-11-13T12:00:00-23:59:59.999999",
    "2025-11-13T12:00:00+00:99",
    "2025-11-13T12:00:00+99",
    "9999-12-31T24:00:00+00:00",
    "2025-12-31T24:00:00",
    "2024-02-28T24:00",
    "2025-02-30T24:00:00",
    "2025-11-13T24:00:00.000001",
    "2025-11-13T12:00:00.1234567890123",
    "2025-11-13T12:00:00.12a",
    "2025-11-13T12:00:00-",
    "2025-11-13T12:00:00Z+00:00",
    "2025-11-13T12:3",
    "2025-11-13T12:00:0",
    "2025-11-13T12:00:00,",
    "2025-11-13T12:0000",
    "2025-11-13T1200:00",
    "20251113T120000123+0000",
    "20251113T120000.5",
    "2025-11-13T12:00:00\x00",
    "2025-W464",
    "2025W46-4",
    "2025W46",
    "2025-W4",
    "2025-W46-",
    "2025-W46-41",
    "2020-W53-1",
    "2026-W53-1",
    "2025-W00-1",
    "2025-W46-8",
    "2025-W46-0",
    "2025-W46T12",
    "2025-W46-4T12",
    "2025W464T12",
    "2025W4612",
    "9999-W52-6",
    "0001-W01-1",
    "2025-11-13T",
    "2025-11-13TT",
    "2025-11-1",
    "2025-11-13 ",
    "2025-11-13\u00e9",
    "2025-11-13\U0001f600" "12:00",
    "1234567",
    "2025-11-13T12:00:00+05:30",
]

FIXED = [
    (0.125, 2),
    (0.375, 2),
    (2.5, 0),
    (3.5, 0),
    (1.0, 4),
    (0.000049999, 4),
    (-0.005, 2),
    (-0.0, 2),
    (0.0, 0),
    (99.999, 2),
    (1e22, 2),
    (1e-7, 4),
    (123.456789, 4),
    (0.1, 20),
    (5e-324, 2),
    (1.7976931348623157e308, 2),
    (math.nan, 2),
    (math.inf, 2),
    (-math.inf, 4),
]

NUMBERS = [0, 1, -5, 2**53 + 1, 2**63 - 1, -(2**63), 0.0, -0.0, 1.0, 0.1, 1.5, 1e16, 1e-5, 123456789.123, math.inf, -math.inf]

EPOCH = datetime(1970, 1, 1)


def fromisoformat(text):
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        return {"text": text}
    if moment.tzinfo is None:
        return {"text": text, "naive": (moment - EPOCH) // timedelta(microseconds=1)}
    return {"text": text, "aware": (moment - EPOCH.replace(tzinfo=timezone.utc)) // timedelta(microseconds=1)}


def number_str(value):
    return {"value": value, "str": str(value)} if math.isfinite(value) else {"float": repr(value), "str": str(value)}


def fixed(value, precision):
    shown = value if math.isfinite(value) else repr(value)
    return {"value": shown, "precision": precision, "text": format(value, f".{precision}f")}


def utc(seconds, micros):
    moment = datetime.fromtimestamp(seconds, timezone.utc).replace(microsecond=micros)
    return {
        "seconds": seconds,
        "micros": micros,
        "fields": [moment.year, moment.month, moment.day, moment.hour, moment.minute, moment.second, moment.microsecond],
    }


if __name__ == "__main__":
    json.dump(
        {
            "python": sys.version.split()[0],
            "quote": [{"text": text, "safe": safe, "quoted": quote(text, safe=safe)} for text in QUOTE_TEXTS for safe in SAFES],
            "utc": [utc(seconds, micros) for seconds, micros in TIMESTAMPS],
            "fromisoformat": [fromisoformat(text) for text in ISO_TEXTS],
            "fixed": [fixed(value, precision) for value, precision in FIXED],
            "number_str": [number_str(value) for value in NUMBERS],
            "isspace": [code for code in range(sys.maxunicode + 1) if chr(code).isspace()],
        },
        sys.stdout,
        indent=2,
        ensure_ascii=True,
    )
