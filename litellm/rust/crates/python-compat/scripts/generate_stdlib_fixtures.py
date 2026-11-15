"""Regenerate generated/stdlib.json: `urllib.parse.quote` and UTC `datetime` fields in CPython.

    python scripts/generate_stdlib_fixtures.py > generated/stdlib.json

`quote` rows record `urllib.parse.quote(text, safe=safe)`. `utc` rows record the fields of
`datetime.fromtimestamp(seconds, timezone.utc).replace(microsecond=micros)`.
"""

import json
import sys
from datetime import datetime, timezone
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
        },
        sys.stdout,
        indent=2,
        ensure_ascii=True,
    )
