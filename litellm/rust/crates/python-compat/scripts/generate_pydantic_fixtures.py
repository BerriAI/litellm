"""Regenerate generated/pydantic.json: what pydantic's lax `bool` and `int` validation return.

    uv run --no-project --with pydantic==2.13.4 python scripts/generate_pydantic_fixtures.py > generated/pydantic.json

Each row records `TypeAdapter(bool).validate_python(value)` and `TypeAdapter(int).validate_python(value)`
for one input written as JSON, or null where pydantic raises `ValidationError`. Inputs are what a
JSON-shaped setting can carry: None, booleans, integers, floats, strings, lists and dicts.
"""

import json
import sys
from typing import TypeVar

import pydantic
from pydantic import TypeAdapter, ValidationError

T = TypeVar("T")

TEXTS = [
    "",
    " ",
    "0",
    "1",
    "2",
    "-1",
    "+1",
    "-0",
    "00",
    "007",
    "1_000",
    "1__000",
    "_1",
    "1_",
    "1.",
    ".5",
    "1.0",
    "1.00",
    "1.5",
    "-1.0",
    "1e3",
    "1E3",
    "0x10",
    "0b1",
    "0o7",
    " 5 ",
    "\t5\n",
    "5 0",
    "١٢",
    "inf",
    "nan",
    "99999999999999999999",
    "-99999999999999999999",
    "true",
    "True",
    "TRUE",
    " true ",
    "false",
    "False",
    "t",
    "f",
    "T",
    "y",
    "n",
    "yes",
    "Yes",
    "no",
    "on",
    "On",
    "off",
    "OFF",
    "maybe",
    "truthy",
    "nul",
    "+",
    "-",
    "+1.0",
    "-0.0",
    "1_0.0",
    "1.0_0",
    "1.0.0",
    "1 ",
    "\u00a05",
    "\u20035",
    "-007",
    "-00",
    "-0_1",
    "+007",
    "-1_000",
    "+0",
    "00.0",
    "0_0",
    "-01.00",
    "+-1",
    "--1",
]

# pydantic's int parser refuses more than 4300 characters, counting a `-` but not the `+`, whitespace,
# leading zeros, underscores or `.0` that pydantic strips first.
LONG_TEXTS = [
    "9" * 4300,
    "9" * 4301,
    "-" + "9" * 4299,
    "-" + "9" * 4300,
    "+" + "9" * 4300,
    " " + "9" * 4300 + " ",
    "0" * 5000 + "9" * 4300,
    "0" * 5000 + "9" * 4301,
    "-" + "0" * 5000 + "9" * 4299,
    "1" + "_1" * 4299,
    "1" + "_1" * 4300,
    "9" * 4300 + ".0",
    "9" * 4301 + ".0",
]

VALUES = [
    None,
    True,
    False,
    0,
    1,
    2,
    -1,
    2**63 - 1,
    2**63,
    2**64,
    -(2**63),
    0.0,
    -0.0,
    1.0,
    0.5,
    1.5,
    2.0,
    -3.0,
    1e15,
    1e20,
    1e300,
    2.0**62,
    2.0**63,
    -(2.0**63),
    9007199254740993.0,
    float("inf"),
    float("-inf"),
    float("nan"),
    [],
    [1],
    {},
    {"a": 1},
    *TEXTS,
    *LONG_TEXTS,
]


def validated(adapter: TypeAdapter[T], value: object) -> T | None:
    try:
        return adapter.validate_python(value)
    except ValidationError:
        return None


def row(value: object) -> dict[str, str | bool | None]:
    as_int = validated(TypeAdapter(int), value)
    return {
        "input": json.dumps(value),
        "bool": validated(TypeAdapter(bool), value),
        "int": None if as_int is None else str(as_int),
    }


if __name__ == "__main__":
    json.dump(
        {"pydantic": pydantic.VERSION, "rows": [row(value) for value in VALUES]},
        sys.stdout,
        indent=2,
        ensure_ascii=True,
    )
