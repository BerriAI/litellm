"""Regenerate generated/flags.json: what LiteLLM's own text-to-bool helpers return.

    uv run --no-project --with pydantic==2.13.4 python scripts/generate_flag_fixtures.py > generated/flags.json

It compiles each helper from the checked-out `litellm` sources, so it needs neither the package's
other dependencies nor its native extension. Each row records one input written as JSON and what each helper returns for it, with the text-only
helpers left null for an input that is not a string.
"""

import ast
import json
import sys
from pathlib import Path
from typing import Final

from pydantic import TypeAdapter, ValidationError

PACKAGE = Path(__file__).resolve().parents[4]


def helpers(module, *names: str):
    tree = ast.parse((PACKAGE / module).read_text())
    functions = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id in names)
    ]
    namespace = {"Final": Final, "TypeAdapter": TypeAdapter, "ValidationError": ValidationError}
    exec(compile(ast.Module(body=functions, type_ignores=[]), module, "exec"), namespace)
    return [namespace[name] for name in names]


_str_to_bool, _coerces_to_true = helpers("_redis.py", "_str_to_bool", "_coerces_to_true")
(str_to_bool,) = helpers("secret_managers/main.py", "str_to_bool")
_, lax_bool = helpers("litellm_core_utils/core_helpers.py", "_LAX_BOOL", "lax_bool")

TEXTS = [
    "",
    " ",
    "true",
    "True",
    "TRUE",
    "tRuE",
    " true ",
    "\ttrue\n",
    "\x1ctrue\x1f",
    " false ",
    "true​",
    "false",
    "False",
    "FALSE",
    " false",
    "1",
    "0",
    " 1",
    "yes",
    "Yes",
    "YES",
    " yes",
    "no",
    "on",
    "On",
    "off",
    "t",
    "f",
    "y",
    "n",
    "2",
    "-1",
    "1.0",
    "truthy",
    "maybe",
    "None",
    "null",
    "İ",
    "truK",
]

VALUES = [
    None,
    True,
    False,
    0,
    1,
    2,
    -1,
    0.0,
    0.5,
    1.0,
    [],
    [False],
    {},
    {"a": None},
    *TEXTS,
]


def row(value):
    is_text = isinstance(value, str)
    return {
        "input": json.dumps(value),
        "str_to_bool": str_to_bool(value) if is_text else None,
        "redis_str_to_bool": _str_to_bool(value) if is_text else None,
        "lower_is_true": value.lower() == "true" if is_text else None,
        "coerces_to_true": _coerces_to_true(value),
        "str_to_bool_or_truthy": bool(str_to_bool(value) if is_text else value),
        "lax_bool": lax_bool(value),
    }


if __name__ == "__main__":
    json.dump({"rows": [row(value) for value in VALUES]}, sys.stdout, indent=2, ensure_ascii=True)
