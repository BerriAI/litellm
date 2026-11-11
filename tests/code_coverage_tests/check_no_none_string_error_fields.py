from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from itertools import chain
from pathlib import Path
from typing import Final

ERROR_FIELDS: Final = frozenset({"type", "param"})
SOURCE_ROOT: Final = Path(__file__).resolve().parents[2] / "litellm"


def _is_none_string(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value == "None"


def _names_an_error_field(node: ast.expr | None) -> bool:
    return isinstance(node, ast.Constant) and node.value in ERROR_FIELDS


def _keyword_defaults_to_none_string(call: ast.Call) -> bool:
    return any(keyword.arg in ERROR_FIELDS and _is_none_string(keyword.value) for keyword in call.keywords)


def _getattr_defaults_to_none_string(call: ast.Call) -> bool:
    return (
        isinstance(call.func, ast.Name)
        and call.func.id == "getattr"
        and len(call.args) == 3
        and _names_an_error_field(call.args[1])
        and _is_none_string(call.args[2])
    )


def _dict_holds_none_string(node: ast.Dict) -> bool:
    return any(_names_an_error_field(key) and _is_none_string(value) for key, value in zip(node.keys, node.values))


def _is_violation(node: ast.AST) -> bool:
    match node:
        case ast.Call():
            return _keyword_defaults_to_none_string(node) or _getattr_defaults_to_none_string(node)
        case ast.Dict():
            return _dict_holds_none_string(node)
        case _:
            return False


def violations(source: str, filename: str) -> Iterator[str]:
    return (f"{filename}:{node.lineno}" for node in ast.walk(ast.parse(source, filename=filename)) if _is_violation(node))


def _file_violations(path: Path) -> Iterator[str]:
    return violations(path.read_text(encoding="utf-8"), str(path.relative_to(SOURCE_ROOT.parent)))


def main() -> int:
    found: Final = tuple(chain.from_iterable(_file_violations(path) for path in sorted(SOURCE_ROOT.rglob("*.py"))))
    if not found:
        return 0
    print(
        'An error\'s "type" or "param" is set to the string "None", which OpenAI clients read as a real '
        "error class or field name. Use a named error type and a real None for param instead:"
    )
    print("\n".join(found))
    return 1


if __name__ == "__main__":
    sys.exit(main())
