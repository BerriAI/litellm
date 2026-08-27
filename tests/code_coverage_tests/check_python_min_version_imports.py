"""
`pyproject.toml` declares `requires-python = ">=3.10"`, but several stdlib names
only exist from 3.11 on. Importing one unguarded from `typing`, `enum` or
`asyncio` raises ImportError at module import time, which on a proxy start path
takes the whole proxy down on an interpreter we claim to support.

Import these from `typing_extensions` instead, as the rest of the codebase does,
or gate the import on `sys.version_info`.
"""

import ast
import os
from typing import Any

SCAN_DIR = "litellm"

# Added in 3.11; absent from the 3.10 stdlib
PY311_ONLY = {
    "typing": {
        "NotRequired",
        "Required",
        "Self",
        "LiteralString",
        "Never",
        "assert_never",
        "assert_type",
        "TypeVarTuple",
        "Unpack",
        "dataclass_transform",
        "reveal_type",
        "get_overloads",
        "clear_overloads",
    },
    "enum": {"StrEnum", "ReprEnum", "EnumCheck", "verify", "member", "nonmember", "global_enum"},
    "asyncio": {"TaskGroup", "Runner", "Barrier"},
}

SUGGESTED_SOURCE = {
    "typing": "typing_extensions",
    "enum": "backports.strenum or a plain str/Enum subclass",
    "asyncio": "a sys.version_info guard",
}


def _is_version_guarded(tree: ast.Module, lineno: int) -> bool:
    """True when the import sits inside an `if sys.version_info ...` branch."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        source = ast.dump(node.test)
        if "version_info" not in source:
            continue
        for branch in (node.body, node.orelse):
            for stmt in branch:
                start = getattr(stmt, "lineno", None)
                end = getattr(stmt, "end_lineno", start)
                if start is not None and end is not None and start <= lineno <= end:
                    return True
    return False


def _violations_in_file(file_path: str) -> list[dict[str, Any]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=file_path)
    except Exception:
        return []

    results: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module not in PY311_ONLY:
            continue
        flagged = sorted({a.name for a in node.names} & PY311_ONLY[node.module])
        if not flagged or _is_version_guarded(tree, node.lineno):
            continue
        results.append(
            {
                "file": os.path.normpath(file_path),
                "line": node.lineno,
                "module": node.module,
                "names": flagged,
            }
        )
    return results


def scan_directory(base_dir: str) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for root, _dirs, files in os.walk(os.path.join(base_dir, SCAN_DIR)):
        for name in files:
            if name.endswith(".py"):
                violations.extend(_violations_in_file(os.path.join(root, name)))
    return sorted(violations, key=lambda v: (v["file"], v["line"]))


def main() -> None:
    base_dir = "."  # tests run from repo root in CI
    violations = scan_directory(base_dir)
    if violations:
        print("\n🚨 Found stdlib imports that do not exist on Python 3.10:")
        for v in violations:
            names = ", ".join(v["names"])
            suggestion = SUGGESTED_SOURCE[v["module"]]
            print(f"* {v['file']}:{v['line']} -> from {v['module']} import {names} (use {suggestion})")
        print("\n")
        raise Exception(
            "pyproject.toml declares requires-python >=3.10, so these imports break "
            "`import litellm` on a supported interpreter."
        )
    print("No Python 3.11+ stdlib imports found outside a version guard.")


if __name__ == "__main__":
    main()
