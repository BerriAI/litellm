from __future__ import annotations

import ast
from pathlib import Path


def enumerate_python_tests(repo_root: Path, relative_path: str) -> frozenset[str]:
    source = (repo_root / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=relative_path)

    module_level: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            module_level.append(node.name)
        elif isinstance(node, ast.ClassDef):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith(
                    "test_"
                ):
                    module_level.append(f"{node.name}::{child.name}")

    return frozenset(module_level)
