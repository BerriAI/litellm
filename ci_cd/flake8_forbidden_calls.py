"""Flake8 Extension that finds usage of load_dotenv."""

import ast
from typing import Generator

FORBIDDEN_CALLS = ["load_dotenv"]
PLUGIN_CODE = "FC1"


class ForbiddenCallPlugin:
    name = "flake8_forbidden_calls"
    version = "1.0.0"

    def __init__(self, tree: ast.AST):
        self._tree = tree

    def run(self) -> Generator:
        for node in ast.walk(self._tree):
            if isinstance(node, ast.Call):
                func_name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else (
                        node.func.attr if isinstance(node.func, ast.Attribute) else None
                    )
                )
                if func_name in FORBIDDEN_CALLS:
                    yield (
                        node.lineno,
                        node.col_offset,
                        f"{PLUGIN_CODE} `{func_name}` is not allowed.",
                        type(self),
                    )
