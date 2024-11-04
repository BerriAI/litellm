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
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in FORBIDDEN_CALLS:
                    yield (node.lineno, node.col_offset, f"{PLUGIN_CODE} `{node.func.id}` is not allowed.", type(self))