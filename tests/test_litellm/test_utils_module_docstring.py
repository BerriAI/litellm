import ast
from pathlib import Path


def test_utils_module_has_docstring():
    utils_path = Path(__file__).parents[2] / "litellm" / "utils.py"
    module = ast.parse(utils_path.read_text())

    assert ast.get_docstring(module) == (
        "Utility helpers for LiteLLM core request handling and provider support."
    )
