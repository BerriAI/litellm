"""Layering contract: the translation package stays decoupled from the v1 stack.

The whole point of the rewrite is seams, so the package must not reach into the
monolith it replaces. This walks every module under ``litellm/translation`` and
fails if any imports a forbidden v1 surface, the deterministic check that stands
in for the import-linter contract until one is wired into CI.
"""

import ast
import pathlib

import litellm.translation as translation_pkg

_FORBIDDEN_PREFIXES = (
    "litellm.llms",
    "litellm.main",
    "litellm.utils",
    "litellm.proxy",
    "litellm.litellm_core_utils",
)

_PACKAGE_ROOT = pathlib.Path(translation_pkg.__file__).parent


def _imported_modules(tree: ast.AST) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.append(node.module)
    return modules


def test_no_module_imports_a_forbidden_v1_surface() -> None:
    offenders = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for module in _imported_modules(tree):
            if module.startswith(_FORBIDDEN_PREFIXES):
                offenders.append(f"{path.relative_to(_PACKAGE_ROOT)} -> {module}")
    assert offenders == [], "forbidden imports: " + "; ".join(offenders)
