"""Fails when a module under litellm/telemetry/ imports any other part of litellm."""

import ast
import sys
from collections.abc import Iterator
from itertools import chain
from pathlib import Path
from typing import Final

PACKAGE: Final = "litellm.telemetry"
PACKAGE_DIR: Final = Path(__file__).resolve().parents[2] / "litellm" / "telemetry"


def _imported_modules(tree: ast.AST) -> Iterator[tuple[int, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from ((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module is not None:
            yield (node.lineno, node.module)


def _is_forbidden(module: str) -> bool:
    is_litellm: Final = module == "litellm" or module.startswith("litellm.")
    is_own_package: Final = module == PACKAGE or module.startswith(f"{PACKAGE}.")
    return is_litellm and not is_own_package


def _violations_in(path: Path) -> Iterator[str]:
    tree: Final = ast.parse(path.read_text(), filename=str(path))
    return (f"{path}:{lineno} imports {module}" for lineno, module in _imported_modules(tree) if _is_forbidden(module))


def find_violations(package_dir: Path) -> tuple[str, ...]:
    return tuple(chain.from_iterable(_violations_in(path) for path in sorted(package_dir.rglob("*.py"))))


def main() -> int:
    violations: Final = find_violations(PACKAGE_DIR)
    if not violations:
        sys.stdout.write(f"{PACKAGE} imports nothing else from litellm\n")
        return 0
    sys.stderr.write(f"{PACKAGE} must not depend on the rest of litellm:\n" + "\n".join(violations) + "\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
