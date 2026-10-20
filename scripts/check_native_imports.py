import ast
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Final

PACKAGE: Final = Path("litellm/native")
FORBIDDEN: Final = "litellm.main"


def _is_forbidden(module: str) -> bool:
    return module == FORBIDDEN or module.startswith(f"{FORBIDDEN}.")


def _from_import_targets(package: tuple[str, ...], node: ast.ImportFrom) -> Iterator[str]:
    base: Final = package[: len(package) - node.level + 1] if node.level else ()
    module: Final = ".".join((*base, node.module) if node.module else base)
    yield module
    yield from (f"{module}.{alias.name}" for alias in node.names)


def _imported(package: tuple[str, ...], node: ast.AST) -> Iterator[str]:
    match node:
        case ast.Import(names=names):
            yield from (alias.name for alias in names)
        case ast.ImportFrom():
            yield from _from_import_targets(package, node)
        case ast.Attribute(value=ast.Name(id="litellm"), attr=attr):
            yield f"litellm.{attr}"
        case _:
            return


def violations(path: Path, source: str) -> tuple[str, ...]:
    package: Final = (
        ("litellm", "native", *path.relative_to(PACKAGE).parent.parts) if path.is_relative_to(PACKAGE) else ()
    )
    tree: Final = ast.parse(source, filename=str(path))
    return tuple(
        f"{path}:{node.lineno}: litellm.native must not depend on litellm.main"
        for node in ast.walk(tree)
        if any(_is_forbidden(target) for target in _imported(package, node))
    )


def main() -> int:
    findings: Final = tuple(
        finding for path in sorted(PACKAGE.rglob("*.py")) for finding in violations(path, path.read_text())
    )
    if findings:
        sys.stderr.write("\n".join(findings) + "\n")
        return 1
    sys.stdout.write("litellm.native imports: passed\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
