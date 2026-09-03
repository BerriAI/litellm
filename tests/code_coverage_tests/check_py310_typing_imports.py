import ast
import os
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

PY311_PLUS_TYPING_NAMES: Final[frozenset[str]] = frozenset(
    {
        "NotRequired",
        "Required",
        "Self",
        "LiteralString",
        "Never",
        "assert_never",
        "assert_type",
        "reveal_type",
        "TypeVarTuple",
        "Unpack",
        "dataclass_transform",
        "override",
        "TypeAliasType",
        "get_original_bases",
        "ReadOnly",
        "TypeIs",
        "NoDefault",
        "get_protocol_members",
        "is_protocol",
        "evaluate_forward_ref",
        "TypeForm",
    }
)


@dataclass(frozen=True, slots=True)
class TypingImportViolation:
    file: str
    line: int
    name: str


def _contains_sys_version_info(node: ast.AST) -> bool:
    return any(
        isinstance(child, ast.Attribute)
        and isinstance(child.value, ast.Name)
        and child.value.id == "sys"
        and child.attr == "version_info"
        for child in ast.walk(node)
    )


def _walk_with_ancestors(
    node: ast.AST, ancestors: tuple[ast.AST, ...] = ()
) -> Iterator[tuple[ast.AST, tuple[ast.AST, ...]]]:
    yield node, ancestors
    next_ancestors: Final[tuple[ast.AST, ...]] = (*ancestors, node)
    for child in ast.iter_child_nodes(node):
        yield from _walk_with_ancestors(child, next_ancestors)


def _is_version_guarded(ancestors: tuple[ast.AST, ...]) -> bool:
    enclosing_if: ast.If | None = next(
        (
            ancestor
            for ancestor in reversed(ancestors)
            if isinstance(ancestor, ast.If)
        ),
        None,
    )
    return enclosing_if is not None and _contains_sys_version_info(enclosing_if.test)


def scan_file(file_path: str | os.PathLike[str]) -> tuple[TypingImportViolation, ...]:
    path: Final[Path] = Path(file_path)
    tree: Final[ast.Module] = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return tuple(
        violation
        for node, ancestors in _walk_with_ancestors(tree)
        if not _is_version_guarded(ancestors)
        for violation in _violations_for_node(node, path)
    )


def _violations_for_node(
    node: ast.AST, path: Path
) -> tuple[TypingImportViolation, ...]:
    if isinstance(node, ast.ImportFrom) and node.module == "typing":
        return tuple(
            TypingImportViolation(file=str(path), line=node.lineno, name=alias.name)
            for alias in node.names
            if alias.name in PY311_PLUS_TYPING_NAMES
        )
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "typing"
        and node.attr in PY311_PLUS_TYPING_NAMES
    ):
        return (TypingImportViolation(file=str(path), line=node.lineno, name=node.attr),)
    return ()


def scan_directory(base_dir: str | os.PathLike[str] = ".") -> tuple[TypingImportViolation, ...]:
    base_path: Final[Path] = Path(base_dir)
    return tuple(
        violation
        for directory in (base_path / "litellm", base_path / "enterprise")
        if directory.exists()
        for path in directory.rglob("*.py")
        for violation in scan_file(path)
    )


def main() -> None:
    violations: Final[tuple[TypingImportViolation, ...]] = scan_directory()
    if violations:
        message: Final[str] = "\n".join(
            (
                "Python 3.10-incompatible typing imports found:",
                *(
                    f"{violation.file}:{violation.line}: {violation.name} is unavailable in Python 3.10; "
                    "import it from typing_extensions instead because litellm supports Python 3.10"
                    for violation in violations
                ),
            )
        )
        sys.stdout.write(f"{message}\n")
        raise RuntimeError("Import Python 3.10-incompatible typing names from typing_extensions instead")
    sys.stdout.write("No Python 3.10-incompatible typing imports found.\n")


if __name__ == "__main__":
    main()
