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


def _walk_with_ancestors(
    node: ast.AST, ancestors: tuple[tuple[ast.AST, str], ...] = ()
) -> Iterator[tuple[ast.AST, tuple[tuple[ast.AST, str], ...]]]:
    yield node, ancestors
    for field_name, field_value in ast.iter_fields(node):
        if isinstance(field_value, ast.AST):
            yield from _walk_with_ancestors(field_value, (*ancestors, (node, field_name)))
        elif isinstance(field_value, list):
            for child in field_value:
                if isinstance(child, ast.AST):
                    yield from _walk_with_ancestors(child, (*ancestors, (node, field_name)))


def _is_sys_version_info(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "sys"
        and node.attr == "version_info"
    )


def _is_version_guarded(ancestors: tuple[tuple[ast.AST, str], ...]) -> bool:
    nearest_if: Final[tuple[ast.If, str] | None] = next(
        (
            (ancestor, field_name)
            for ancestor, field_name in reversed(ancestors)
            if isinstance(ancestor, ast.If)
        ),
        None,
    )
    if nearest_if is None:
        return False
    enclosing_if, branch = nearest_if
    test: Final[ast.expr] = enclosing_if.test
    if not isinstance(test, ast.Compare) or len(test.ops) != 1 or not _is_sys_version_info(test.left):
        return False
    operator: Final[ast.cmpop] = test.ops[0]
    return (isinstance(operator, (ast.Gt, ast.GtE)) and branch == "body") or (
        isinstance(operator, (ast.Lt, ast.LtE)) and branch == "orelse"
    )


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
        for directory in (
            base_path / "litellm",
            base_path / "enterprise",
            base_path / "litellm-proxy-extras" / "litellm_proxy_extras",
        )
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
