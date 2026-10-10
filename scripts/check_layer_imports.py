import argparse
import ast
import subprocess
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import chain, product
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

PACKAGE_ROOT: Final = Path("litellm")
ALLOWLIST: Final = Path("scripts/layer_imports_allowlist.txt")
LEAF_MODULES: Final = frozenset(
    {
        "litellm.litellm_core_utils.completion_timeout",
        "litellm.litellm_core_utils.core_helpers",
        "litellm.litellm_core_utils.duration_parser",
        "litellm.litellm_core_utils.llm_response_utils.get_headers",
        "litellm.litellm_core_utils.provider_affinity",
    }
)
TYPES_LAYER: Final = (
    frozenset({"litellm.types", "litellm.models", "litellm.constants", "litellm._logging", "litellm._uuid"})
    | LEAF_MODULES
)
L2_FORBIDDEN: Final = ("litellm.proxy", "litellm.main", "litellm.router")

Context = Literal["module", "function"]


@dataclass(frozen=True, slots=True)
class Rule:
    name: str
    applies_to: Callable[[str], bool]
    forbids: Callable[[str], bool]


@dataclass(frozen=True, slots=True)
class Edge:
    path: str
    importer: str
    imported: str
    context: Context
    line: int


@dataclass(frozen=True, slots=True)
class Violation:
    rule: str
    edge: Edge

    @property
    def key(self) -> str:
        return f"{self.rule} {self.edge.context} {self.edge.path} {self.edge.imported}"


def is_under(module: str, package: str) -> bool:
    return module == package or module.startswith(f"{package}.")


def escapes_types_layer(module: str) -> bool:
    return is_under(module, "litellm") and not any(is_under(module, allowed) for allowed in TYPES_LAYER)


RULES: Final = (
    Rule(
        "L1",
        lambda module: is_under(module, "litellm") and not is_under(module, "litellm.proxy"),
        lambda target: is_under(target, "litellm.proxy"),
    ),
    Rule(
        "L2",
        lambda module: is_under(module, "litellm.llms"),
        lambda target: any(is_under(target, forbidden) for forbidden in L2_FORBIDDEN),
    ),
    Rule("L3", lambda module: is_under(module, "litellm.types"), escapes_types_layer),
    Rule("LEAF", lambda module: module in LEAF_MODULES, escapes_types_layer),
)


def module_name(path: Path) -> str:
    parts: Final = path.with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def import_nodes(node: ast.AST, context: Context) -> Iterator[tuple[ast.Import | ast.ImportFrom, Context]]:
    if isinstance(node, ast.Import | ast.ImportFrom):
        yield node, context
        return
    children: Final[Iterable[ast.AST]] = (
        node.orelse if isinstance(node, ast.If) and is_type_checking(node.test) else ast.iter_child_nodes(node)
    )
    child_context: Final[Context] = (
        "function" if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) else context
    )
    for child in children:
        yield from import_nodes(child, child_context)


def imported_modules(
    node: ast.Import | ast.ImportFrom, importer: str, is_package: bool, known: frozenset[str]
) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    package: Final = importer.split(".") if is_package else importer.split(".")[:-1]
    anchor: Final = tuple(package[: len(package) - node.level + 1]) if node.level else ()
    base: Final = ".".join((*anchor, *((node.module,) if node.module else ())))
    return tuple(f"{base}.{alias.name}" if f"{base}.{alias.name}" in known else base for alias in node.names)


def package_prefixes(module: str) -> Iterator[str]:
    parts: Final = module.split(".")
    return (".".join(parts[:end]) for end in range(1, len(parts) + 1))


def known_modules(paths: Iterable[Path]) -> frozenset[str]:
    return frozenset(chain.from_iterable(package_prefixes(module_name(path)) for path in paths))


def file_edges(path: Path, known: frozenset[str]) -> Iterator[Edge]:
    importer: Final = module_name(path)
    tree: Final = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node, context in import_nodes(tree, "module"):
        for imported in imported_modules(node, importer, path.name == "__init__.py", known):
            yield Edge(path.as_posix(), importer, imported, context, node.lineno)


def violations(paths: tuple[Path, ...]) -> tuple[Violation, ...]:
    known: Final = known_modules(paths)
    edges: Final = tuple(chain.from_iterable(file_edges(path, known) for path in paths))
    return tuple(
        Violation(rule.name, edge)
        for rule, edge in product(RULES, edges)
        if rule.applies_to(edge.importer) and rule.forbids(edge.imported)
    )


def parse_allowlist(text: str) -> frozenset[str]:
    lines: Final = (" ".join(line.split()) for line in text.splitlines())
    return frozenset(line for line in lines if line and not line.startswith("#"))


def report(
    found: tuple[Violation, ...], allowed: frozenset[str], base_allowed: frozenset[str] | None
) -> tuple[str, ...]:
    first_seen: Final = MappingProxyType({violation.key: violation for violation in reversed(found)})
    new: Final = tuple(
        f"{violation.edge.path}:{violation.edge.line}: {violation.rule} forbids "
        f"{violation.edge.importer} importing {violation.edge.imported} ({violation.edge.context} level)"
        for key, violation in sorted(first_seen.items())
        if key not in allowed
    )
    stale: Final = tuple(
        f"{ALLOWLIST}: stale entry '{key}' no longer matches an import, delete it"
        for key in sorted(allowed - first_seen.keys())
    )
    added: Final = (
        tuple(
            f"{ALLOWLIST}: entry '{key}' is not in the base allowlist, fix the import instead of allowlisting it"
            for key in sorted(allowed - base_allowed)
        )
        if base_allowed is not None
        else ()
    )
    return new + stale + added


def git_base_allowlist(ref: str) -> frozenset[str] | None:
    merge_base: Final = subprocess.run(
        ["git", "merge-base", ref, "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    shown: Final = subprocess.run(
        ["git", "show", f"{merge_base}:{ALLOWLIST.as_posix()}"], capture_output=True, text=True, check=False
    )
    return parse_allowlist(shown.stdout) if shown.returncode == 0 else None


class Arguments(argparse.Namespace):
    base: str | None = None


def main(
    argv: Sequence[str] | None = None,
    base_allowlist: Callable[[str], frozenset[str] | None] = git_base_allowlist,
) -> int:
    parser: Final = argparse.ArgumentParser(description="Check SDK/proxy layer import rules")
    parser.add_argument("--base", help="Also reject allowlist entries missing at the merge base with this ref")
    args: Final = parser.parse_args(argv, namespace=Arguments())
    paths: Final = tuple(sorted(PACKAGE_ROOT.rglob("*.py")))
    base_allowed: Final = base_allowlist(args.base) if args.base else None
    problems: Final = report(violations(paths), parse_allowlist(ALLOWLIST.read_text(encoding="utf-8")), base_allowed)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print("Layer imports: passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
