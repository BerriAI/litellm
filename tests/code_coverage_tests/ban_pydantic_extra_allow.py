"""Fail CI when a new Pydantic model opts into ``extra="allow"``.

``extra="allow"`` silently accepts undeclared keys, so typos survive validation and
real fields (pricing on ``ModelInfo``, for example) never get declared anywhere. The
models listed in ``extra-allow-budget.json`` predate this check and stay allowed;
anything new must declare its fields.

That list is a budget like the others, so ``scripts/budget_ratchet_check.py`` is what
reds when it grows, and its ``limit`` must equal the number of models it lists.

What it reads is the opt-in a module spells out, so it says nothing about a model that
inherits ``extra="allow"`` from a base declared elsewhere. Subclassing a listed model
still widens it, and only review catches that.
"""

import ast
import json
import os
import sys
from types import MappingProxyType
from typing import Final, Iterator, NamedTuple, Sequence

SCAN_ROOT: Final = "litellm"
BUDGET_PATH: Final = "extra-allow-budget.json"
BUDGET_RULE: Final = "extra_allow_models"


class Budget(NamedTuple):
    limit: int
    models: frozenset[str]


def read_budget(path: str) -> Budget:
    with open(path, encoding="utf-8") as handle:
        rule: Final = MappingProxyType(json.load(handle)[BUDGET_RULE])
    return Budget(limit=int(rule["limit"]), models=frozenset(rule["models"]))


class Violation(NamedTuple):
    file: str
    line: int
    model: str

    def identifier(self) -> str:
        return f"{self.file}::{self.model}"


def _assigned_names(statement: ast.stmt) -> tuple[tuple[str, ast.expr], ...]:
    if isinstance(statement, ast.Assign):
        targets, value = statement.targets, statement.value
    elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
        targets, value = [statement.target], statement.value
    else:
        return ()
    return tuple((target.id, value) for target in targets if isinstance(target, ast.Name))


class Binding(NamedTuple):
    line: int
    name: str
    value: ast.expr
    branch: tuple[str, ...]


def _shadows(later: Binding, earlier: Binding) -> bool:
    """Whether ``later`` is guaranteed to have replaced ``earlier`` by the time something below
    reads the name. It is, only when it runs on the same branch or an enclosing one; a binding in
    a sibling branch of the same ``if`` or ``try`` is an alternative, not a replacement, so both
    values stay in play."""
    return later.line > earlier.line and earlier.branch[: len(later.branch)] == later.branch


class Scope(NamedTuple):
    """The bindings in view, read as of ``line``.

    Resolution walks a name to the values it could hold just above the line reading it, and each
    hop down a chain of aliases carries that alias's own line, so rebinding a name later can
    neither hide an earlier opt-in nor implicate a model that never had one. A name bound in more
    than one branch resolves to every branch's value, since which one runs is a runtime question.
    """

    bindings: tuple[Binding, ...]
    line: int

    def resolve(self, node: ast.expr, seen: frozenset[str] = frozenset()) -> tuple[ast.expr, ...]:
        if not isinstance(node, ast.Name) or node.id in seen:
            return (node,)
        visible: Final = tuple(
            binding for binding in self.bindings if binding.name == node.id and binding.line < self.line
        )
        live: Final = tuple(binding for binding in visible if not any(_shadows(other, binding) for other in visible))
        if not live:
            return (node,)
        return tuple(
            resolved
            for binding in live
            for resolved in Scope(self.bindings, binding.line).resolve(binding.value, seen | {node.id})
        )


def _scoped_statements(node: ast.AST, branch: tuple[str, ...] = ()) -> Iterator[tuple[ast.stmt, tuple[str, ...]]]:
    """Statements that belong to ``node``'s own scope, each with the branch it sits on, descending
    through ``if``, ``try``, ``with``, loop and ``match`` blocks, since a name bound or a class
    declared inside one of those is still bound in the enclosing scope. Functions and nested
    classes open a new scope, so a nested class is yielded but not entered, and function bodies
    are left alone."""
    for field, value in ast.iter_fields(node):
        for child in value if isinstance(value, list) else [value]:
            if not isinstance(child, ast.AST) or isinstance(child, ast.expr):
                continue
            nested = branch if isinstance(node, (ast.Module, ast.ClassDef)) else (*branch, f"{id(node)}.{field}")
            if isinstance(child, ast.ClassDef):
                yield child, nested
                continue
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(child, ast.stmt):
                yield child, nested
            yield from _scoped_statements(child, nested)


def _statements(node: ast.AST) -> tuple[ast.stmt, ...]:
    return tuple(statement for statement, _ in _scoped_statements(node))


def _bindings(node: ast.AST) -> tuple[Binding, ...]:
    return tuple(
        Binding(statement.lineno, name, value, branch)
        for statement, branch in _scoped_statements(node)
        for name, value in _assigned_names(statement)
    )


def _names_allow(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return node.value == "allow"
    return isinstance(node, ast.Attribute) and node.attr == "allow"


def _is_allow_literal(node: ast.expr, scope: Scope) -> bool:
    return any(_names_allow(resolved) for resolved in scope.resolve(node))


def _is_extra_allow_keyword(keyword: ast.keyword, scope: Scope) -> bool:
    """A keyword with no ``arg`` is a ``**`` spread, so what it spreads is read as a config itself."""
    if keyword.arg is None:
        return _is_extra_allow_value(keyword.value, scope)
    return keyword.arg == "extra" and _is_allow_literal(keyword.value, scope)


def _mapping_sets_extra_allow(node: ast.Dict, scope: Scope) -> bool:
    """A ``None`` key is a ``**`` spread, so what it spreads is read as a config itself."""
    return any(
        _is_extra_allow_value(value, scope)
        if key is None
        else isinstance(key, ast.Constant) and key.value == "extra" and _is_allow_literal(value, scope)
        for key, value in zip(node.keys, node.values)
    )


def _config_sets_extra_allow(node: ast.expr, scope: Scope) -> bool:
    if isinstance(node, ast.Call):
        return any(_is_extra_allow_keyword(keyword, scope) for keyword in node.keywords)
    if isinstance(node, ast.Dict):
        return _mapping_sets_extra_allow(node, scope)
    return False


def _is_extra_allow_value(node: ast.expr, scope: Scope) -> bool:
    return any(_config_sets_extra_allow(resolved, scope) for resolved in scope.resolve(node))


def _assigns_extra_allow(statement: ast.stmt, target_names: Sequence[str], scope: Scope) -> bool:
    return any(
        name in set(target_names) and _is_extra_allow_value(value, scope) for name, value in _assigned_names(statement)
    )


def _body_scope(node: ast.AST, enclosing: tuple[Binding, ...]) -> tuple[Binding, ...]:
    """The bindings a statement in ``node``'s body reads. A class-local binding joins the enclosing
    ones rather than replacing them, so which one wins is decided per read line the way the body
    itself decides it: a class-local constant only shadows an enclosing name below the line it is
    bound on, and a read above that line still sees the enclosing value."""
    return enclosing + _bindings(node)


def _legacy_config_sets_extra_allow(config: ast.ClassDef, enclosing: tuple[Binding, ...]) -> bool:
    bindings: Final = _body_scope(config, enclosing)
    return any(
        isinstance(inner, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "extra" for target in inner.targets)
        and _is_allow_literal(inner.value, Scope(bindings, inner.lineno))
        for inner in _statements(config)
    )


def _class_allows_extra(class_def: ast.ClassDef, enclosing: tuple[Binding, ...]) -> bool:
    if any(_is_extra_allow_keyword(keyword, Scope(enclosing, class_def.lineno)) for keyword in class_def.keywords):
        return True
    bindings: Final = _body_scope(class_def, enclosing)
    statements: Final = _statements(class_def)
    if any(
        _assigns_extra_allow(statement, ["model_config"], Scope(bindings, statement.lineno)) for statement in statements
    ):
        return True
    return any(
        isinstance(statement, ast.ClassDef)
        and statement.name == "Config"
        and _legacy_config_sets_extra_allow(statement, bindings)
        for statement in statements
    )


def _iter_classes(node: ast.AST, prefix: str = "") -> Iterator[tuple[str, ast.ClassDef]]:
    for statement in _statements(node):
        if isinstance(statement, ast.ClassDef):
            qualified = f"{prefix}{statement.name}"
            yield qualified, statement
            yield from _iter_classes(statement, f"{qualified}.")


def find_violations_in_source(source: str, relative_path: str) -> tuple[Violation, ...]:
    """Every opt-in this check understands spells ``allow`` in the module that declares the
    model, whether as a literal, an ``Extra.allow`` attribute, or a constant it resolves, so a
    module without that substring cannot hold one and is not worth parsing."""
    if "allow" not in source:
        return ()
    tree: Final = ast.parse(source, filename=relative_path)
    bindings: Final = _bindings(tree)
    return tuple(
        Violation(file=relative_path, line=class_def.lineno, model=qualified)
        for qualified, class_def in _iter_classes(tree)
        if _class_allows_extra(class_def, bindings)
    )


def _scan_file(file_path: str, base_dir: str) -> tuple[Violation, ...]:
    relative = os.path.relpath(file_path, base_dir).replace(os.sep, "/")
    with open(file_path, "r", encoding="utf-8") as handle:
        return find_violations_in_source(handle.read(), relative)


def find_extra_allow_models(base_dir: str) -> tuple[Violation, ...]:
    return tuple(
        violation
        for root, _, files in os.walk(os.path.join(base_dir, SCAN_ROOT))
        for file_name in sorted(files)
        if file_name.endswith(".py")
        for violation in _scan_file(os.path.join(root, file_name), base_dir)
    )


def main() -> int:
    base_dir: Final = os.getcwd()
    budget: Final = read_budget(os.path.join(base_dir, BUDGET_PATH))
    found: Final = find_extra_allow_models(base_dir)
    violations: Final = tuple(violation for violation in found if violation.identifier() not in budget.models)
    stale: Final = tuple(sorted(budget.models - {violation.identifier() for violation in found}))
    miscounted: Final = budget.limit != len(budget.models)

    for violation in violations:
        print(f'{violation.file}:{violation.line}: {violation.model} sets extra="allow"')
    if violations:
        print(
            f'\nFound {len(violations)} new Pydantic model(s) using extra="allow".\n'
            'Declare the fields you accept instead. extra="allow" hides typos and\n'
            "leaves real fields undocumented and untyped. If a model genuinely has to\n"
            f"forward opaque provider payloads, add it to {BUDGET_RULE} in {BUDGET_PATH},\n"
            "raise the limit to match, and say why in the PR: raising it reds the\n"
            "non-gating budget-ratchet check so the loosening is seen and accepted."
        )
    if stale:
        print(
            f'\nThese {BUDGET_PATH} models no longer use extra="allow" (or moved).\n'
            "Remove them and lower the limit so it keeps ratcheting down:"
        )
        for entry in stale:
            print(f"  {entry}")
    if miscounted:
        print(
            f"\n{BUDGET_PATH} lists {len(budget.models)} models under a limit of {budget.limit}.\n"
            "They must match, since the limit is what the budget ratchet reads."
        )

    if violations or stale or miscounted:
        return 1
    print('No new extra="allow" Pydantic models found.')
    return 0


if __name__ == "__main__":
    sys.exit(main())
