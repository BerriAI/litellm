#!/usr/bin/env python3
"""Test-quality checker: the test-suite smells no linter enforces.

Sibling of scripts/check_type_discipline.py, same output contract
(``path:line: CODE message``) and same stdlib-only constraint, aimed at the test
tree instead of the package. Each rule is a shape the testing-strategy audit
measured and named; scripts/test_quality_gate.py caps the codebase total of each
one against test-quality-budget.json so the counts can only ratchet down.

Rules
-----
TQ001   A collectible test function whose body contains no assertion of any kind:
        no `assert` statement, no `pytest.raises`/`warns`/`deprecated_call`/`fail`,
        and no `assert*` method call (mock's `assert_called_once`, unittest's
        `assertEqual`, `numpy.testing.assert_allclose`). Such a test passes as long
        as the code under it does not raise, so it pins nothing and cannot fail for
        the reason anyone would want it to. Assert the observable output instead.
        The whole function subtree counts, nested helper definitions included, so a
        test that asserts inside a locally-defined async helper passes.
TQ002   Mock-echo: a test that patches something and whose every assertion only
        inspects the mock that replaced it (`assert_called_once_with`, `.called`,
        `.call_args`, `.call_count`, `.mock_calls`). The test restates the
        implementation back at itself: it verifies that the code called what the
        code calls, so it survives any refactor that keeps the call and breaks the
        behavior. Assert what the caller observes -- the returned value, the
        rebuilt response, the raised exception -- and fake at the HTTP boundary
        (respx / MockTransport) rather than patching litellm internals.
        A test with no assertions at all is TQ001, never TQ002.
TQ003   `sys.path.insert(...)` inside the test tree. pytest's rootdir handling and
        the installed package already make `litellm` importable, so these are
        no-ops carried by copy-paste; the ones that are not no-ops make the test's
        imports depend on the working directory it happens to run from.
TQ004   Raw `os.environ[...] = ...` assignment. The write outlives the test and
        leaks into whatever runs next in the same process, which is how a suite
        acquires an ordering dependency. Use `monkeypatch.setenv`, which is undone
        at teardown.
TQ005   `litellm.<attr> = ...` module-global mutation. The SDK's module globals are
        process-wide, so this is the same leak as TQ004 one level up, and it is
        what the 491-line save/restore conftest exists to paper over. Inject the
        dependency or use a fixture that restores it.
TQ006   A `pytest.skip` reached only when a credential-shaped environment variable is
        absent. On a runner that does not hold that credential the guard fires every
        time, so the test reports green having executed nothing and is indistinguishable
        from coverage that exists. Fake the provider at the HTTP boundary, or fail
        loudly, so a missing credential shows up as a missing credential. Absence is
        what the condition has to say -- `not key`, `key is None`, `"KEY" not in
        os.environ` -- since a skip taken when the credential is present is somebody's
        deliberate branch. The gate follows one local or module-level binding, which is
        the `key = os.getenv(...)` then `if not key: pytest.skip(...)` shape most of
        these use.
TQ008   A `patch(...)` whose target is a `litellm.` internal. Patching the SDK's own
        functions pins the test to the current wiring instead of the behaviour, and it
        is the idiom the suite reaches for instead of faking the HTTP boundary. Mocking
        a third-party client, a transport, or anything outside `litellm.` is untouched.
TQ007   A module global that a conftest saves before every test and restores after it.
        The save/restore list is a hand-maintained inventory of the leaks the suite
        already knows about, so it is allowed to shrink and never to grow: a new entry
        means one more global whose lifetime the tests manage instead of the code owning
        it. Give the consumers an injection seam rather than another snapshot line. The
        names are read from the keys the conftest assigns directly and from whatever the
        save loop iterates, including a module-level tuple or dict it names rather than
        spells out.

Every rule is suppressible with `# test-quality-ok: <reason>` on the reported
line, following the repo's `*-ok: <reason>` convention. A suppression without a
reason does not suppress.

What counts as an assertion
---------------------------
An `assert` statement; `pytest.raises` / `warns` / `deprecated_call` / `fail`,
qualified or bare (`skip` and `xfail` are deliberately excluded, since they abort
the test rather than pin a behaviour); and any callable whose name starts with
`assert`, qualified (`m.assert_called_once`, `self.assertEqual`,
`np.testing.assert_allclose`) or bare (`assert_auth_denied(...)`, the shape the
e2e harness uses). A test also counts as asserting when it reaches an assertion
through a function defined in the same module, followed transitively, because
extracting the assertions into a shared helper is good factoring rather than a
test that pins nothing. A helper imported from another module is not followed, so
a test whose only assertions live across a module boundary still reports TQ001
and needs a suppression.

What counts as mock inspection (TQ002)
--------------------------------------
An `assert_`-prefixed call, which is mock's own family, or a reference to
`called` / `call_args` / `call_args_list` / `call_count` / `mock_calls` and their
await-counterparts. unittest's `assertEqual` has no underscore after "assert" and
so is never mistaken for one. A patch is installed by any call or decorator whose
name is `patch` or `patch.object` / `patch.dict` / `patch.multiple`, which covers
`unittest.mock` however it was imported as well as pytest-mock's `mocker.patch`.

Scope
-----
Only files under the test roots passed on the command line are examined, and
TQ001/TQ002 only look at functions pytest would collect: a `test_`-prefixed
function at module level, or a `test_`-prefixed method of a `Test`-prefixed
class that defines no `__init__`.

Usage
-----
    python check_test_quality.py tests/

Exit code 1 if any violation is found. Stdlib only.
"""

from __future__ import annotations

import ast
import io
import os
import re
import sys
import tokenize
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from types import MappingProxyType
from typing import Final, NamedTuple

TEST_FUNCTION_PREFIX: Final = "test_"
TEST_CLASS_PREFIX: Final = "Test"
MIN_REASON_LEN: Final = 3

SUPPRESSION_TOKEN: Final = "test-quality-ok"
SUPPRESSION_RE: Final = re.compile(r"#\s*test-quality-ok(?::\s*(?P<reason>.*))?")

PYTEST_ASSERTION_HELPERS: Final = frozenset(("raises", "warns", "deprecated_call", "fail"))

MOCK_INSPECTION_ATTRIBUTES: Final = frozenset((
    "called", "call_args", "call_args_list", "call_count", "mock_calls",
    "await_args", "await_args_list", "await_count", "awaited",
))
MOCK_ASSERTION_PREFIX: Final = "assert_"

PATCH_MEMBERS: Final = frozenset(("object", "dict", "multiple"))

ENVIRON_READERS: Final = frozenset(("os.environ.get", "environ.get", "os.getenv", "getenv"))
ENVIRON_MAPPINGS: Final = frozenset(("os.environ", "environ"))
SKIP_CALLS: Final = frozenset(("pytest.skip", "skip"))
CONFTEST_NAME: Final = "conftest.py"
SDK_MODULE: Final = "litellm"

CREDENTIAL_NAME_RE: Final = re.compile(
    r"(?:API_KEY|_KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|DATABASE_URL|ACCESS_KEY_ID)$"
)

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


class Violation(NamedTuple):
    path: Path
    line: int
    code: str
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.code} {self.message}"


def _dotted_name(node: ast.expr) -> str:
    """`a.b.c` for an attribute chain rooted in a plain name, else ""."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        root: Final = _dotted_name(node.value)
        return f"{root}.{node.attr}" if root else ""
    return ""


def suppressed_lines(source: str) -> frozenset[int]:
    """Lines carrying `# test-quality-ok: <reason>` with a reason of usable length."""
    try:
        tokens: Final = tuple(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return frozenset()
    return frozenset(
        token.start[0]
        for token in tokens
        if token.type == tokenize.COMMENT
        and (match := SUPPRESSION_RE.search(token.string)) is not None
        and len((match.group("reason") or "").strip()) >= MIN_REASON_LEN
    )


def _is_collectible_class(node: ast.ClassDef) -> bool:
    """pytest collects `Test`-prefixed classes that define no constructor."""
    if not node.name.startswith(TEST_CLASS_PREFIX):
        return False
    return not any(
        isinstance(child, ast.FunctionDef) and child.name == "__init__"
        for child in node.body
    )


def iter_test_functions(tree: ast.Module) -> Iterator[FunctionNode]:
    """Every function pytest would collect from this module, in source order."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith(TEST_FUNCTION_PREFIX):
                yield node
        elif isinstance(node, ast.ClassDef) and _is_collectible_class(node):
            yield from (
                child
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name.startswith(TEST_FUNCTION_PREFIX)
            )


def _is_pytest_assertion_call(call: ast.Call) -> bool:
    func: Final = call.func
    if isinstance(func, ast.Attribute):
        return func.attr in PYTEST_ASSERTION_HELPERS
    if isinstance(func, ast.Name):
        return func.id in PYTEST_ASSERTION_HELPERS
    return False


def _is_assertion_helper_call(call: ast.Call) -> bool:
    """Any `assert*` callable: `x.assertEqual(...)`, `m.assert_called_once()`,
    `np.testing.assert_allclose(...)`, and the bare shared helpers the e2e harness
    uses (`assert_auth_denied(result, ...)`)."""
    func: Final = call.func
    if isinstance(func, ast.Attribute):
        return func.attr.startswith("assert")
    return isinstance(func, ast.Name) and func.id.startswith("assert")


def iter_assertions(function: FunctionNode) -> Iterator[ast.stmt | ast.Call]:
    """Every node in the function that pins a behaviour, nested definitions included."""
    for node in ast.walk(function):
        if isinstance(node, ast.Assert):
            yield node
        elif isinstance(node, ast.Call) and (
            _is_pytest_assertion_call(node) or _is_assertion_helper_call(node)
        ):
            yield node


class CallTarget(NamedTuple):
    """A call that might resolve to a function defined in this module: either a bare
    name, looked up among the module-level functions, or a `self.` attribute, looked
    up among the enclosing class's own methods."""

    through_self: bool
    name: str


@dataclass(frozen=True, slots=True)
class Scope:
    """What one function can reach by name. Keeping methods per-class is what stops
    two same-named helpers in different classes from resolving to each other."""

    module_level: Mapping[str, FunctionNode]
    methods: Mapping[str, FunctionNode]

    def resolve(self, target: CallTarget) -> FunctionNode | None:
        source: Final = self.methods if target.through_self else self.module_level
        return source.get(target.name)


def _call_target(func: ast.expr) -> CallTarget | None:
    if isinstance(func, ast.Name):
        return CallTarget(through_self=False, name=func.id)
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
        return CallTarget(through_self=True, name=func.attr)
    return None


def _call_targets(function: FunctionNode) -> frozenset[CallTarget]:
    return frozenset(
        target
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        for target in (_call_target(node.func),)
        if target is not None
    )


def _functions_in(body: Iterable[ast.stmt]) -> Mapping[str, FunctionNode]:
    return MappingProxyType({
        node.name: node
        for node in body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    })


def build_scopes(tree: ast.Module) -> Mapping[FunctionNode, Scope]:
    """Every function in the module paired with what it can reach by name. A
    module-level function sees only module-level functions; a method also sees its
    own class's methods, and no other class's."""
    module_level: Final = _functions_in(tree.body)
    module_scope: Final = Scope(module_level=module_level, methods=MappingProxyType({}))
    class_scopes: Final = tuple(
        (node, Scope(module_level=module_level, methods=_functions_in(node.body)))
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    )
    return MappingProxyType({
        **{function: module_scope for function in module_level.values()},
        **{
            function: scope
            for node, scope in class_scopes
            for function in scope.methods.values()
        },
    })


def _reaches_assertion(
    function: FunctionNode,
    scopes: Mapping[FunctionNode, Scope],
    seen: frozenset[FunctionNode],
) -> bool:
    if function in seen:
        return False
    if any(iter_assertions(function)):
        return True
    scope: Final = scopes.get(function)
    if scope is None:
        return False
    return any(
        _reaches_assertion(callee, scopes, seen | frozenset((function,)))
        for target in _call_targets(function)
        for callee in (scope.resolve(target),)
        if callee is not None
    )


def asserts_through_helpers(
    function: FunctionNode, scopes: Mapping[FunctionNode, Scope]
) -> bool:
    """Whether the test reaches an assertion through a function defined in this
    module, followed transitively. Extracting the assertions into a shared helper is
    good factoring rather than a test that pins nothing, so following one is what
    keeps TQ001 honest."""
    scope: Final = scopes.get(function)
    if scope is None:
        return False
    return any(
        _reaches_assertion(callee, scopes, frozenset((function,)))
        for target in _call_targets(function)
        for callee in (scope.resolve(target),)
        if callee is not None
    )


def _is_patch_installer(dotted: str) -> bool:
    """`patch`, `mock.patch`, `mocker.patch`, `patch.object`, `mock.patch.dict`, ..."""
    parts: Final = dotted.split(".")
    if parts[-1] == "patch":
        return True
    return len(parts) >= 2 and parts[-2] == "patch" and parts[-1] in PATCH_MEMBERS


def _installs_patch(function: FunctionNode) -> bool:
    decorators: Final = tuple(
        _dotted_name(d.func) if isinstance(d, ast.Call) else _dotted_name(d)
        for d in function.decorator_list
    )
    if any(name and _is_patch_installer(name) for name in decorators):
        return True
    return any(
        _is_patch_installer(_dotted_name(node.func))
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and _dotted_name(node.func)
    )


def _only_inspects_a_mock(node: ast.stmt | ast.Call) -> bool:
    """True when this assertion reads a mock's call record and nothing else."""
    if isinstance(node, ast.Call):
        func = node.func
        return isinstance(func, ast.Attribute) and func.attr.startswith(MOCK_ASSERTION_PREFIX)
    return any(
        isinstance(child, ast.Attribute)
        and (
            child.attr in MOCK_INSPECTION_ATTRIBUTES
            or child.attr.startswith(MOCK_ASSERTION_PREFIX)
        )
        for child in ast.walk(node)
    )


def iter_assertion_violations(path: Path, tree: ast.Module) -> Iterator[Violation]:
    scopes: Final = build_scopes(tree)
    for function in iter_test_functions(tree):
        assertions: Final = tuple(iter_assertions(function))
        if not assertions and asserts_through_helpers(function, scopes):
            continue
        if not assertions:
            yield Violation(
                path,
                function.lineno,
                "TQ001",
                f"test `{function.name}` asserts nothing, so it can only fail by raising; "
                f"assert the observable output (suppress: `# {SUPPRESSION_TOKEN}: <reason>`)",
            )
        elif _installs_patch(function) and all(map(_only_inspects_a_mock, assertions)):
            yield Violation(
                path,
                function.lineno,
                "TQ002",
                f"test `{function.name}` patches something and only asserts that the mock was "
                f"called, which restates the implementation; assert what the caller observes "
                f"(suppress: `# {SUPPRESSION_TOKEN}: <reason>`)",
            )


def iter_sys_path_violations(path: Path, tree: ast.Module) -> Iterator[Violation]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _dotted_name(node.func) == "sys.path.insert":
            yield Violation(
                path,
                node.lineno,
                "TQ003",
                "sys.path.insert in a test; pytest's rootdir and the installed package already "
                f"make litellm importable (suppress: `# {SUPPRESSION_TOKEN}: <reason>`)",
            )


def _environ_subscript_targets(target: ast.expr) -> Iterator[ast.Subscript]:
    if isinstance(target, ast.Tuple):
        for element in target.elts:
            yield from _environ_subscript_targets(element)
        return
    if isinstance(target, ast.Subscript) and _dotted_name(target.value) in ("os.environ", "environ"):
        yield target


def iter_environ_violations(path: Path, tree: ast.Module) -> Iterator[Violation]:
    for node in ast.walk(tree):
        targets: Final = (
            node.targets if isinstance(node, ast.Assign)
            else (node.target,) if isinstance(node, (ast.AugAssign, ast.AnnAssign))
            else ()
        )
        for target in targets:
            for subscript in _environ_subscript_targets(target):
                yield Violation(
                    path,
                    subscript.lineno,
                    "TQ004",
                    "raw os.environ write leaks into every test that runs after this one; "
                    f"use monkeypatch.setenv (suppress: `# {SUPPRESSION_TOKEN}: <reason>`)",
                )


def _litellm_attribute_targets(target: ast.expr) -> Iterator[ast.Attribute]:
    if isinstance(target, ast.Tuple):
        for element in target.elts:
            yield from _litellm_attribute_targets(element)
        return
    if isinstance(target, ast.Attribute) and _dotted_name(target.value) == "litellm":
        yield target


def iter_global_mutation_violations(path: Path, tree: ast.Module) -> Iterator[Violation]:
    for node in ast.walk(tree):
        targets: Final = (
            node.targets if isinstance(node, ast.Assign)
            else (node.target,) if isinstance(node, (ast.AugAssign, ast.AnnAssign))
            else ()
        )
        for target in targets:
            for attribute in _litellm_attribute_targets(target):
                yield Violation(
                    path,
                    attribute.lineno,
                    "TQ005",
                    f"litellm.{attribute.attr} is a process-wide global; writing it here is what the "
                    "save/restore conftest exists to undo, so inject the dependency or use a fixture "
                    f"(suppress: `# {SUPPRESSION_TOKEN}: <reason>`)",
                )


def _is_sdk_internal(dotted: str) -> bool:
    return dotted == SDK_MODULE or dotted.startswith(f"{SDK_MODULE}.")


def _sdk_import_bindings(tree: ast.Module) -> Iterator[tuple[str, str]]:
    """(local name, dotted path) for every import that binds something under `litellm`."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (
                (alias.asname, alias.name) if alias.asname else (root, root)
                for alias in node.names
                if _is_sdk_internal(alias.name)
                for root in (alias.name.partition(".")[0],)
            )
        elif isinstance(node, ast.ImportFrom) and node.module and _is_sdk_internal(node.module):
            yield from ((alias.asname or alias.name, f"{node.module}.{alias.name}") for alias in node.names)


def _sdk_aliases(tree: ast.Module) -> Mapping[str, str]:
    """Local names bound to something under `litellm`, mapped to the path they stand for.

    `from litellm.llms.openai.chat import handler` then `patch.object(handler.X, ...)`
    reaches the same internal as the dotted string form and has to read the same way.
    """
    return MappingProxyType({name: dotted for name, dotted in _sdk_import_bindings(tree)})


def _resolved(dotted: str, aliases: Mapping[str, str]) -> str:
    root, _, rest = dotted.partition(".")
    base: Final = aliases.get(root, root)
    return f"{base}.{rest}" if rest else base


def _patch_targets(call: ast.Call, aliases: Mapping[str, str]) -> Iterator[str]:
    """What a patch installer is replacing: the dotted string it names, or the
    attribute chain handed to `patch.object` / `patch.dict`, resolved through the
    module's imports so a locally bound SDK object reads as its full path."""
    for first in call.args[:1]:
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            yield first.value
        elif dotted := _dotted_name(first):
            yield _resolved(dotted, aliases)


def iter_internal_patch_violations(path: Path, tree: ast.Module) -> Iterator[Violation]:
    aliases: Final = _sdk_aliases(tree)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _is_patch_installer(_dotted_name(node.func))):
            continue
        for target in _patch_targets(node, aliases):
            if _is_sdk_internal(target):
                yield Violation(
                    path,
                    node.lineno,
                    "TQ008",
                    f"patches `{target}`, an SDK internal, so the test is pinned to how the code is "
                    "wired rather than what it does; fake the HTTP boundary (respx / MockTransport) "
                    f"or inject the collaborator (suppress: `# {SUPPRESSION_TOKEN}: <reason>`)",
                )


def _environ_keys(node: ast.AST) -> Iterator[str]:
    for inner in ast.walk(node):
        if isinstance(inner, ast.Call) and _dotted_name(inner.func) in ENVIRON_READERS:
            yield from (
                argument.value
                for argument in inner.args[:1]
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
            )
        elif isinstance(inner, ast.Subscript) and _dotted_name(inner.value) in ENVIRON_MAPPINGS:
            if isinstance(inner.slice, ast.Constant) and isinstance(inner.slice.value, str):
                yield inner.slice.value
        elif isinstance(inner, ast.Compare) and any(isinstance(op, (ast.In, ast.NotIn)) for op in inner.ops):
            if any(_dotted_name(right) in ENVIRON_MAPPINGS for right in inner.comparators):
                if isinstance(inner.left, ast.Constant) and isinstance(inner.left.value, str):
                    yield inner.left.value


def _credential_bindings(tree: ast.Module) -> Mapping[str, str]:
    return MappingProxyType({
        target.id: key
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for key in tuple(k for k in _environ_keys(node.value) if CREDENTIAL_NAME_RE.search(k))[:1]
        for target in node.targets
        if isinstance(target, ast.Name)
    })


def _absence_operands(test: ast.expr) -> Iterator[ast.expr]:
    """The subtrees of an `if` condition that are true when what they name is missing."""
    for node in ast.walk(test):
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            yield node.operand
        elif isinstance(node, ast.Compare) and _is_absent_from_environ(node):
            yield node
        elif isinstance(node, ast.Compare) and _is_compared_to_none(node):
            yield node.left


def _is_absent_from_environ(node: ast.Compare) -> bool:
    return any(isinstance(op, ast.NotIn) for op in node.ops) and any(
        _dotted_name(right) in ENVIRON_MAPPINGS for right in node.comparators
    )


def _is_compared_to_none(node: ast.Compare) -> bool:
    return all(isinstance(op, (ast.Is, ast.Eq)) for op in node.ops) and any(
        isinstance(right, ast.Constant) and right.value is None for right in node.comparators
    )


def _gating_credential(test: ast.expr, bindings: Mapping[str, str]) -> str | None:
    return next(
        (
            credential
            for operand in _absence_operands(test)
            for credential in _named_credentials(operand, bindings)
        ),
        None,
    )


def _named_credentials(node: ast.expr, bindings: Mapping[str, str]) -> Iterator[str]:
    yield from (key for key in _environ_keys(node) if CREDENTIAL_NAME_RE.search(key))
    yield from (
        bindings[inner.id] for inner in ast.walk(node) if isinstance(inner, ast.Name) and inner.id in bindings
    )


def iter_credential_skip_violations(path: Path, tree: ast.Module) -> Iterator[Violation]:
    bindings: Final = _credential_bindings(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        credential: Final = _gating_credential(node.test, bindings)
        if credential is None:
            continue
        for statement in node.body:
            for inner in ast.walk(statement):
                if isinstance(inner, ast.Call) and _dotted_name(inner.func) in SKIP_CALLS:
                    yield Violation(
                        path,
                        inner.lineno,
                        "TQ006",
                        f"this test skips itself when {credential} is absent, so a run without "
                        "that credential reports green having executed nothing; fake the provider at "
                        "the HTTP boundary, or fail loudly so the missing credential is visible "
                        f"(suppress: `# {SUPPRESSION_TOKEN}: <reason>`)",
                    )


def _reads_sdk_attribute(node: ast.AST) -> bool:
    return any(
        (
            isinstance(inner, ast.Call)
            and _dotted_name(inner.func) == "getattr"
            and bool(inner.args)
            and _dotted_name(inner.args[0]) == SDK_MODULE
        )
        or (isinstance(inner, ast.Attribute) and _dotted_name(inner.value) == SDK_MODULE)
        for inner in ast.walk(node)
    )


def _subscript_targets(node: ast.AST) -> Iterator[ast.Subscript]:
    for inner in ast.walk(node):
        if isinstance(inner, ast.Assign):
            yield from (target for target in inner.targets if isinstance(target, ast.Subscript))


def _saves_sdk_attribute_by_key(node: ast.AST) -> Iterator[ast.Subscript]:
    """Every `<dict>["name"] = <something read off litellm>`, whatever the dict is called.

    Matching on the shape rather than on a list of blessed dict names is what reaches
    the conftest that builds its snapshot inside a helper and calls the dict `state`.
    """
    for inner in ast.walk(node):
        if isinstance(inner, ast.Assign) and _reads_sdk_attribute(inner.value):
            yield from (target for target in inner.targets if isinstance(target, ast.Subscript))


def _saves_sdk_attributes_in_loop(node: ast.For) -> bool:
    """A save loop reads the SDK and stores under the loop variable, in either order.

    The read is often bound to a local first (`val = getattr(litellm, attr)`) and only
    then stored, so the read and the store are separate statements and cannot be
    required of the same assignment.
    """
    if not isinstance(node.target, ast.Name):
        return False
    stores_by_key: Final = any(
        isinstance(subscript.slice, ast.Name) and subscript.slice.id == node.target.id
        for statement in node.body
        for subscript in _subscript_targets(statement)
    )
    return stores_by_key and any(_reads_sdk_attribute(statement) for statement in node.body)


def _module_constants(tree: ast.Module) -> Mapping[str, ast.expr]:
    return MappingProxyType({
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    })


def _string_members(node: ast.expr) -> Iterator[tuple[str, int]]:
    """The string names a collection literal holds: a tuple/list's items, a dict's keys."""
    elements: Final = (
        node.elts if isinstance(node, (ast.Tuple, ast.List)) else node.keys if isinstance(node, ast.Dict) else ()
    )
    yield from (
        (element.value, element.lineno)
        for element in elements
        if isinstance(element, ast.Constant) and isinstance(element.value, str)
    )


def _snapshotted_names(tree: ast.Module) -> Iterator[tuple[str, int]]:
    constants: Final = _module_constants(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            yield from (
                (subscript.slice.value, subscript.lineno)
                for subscript in _saves_sdk_attribute_by_key(node)
                if isinstance(subscript.slice, ast.Constant) and isinstance(subscript.slice.value, str)
            )
        elif isinstance(node, ast.For) and _saves_sdk_attributes_in_loop(node):
            iterable: Final = constants.get(node.iter.id) if isinstance(node.iter, ast.Name) else node.iter
            if iterable is not None:
                yield from _string_members(iterable)


def iter_conftest_inventory_violations(path: Path, tree: ast.Module) -> Iterator[Violation]:
    if path.name != CONFTEST_NAME:
        return
    seen: Final = dict(reversed(tuple(_snapshotted_names(tree))))
    for name, line in sorted(seen.items(), key=lambda item: item[1]):
        yield Violation(
            path,
            line,
            "TQ007",
            f"`litellm.{name}` is saved and restored around every test in this tree; the list is an "
            "inventory of known leaks and may only shrink, so give the consumers an injection seam "
            f"instead of adding to it (suppress: `# {SUPPRESSION_TOKEN}: <reason>`)",
        )


def check_file(path: Path) -> tuple[Violation, ...]:
    try:
        source: Final = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return (Violation(path, 0, "TQ000", f"unreadable: {exc}"),)

    try:
        tree: Final = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return (Violation(path, exc.lineno or 0, "TQ000", f"syntax error: {exc.msg}"),)

    skip: Final = suppressed_lines(source)
    return tuple(
        violation
        for violation in (
            *iter_assertion_violations(path, tree),
            *iter_sys_path_violations(path, tree),
            *iter_environ_violations(path, tree),
            *iter_global_mutation_violations(path, tree),
            *iter_credential_skip_violations(path, tree),
            *iter_conftest_inventory_violations(path, tree),
            *iter_internal_patch_violations(path, tree),
        )
        if violation.line not in skip
    )


def collect_paths(raw: Iterable[str]) -> Iterator[Path]:
    for item in raw:
        candidate: Final = Path(item)
        if candidate.is_dir():
            yield from sorted(candidate.rglob("*.py"))
        elif candidate.suffix == ".py":
            yield candidate


PARALLEL_MIN_PATHS = 200
MAX_WORKERS = 8


def _worker_count(path_count: int) -> int:
    """1 when the run is too small to repay process startup, else one worker per
    core up to MAX_WORKERS."""
    if path_count < PARALLEL_MIN_PATHS:
        return 1
    return max(1, min(os.cpu_count() or 1, MAX_WORKERS))


def scan_paths(paths: Sequence[Path]) -> tuple[Violation, ...]:
    """check_file over every path. Pure per-file work, so it fans out across
    processes; callers sort, which is what keeps output order stable."""
    workers = _worker_count(len(paths))
    if workers == 1:
        return tuple(v for path in paths for v in check_file(path))
    with Pool(workers) as pool:
        return tuple(v for found in pool.imap_unordered(check_file, paths, chunksize=32) for v in found)


def main(argv: Sequence[str]) -> int:
    paths: Final = tuple(a for a in argv if not a.startswith("-"))
    if not paths:
        print("usage: check_test_quality.py <files-or-dirs>...", file=sys.stderr)
        return 2

    targets: Final = tuple(collect_paths(paths))
    violations: Final = sorted(scan_paths(targets))
    for violation in violations:
        print(violation.render())

    if violations:
        print(f"\n{len(violations)} violation(s).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
