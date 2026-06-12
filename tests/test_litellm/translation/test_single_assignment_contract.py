"""Single-assignment contract: every name in litellm/translation binds once per scope.

The no-mutation semgrep tenets stop ``.append``/``[]=``/``+=`` but a plain
rebinding like ``x = f(x)`` slips through them (and through every ruff rule)
while reintroducing exactly the order-dependent local state the tenets exist
to keep out. This walks every module under ``litellm/translation`` and fails
on any name bound more than once in a scope. Mutually exclusive branches
(if/elif/else, match cases, except arms) may each bind the same name once,
and a branch that ends in return/raise/continue/break does not leak its
bindings past the statement. A for/while body is one loop generation, so a
fold like ``state, out = step(state, event)`` stays legal: loop-carried
accumulators are the same deliberate carve-out the mutation tenet grants
local accumulators.
"""

import ast
import pathlib

import litellm.translation as translation_pkg

_PACKAGE_ROOT = pathlib.Path(translation_pkg.__file__).parent

_Bindings = dict[str, int]


def _name_targets(target: ast.expr) -> list[ast.Name]:
    return [
        node
        for node in ast.walk(target)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    ]


def _pattern_captures(pattern: ast.pattern) -> list[tuple[str, int]]:
    captures: list[tuple[str, int]] = []
    for node in ast.walk(pattern):
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            captures.append((node.name, node.lineno))
        elif isinstance(node, ast.MatchMapping) and node.rest:
            captures.append((node.rest, node.lineno))
    return captures


def _own_expressions(stmt: ast.stmt) -> list[ast.expr]:
    expressions: list[ast.expr] = []
    for _, value in ast.iter_fields(stmt):
        if isinstance(value, ast.expr):
            expressions.append(value)
        elif isinstance(value, list):
            expressions.extend(item for item in value if isinstance(item, ast.expr))
    return expressions


def _argument_names(args: ast.arguments) -> list[ast.arg]:
    return [
        *args.posonlyargs,
        *args.args,
        *([args.vararg] if args.vararg else []),
        *args.kwonlyargs,
        *([args.kwarg] if args.kwarg else []),
    ]


def _terminates(body: list[ast.stmt]) -> bool:
    for stmt in body:
        if isinstance(stmt, (ast.Return, ast.Raise, ast.Continue, ast.Break)):
            return True
        if isinstance(stmt, ast.If) and stmt.orelse:
            if _terminates(stmt.body) and _terminates(stmt.orelse):
                return True
        if isinstance(stmt, ast.Match):
            exhaustive = any(
                isinstance(case.pattern, ast.MatchAs) and case.pattern.pattern is None
                for case in stmt.cases
            )
            if exhaustive and all(_terminates(case.body) for case in stmt.cases):
                return True
    return False


class _ScopeChecker:
    def __init__(self, relative_path: str) -> None:
        self.relative_path = relative_path
        self.violations: list[str] = []

    def check_scope(self, body: list[ast.stmt], initial: _Bindings) -> None:
        self._fold(body, dict(initial))

    def _bind(self, bound: _Bindings, name: str, lineno: int) -> None:
        if name == "_":
            return
        if name in bound:
            self.violations.append(
                f"{self.relative_path}:{lineno}: '{name}' rebound "
                f"(first bound at line {bound[name]})"
            )
        bound[name] = lineno

    def _bind_walrus_targets(
        self, expressions: list[ast.expr], bound: _Bindings
    ) -> None:
        for expression in expressions:
            for node in ast.walk(expression):
                if isinstance(node, ast.NamedExpr):
                    for name in _name_targets(node.target):
                        self._bind(bound, name.id, name.lineno)

    def _fold(self, body: list[ast.stmt], bound: _Bindings) -> _Bindings:
        for stmt in body:
            self._fold_statement(stmt, bound)
        return bound

    def _merge_exclusive_branches(
        self, bound: _Bindings, branches: list[tuple[_Bindings, list[ast.stmt]]]
    ) -> None:
        merged = dict(bound)
        for seed, branch_body in branches:
            folded = self._fold(branch_body, seed)
            if not _terminates(branch_body):
                for name, lineno in folded.items():
                    merged.setdefault(name, lineno)
        bound.update(merged)

    def _fold_statement(self, stmt: ast.stmt, bound: _Bindings) -> None:
        self._bind_walrus_targets(_own_expressions(stmt), bound)
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                for name in _name_targets(target):
                    self._bind(bound, name.id, name.lineno)
        elif isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None and isinstance(stmt.target, ast.Name):
                self._bind(bound, stmt.target.id, stmt.target.lineno)
        elif isinstance(stmt, ast.AugAssign):
            for name in _name_targets(stmt.target):
                self._bind(bound, name.id, name.lineno)
        elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._bind(bound, stmt.name, stmt.lineno)
            parameters = {arg.arg: stmt.lineno for arg in _argument_names(stmt.args)}
            self.check_scope(stmt.body, parameters)
        elif isinstance(stmt, ast.ClassDef):
            self._bind(bound, stmt.name, stmt.lineno)
            self.check_scope(stmt.body, {})
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for alias in stmt.names:
                bound_name = alias.asname or alias.name.split(".")[0]
                self._bind(bound, bound_name, stmt.lineno)
        elif isinstance(stmt, ast.If):
            self._merge_exclusive_branches(
                bound, [(dict(bound), stmt.body), (dict(bound), stmt.orelse)]
            )
        elif isinstance(stmt, ast.Match):
            branches: list[tuple[_Bindings, list[ast.stmt]]] = []
            for case in stmt.cases:
                seed = dict(bound)
                for capture_name, capture_lineno in _pattern_captures(case.pattern):
                    if capture_name != "_" and capture_name in bound:
                        self.violations.append(
                            f"{self.relative_path}:{capture_lineno}: "
                            f"'{capture_name}' rebound by match capture "
                            f"(first bound at line {bound[capture_name]})"
                        )
                    seed[capture_name] = capture_lineno
                branches.append((seed, case.body))
            self._merge_exclusive_branches(bound, branches)
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            generation: _Bindings = {}
            for name in _name_targets(stmt.target):
                self._bind(generation, name.id, name.lineno)
            folded = self._fold(stmt.body, generation)
            self._fold(stmt.orelse, dict(bound))
            for bound_name, lineno in folded.items():
                bound.setdefault(bound_name, lineno)
        elif isinstance(stmt, ast.While):
            folded = self._fold(stmt.body, {})
            self._fold(stmt.orelse, dict(bound))
            for bound_name, lineno in folded.items():
                bound.setdefault(bound_name, lineno)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                if item.optional_vars is not None:
                    for name in _name_targets(item.optional_vars):
                        self._bind(bound, name.id, name.lineno)
            self._fold(stmt.body, bound)
        elif isinstance(stmt, ast.Try):
            branches = [(dict(bound), stmt.body + stmt.orelse)]
            for handler in stmt.handlers:
                handler_seed = dict(bound)
                if handler.name:
                    handler_seed[handler.name] = handler.lineno
                branches.append((handler_seed, handler.body))
            self._merge_exclusive_branches(bound, branches)
            self._fold(stmt.finalbody, bound)


def _module_violations(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    checker = _ScopeChecker(str(path.relative_to(_PACKAGE_ROOT)))
    checker.check_scope(tree.body, {})
    return checker.violations


def _check_source(source: str) -> list[str]:
    checker = _ScopeChecker("<fixture>")
    checker.check_scope(ast.parse(source).body, {})
    return checker.violations


def test_checker_flags_a_sequential_rebinding() -> None:
    violations = _check_source("def f():\n    x = 1\n    x = 2\n")
    assert len(violations) == 1 and "'x' rebound" in violations[0]


def test_checker_flags_a_parameter_shadow() -> None:
    assert len(_check_source("def f(x):\n    x = 1\n")) == 1


def test_checker_flags_conditional_overwrite_of_an_outer_binding() -> None:
    source = "def f(c):\n    x = 1\n    if c:\n        x = 2\n    return x\n"
    assert len(_check_source(source)) == 1


def test_checker_flags_an_augmented_assignment() -> None:
    assert len(_check_source("def f():\n    x = 1\n    x += 1\n")) == 1


def test_checker_flags_a_walrus_rebinding() -> None:
    assert len(_check_source("def f():\n    x = 1\n    y = (x := 2)\n")) == 1


def test_checker_flags_a_match_capture_shadow() -> None:
    source = (
        "def f(v):\n"
        "    item = 1\n"
        "    match v:\n"
        "        case [item]:\n"
        "            return item\n"
    )
    assert len(_check_source(source)) == 1


def test_checker_allows_one_binding_per_exclusive_branch() -> None:
    source = (
        "def f(c):\n    if c:\n        x = 1\n    else:\n        x = 2\n    return x\n"
    )
    assert _check_source(source) == []


def test_checker_allows_one_binding_per_match_case() -> None:
    source = (
        "def f(v):\n"
        "    match v:\n"
        "        case 1:\n"
        "            x = 'one'\n"
        "        case _:\n"
        "            x = 'other'\n"
        "    return x\n"
    )
    assert _check_source(source) == []


def test_checker_allows_a_binding_after_a_terminating_branch() -> None:
    source = "def f(c):\n    if c:\n        x = 1\n        return x\n    x = 2\n    return x\n"
    assert _check_source(source) == []


def test_checker_allows_a_loop_carried_fold() -> None:
    source = (
        "def f(items, step, s0):\n"
        "    state = s0\n"
        "    for item in items:\n"
        "        state, out = step(state, item)\n"
        "    return state\n"
    )
    assert _check_source(source) == []


def test_checker_flags_a_rebinding_within_one_loop_iteration() -> None:
    source = "def f(items):\n    for item in items:\n        x = 1\n        x = 2\n"
    assert len(_check_source(source)) == 1


def test_no_name_is_rebound_within_a_scope() -> None:
    offenders = [
        violation
        for path in sorted(_PACKAGE_ROOT.rglob("*.py"))
        for violation in _module_violations(path)
    ]
    assert offenders == [], "rebindings:\n" + "\n".join(offenders)
