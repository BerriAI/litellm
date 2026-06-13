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


# --- the ONE recursion-depth-cap truth (wave-2b sibling-merge unification) ---
#
# verifier-wave2b-beta F2's lesson generalized: a hand-copied depth cap (the
# old mistral _REFS_MAX_DEPTH = 10, the old compat_sdk cache scan cap of 16)
# drifts from the constant v1's call sites actually read. The beta pattern —
# import litellm.constants.DEFAULT_MAX_RECURSE_DEPTH directly, drift
# impossible by construction — is now the package rule, and this gate covers
# EVERY consumer instead of mistral alone (the integrator queue's depth-cap
# unification item). APPEND a module name below when a new consumer lands.

_DEPTH_CAP_CONSUMERS = (
    "providers.databricks.tools",
    "providers.google_genai.schema",
    "providers.mistral.serialize",
    "providers.openai_compat.guard",
    "providers.openai_compat.serialize",
    "providers.watsonx.serialize",
    "providers.xai.guard",
)


def test_every_depth_cap_reads_v1s_one_constant() -> None:
    """Each consumer's binding must BE litellm.constants'
    DEFAULT_MAX_RECURSE_DEPTH (identity, not equality) and arrive via a
    direct ``from litellm.constants import`` — and the discovered consumer
    set must match the pinned table, so both adding a hand-rolled cap and
    silently dropping the constant are loud."""
    import importlib

    from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH

    discovered = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "litellm.constants"
                and any(a.name == "DEFAULT_MAX_RECURSE_DEPTH" for a in node.names)
            ):
                rel = path.relative_to(_PACKAGE_ROOT).with_suffix("")
                discovered.append(".".join(rel.parts))
    assert tuple(sorted(discovered)) == _DEPTH_CAP_CONSUMERS, (
        f"depth-cap consumer drift: discovered {sorted(discovered)!r}; update "
        "the pinned table in the same commit as the consumer"
    )
    for name in _DEPTH_CAP_CONSUMERS:
        module = importlib.import_module(f"litellm.translation.{name}")
        assert module.DEFAULT_MAX_RECURSE_DEPTH is DEFAULT_MAX_RECURSE_DEPTH, (
            f"{name} stopped reading v1's constant"
        )


def _is_depth_named(name: str) -> bool:
    return "depth" in name.lower()


def _is_int_literal(node: ast.expr | None) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    )


def _is_positive_int_literal(node: ast.expr | None) -> bool:
    return _is_int_literal(node) and isinstance(node, ast.Constant) and node.value > 0


def _compare_offends(node: ast.Compare) -> bool:
    operands = (node.left, *node.comparators)
    named = any(
        isinstance(n, ast.Name) and _is_depth_named(n.id) for n in operands
    )
    return named and any(_is_int_literal(n) for n in operands)


def _binding_offends(node: ast.Assign | ast.AnnAssign) -> bool:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    named = any(
        isinstance(t, ast.Name) and _is_depth_named(t.id) for t in targets
    )
    return named and _is_positive_int_literal(node.value)


def _arg_default_offends(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    positional = [*node.args.posonlyargs, *node.args.args]
    defaulted = positional[len(positional) - len(node.args.defaults) :]
    pairs = [
        *zip(defaulted, node.args.defaults),
        *zip(node.args.kwonlyargs, node.args.kw_defaults),
    ]
    return any(
        _is_depth_named(arg.arg) and _is_positive_int_literal(default)
        for arg, default in pairs
    )


def _hand_rolled_depth_caps(tree: ast.AST) -> list[int]:
    """Line numbers of int-literal depth caps in any of the three shapes the
    gate bans (extracted so the negative tests below can probe planted
    sources without mutating the tree)."""
    offences: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and _compare_offends(node):
            offences.append(node.lineno)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and _binding_offends(node):
            offences.append(node.lineno)
        elif isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ) and _arg_default_offends(node):
            offences.append(node.lineno)
    return offences


def test_no_module_hand_rolls_a_numeric_depth_cap() -> None:
    """No int-literal depth cap anywhere in the package, in ANY of the three
    shapes: a ``*depth*`` name compared against an int literal, an
    int-literal ASSIGNMENT to a ``*depth*``-named binding (beta F2's
    renamed-constant shape, ``_REFS_MAX_DEPTH = 10``), or a positive
    int-literal DEFAULT on a ``*depth*``-named parameter (the dead
    ``max_depth: int = 10`` shape). Caps must read the imported constant.
    critic-wave2b-final MAJOR-1: the compare-only scan missed the two
    binding-routed shapes — both are now negative-tested below. A ``0``
    default/binding stays allowed: it is a recursion COUNTER's start, not a
    cap (openai_compat.guard.carries_cache_control). Disclosed residual
    (verifier-wave2b-final F5): a cap whose name avoids "depth" (e.g.
    ``level``) slips this name heuristic — review scope; the identity gate
    above still catches all import drift."""
    offenders = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        offenders.extend(
            f"{path.relative_to(_PACKAGE_ROOT)}:{lineno}"
            for lineno in _hand_rolled_depth_caps(tree)
        )
    assert offenders == [], "hand-rolled depth caps: " + "; ".join(offenders)


def test_depth_gate_catches_a_renamed_constant_binding() -> None:
    """critic-wave2b-final MAJOR-1 probe 1, now a pinned negative test: the
    historical mistral ``_REFS_MAX_DEPTH = 10`` shape routes the literal
    through a module binding and compares name-vs-name — the compare-only
    scan passed it; the binding scan must flag it."""
    planted = ast.parse(
        "_REFS_MAX_DEPTH = 10\n"
        "def _walk(value, depth):\n"
        "    if depth >= _REFS_MAX_DEPTH:\n"
        "        return None\n"
        "    return _walk(value, depth + 1)\n"
    )
    assert _hand_rolled_depth_caps(planted) == [1]


def test_depth_gate_catches_an_int_literal_parameter_default() -> None:
    """critic-wave2b-final MAJOR-1 probe 2, now a pinned negative test: the
    dead ``max_depth: int = 10`` default with name-vs-name compares only —
    the exact F2 drift shape the original gate claimed to ban but missed.
    A ``depth: int = 0`` start counter stays green (the live
    carries_cache_control signature)."""
    planted = ast.parse(
        "def _mut_strip(value, max_depth: int = 10):\n"
        "    if max_depth <= len(str(value)):\n"
        "        return value\n"
        "    return _mut_strip(value, max_depth - 1)\n"
    )
    assert _hand_rolled_depth_caps(planted) == [1]
    counter = ast.parse("def _scan(value, depth: int = 0):\n    return depth\n")
    assert _hand_rolled_depth_caps(counter) == []
