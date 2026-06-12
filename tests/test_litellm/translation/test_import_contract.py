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


def test_no_module_hand_rolls_a_numeric_depth_cap() -> None:
    """No comparison of a ``*depth*`` name against an int literal anywhere in
    the package: caps must compare against the imported constant (the rule
    that makes the unification stick — a literal 10/16/100 is exactly the
    drift shape F2 refuted)."""
    offenders = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            named = [
                n
                for n in (node.left, *node.comparators)
                if isinstance(n, ast.Name) and "depth" in n.id.lower()
            ]
            literal_int = [
                n
                for n in (node.left, *node.comparators)
                if isinstance(n, ast.Constant) and isinstance(n.value, int)
            ]
            if named and literal_int:
                offenders.append(
                    f"{path.relative_to(_PACKAGE_ROOT)}:{node.lineno}"
                )
    assert offenders == [], "hand-rolled depth caps: " + "; ".join(offenders)
