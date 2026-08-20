"""Regression tests for the diff-scoped mutation testing gate.

`.github/workflows/mutation-test-pr.yml` runs mutmut on pull requests, and the
only thing keeping that job inside a few minutes is `scripts/mutation_diff_scope.py`
narrowing mutmut down to what the diff touched. Two things have to hold, or the
job either burns hours or silently tests nothing:

- The mutant-name globs must address the exact functions containing the changed
  lines, spelled the way mutmut mangles them (`x_<func>` for a module-level
  function, `xǁ<Class>ǁ<method>` for a method). A glob that matches nothing makes
  mutmut abort; a glob that is too broad drags in unrelated functions.
- The test selection must stay inside `tests/test_litellm/`. `tests/e2e/` needs a
  live proxy and cannot run inside mutmut's `mutants/` sandbox, so pulling one in
  would fail the clean-test check and abort the run.

These tests drive the real scope builder against a throwaway git repository laid
out like this one, so a regression shows up as the wrong mutmut invocation.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "mutation_diff_scope", _REPO_ROOT / "scripts" / "mutation_diff_scope.py"
)
scope_module = importlib.util.module_from_spec(_spec)
# dataclasses resolves a frozen class's module through sys.modules at decoration time.
sys.modules[_spec.name] = scope_module
_spec.loader.exec_module(scope_module)

Scope = scope_module.Scope
build_scope = scope_module.build_scope
render_config = scope_module.render_config
rewrite_pyproject = scope_module.rewrite_pyproject
trampoline_units = scope_module.trampoline_units

PYPROJECT = """\
[project]
name = "litellm"

[tool.mutmut]
paths_to_mutate = [
    "litellm/proxy/management_endpoints/",
]
tests_dir = [
    "tests/test_litellm/proxy/management_endpoints/",
]
pytest_add_cli_args = [
    "-p", "no:retry",
]

[tool.coverage.run]
branch = true
"""

ROUTER = '''\
import functools


def _deployment_key(name: str) -> str:
    return name.strip()


class Router:
    def __init__(self, models: list[str]) -> None:
        self.models = models

    @functools.cached_property
    def get_model_list(self) -> list[str]:
        return [m for m in self.models if m]

    async def acompletion(self, model: str) -> str:
        return model
'''

AUTH_CHECKS = '''\
DEFAULT_ROLE = "user"


def can_call_model(model: str, allowed: list[str]) -> bool:
    return model in allowed
'''

TYPES = '''\
def coerce(value: str) -> str:
    return value
'''


def _run(root: Path, *args: str) -> None:
    subprocess.run(("git", *args), cwd=root, check=True, capture_output=True)


def _write(root: Path, path: str, body: str) -> None:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path.resolve()
    _run(root, "init", "-q", "-b", "base")
    _run(root, "config", "user.email", "test@example.com")
    _run(root, "config", "user.name", "test")

    _write(root, "pyproject.toml", PYPROJECT)
    _write(root, "litellm/router.py", ROUTER)
    _write(root, "litellm/proxy/auth/auth_checks.py", AUTH_CHECKS)
    _write(root, "litellm/types/utils.py", TYPES)
    _write(root, "tests/test_litellm/test_router.py", "def test_router(): pass\n")
    _write(root, "tests/test_litellm/proxy/auth/test_auth_checks.py", "def test_auth(): pass\n")
    _write(root, "tests/e2e/test_live_proxy.py", "def test_live(): pass\n")

    _run(root, "add", "-A")
    _run(root, "commit", "-qm", "base")
    _run(root, "checkout", "-q", "-b", "feature")
    return root


def _commit(root: Path, message: str = "change") -> None:
    _run(root, "add", "-A")
    _run(root, "commit", "-qm", message)


def scope_of(root: Path, max_functions: int = 40) -> Scope:
    return build_scope(root, "base", max_functions)


def test_changed_method_maps_to_its_mangled_mutant_glob(repo: Path) -> None:
    """A line inside a class method must address that method, class-qualified."""
    _write(repo, "litellm/router.py", ROUTER.replace("if m]", "if m and m.strip()]"))
    _commit(repo)

    assert scope_of(repo).globs == ("litellm.router.xǁRouterǁget_model_list__mutmut_*",)


def test_changed_module_level_function_uses_the_underscore_prefix(repo: Path) -> None:
    """Module-level functions mangle to `x_<name>`, not `xǁ...ǁ<name>`."""
    _write(repo, "litellm/proxy/auth/auth_checks.py", AUTH_CHECKS.replace("model in allowed", "model.lower() in allowed"))
    _commit(repo)

    assert scope_of(repo).globs == ("litellm.proxy.auth.auth_checks.x_can_call_model__mutmut_*",)


def test_touching_a_decorator_still_selects_the_decorated_function(repo: Path) -> None:
    """mutmut mutates the whole decorated function, so its decorator lines belong to it."""
    _write(repo, "litellm/router.py", ROUTER.replace("@functools.cached_property", "@functools.cache"))
    _commit(repo)

    assert scope_of(repo).globs == ("litellm.router.xǁRouterǁget_model_list__mutmut_*",)


def test_deletion_only_change_still_selects_the_function(repo: Path) -> None:
    """Removing a guard clause changes behavior, so its function has to stay in scope."""
    guarded = AUTH_CHECKS.replace(
        "    return model in allowed\n",
        "    if not model:\n        return False\n    return model in allowed\n",
    )
    _write(repo, "litellm/proxy/auth/auth_checks.py", guarded)
    _commit(repo, "add the guard")
    _run(repo, "checkout", "-q", "-b", "removal")
    _write(repo, "litellm/proxy/auth/auth_checks.py", AUTH_CHECKS)
    _commit(repo, "remove the guard")

    scope = build_scope(repo, "feature", 40)

    assert scope.globs == ("litellm.proxy.auth.auth_checks.x_can_call_model__mutmut_*",)


def test_module_level_change_selects_no_function(repo: Path) -> None:
    """mutmut only trampolines functions, so a module-level constant has nothing to run."""
    _write(repo, "litellm/proxy/auth/auth_checks.py", AUTH_CHECKS.replace('"user"', '"internal_user"'))
    _commit(repo)

    scope = scope_of(repo)
    assert scope.files, "the file must still register as changed"
    assert scope.globs == ()
    assert scope.is_runnable is False


def test_type_definitions_are_never_mutated(repo: Path) -> None:
    """litellm/types holds declarations; mutating them only produces noise."""
    _write(repo, "litellm/types/utils.py", TYPES.replace("return value", "return value.strip()"))
    _commit(repo)

    assert scope_of(repo).files == ()


def test_test_only_changes_produce_nothing_to_mutate(repo: Path) -> None:
    _write(repo, "tests/test_litellm/test_router.py", "def test_router(): assert True\n")
    _commit(repo)

    assert scope_of(repo).is_runnable is False


def test_test_selection_is_the_mirrored_file_plus_changed_unit_tests(repo: Path) -> None:
    _write(repo, "litellm/router.py", ROUTER.replace("if m]", "if m and m.strip()]"))
    _write(repo, "tests/test_litellm/proxy/auth/test_auth_checks.py", "def test_auth(): assert True\n")
    _commit(repo)

    assert scope_of(repo).tests == (
        "tests/test_litellm/proxy/auth/test_auth_checks.py",
        "tests/test_litellm/test_router.py",
    )


def test_e2e_tests_are_excluded_from_the_selection(repo: Path) -> None:
    """tests/e2e needs a live proxy; including one aborts the run at mutmut's clean-test check."""
    _write(repo, "litellm/router.py", ROUTER.replace("if m]", "if m and m.strip()]"))
    _write(repo, "tests/e2e/test_live_proxy.py", "def test_live(): assert True\n")
    _commit(repo)

    assert "tests/e2e/test_live_proxy.py" not in scope_of(repo).tests


def test_max_functions_caps_the_run_and_reports_what_it_dropped(repo: Path) -> None:
    """Silently truncating would read as full coverage of the diff."""
    _write(repo, "litellm/router.py", ROUTER.replace("return name.strip()", "return name.strip().lower()").replace("if m]", "if m and m.strip()]"))
    _commit(repo)

    uncapped = scope_of(repo)
    assert len(uncapped.globs) == 2

    capped = scope_of(repo, max_functions=1)
    assert len(capped.globs) == 1
    assert capped.dropped_globs == uncapped.globs[1:]


def test_rewritten_pyproject_scopes_mutmut_and_keeps_the_rest_of_the_file(repo: Path) -> None:
    _write(repo, "litellm/router.py", ROUTER.replace("if m]", "if m and m.strip()]"))
    _commit(repo)

    rewrite_pyproject(repo, scope_of(repo), ("-p", "no:retry"))
    config = tomllib.loads((repo / "pyproject.toml").read_text(encoding="utf-8"))

    assert config["tool"]["mutmut"]["paths_to_mutate"] == ["litellm/router.py"]
    assert config["tool"]["mutmut"]["tests_dir"] == ["tests/test_litellm/test_router.py"]
    assert config["tool"]["mutmut"]["mutate_only_covered_lines"] is False
    assert config["tool"]["mutmut"]["pytest_add_cli_args"] == ["-p", "no:retry"]
    assert config["tool"]["coverage"]["run"]["branch"] is True
    assert config["project"]["name"] == "litellm"


def test_render_config_copies_what_the_selected_tests_import(repo: Path) -> None:
    """mutmut runs from mutants/; a test importing litellm or repo tooling needs those copied in."""
    rendered = render_config(Scope(files=(), tests=(), globs=(), dropped_globs=()), ())

    assert tomllib.loads(rendered)["tool"]["mutmut"]["also_copy"] == ["litellm/", "scripts/", ".github/"]


def test_trampoline_units_skips_nested_functions(repo: Path) -> None:
    """mutmut folds a nested def into its parent, so selecting it by name would match nothing."""
    source = "def outer():\n    def inner():\n        return 1\n\n    return inner()\n"

    assert [unit.mangled_name for unit in trampoline_units(source)] == ["x_outer"]
