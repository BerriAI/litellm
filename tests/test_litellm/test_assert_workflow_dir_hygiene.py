"""Tests for .github/scripts/assert_workflow_dir_hygiene.py."""

import importlib.util
import sys
from pathlib import Path
from typing import Final

import pytest

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_MODULE_PATH: Final = _REPO_ROOT / ".github" / "scripts" / "assert_workflow_dir_hygiene.py"
_spec: Final = importlib.util.spec_from_file_location("assert_workflow_dir_hygiene", _MODULE_PATH)
hygiene: Final = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = hygiene  # @dataclass(slots=True) rebuilds via sys.modules
_spec.loader.exec_module(hygiene)


def _codes(path_name, triggers):
    return [f.code for f in hygiene._naming_findings(Path(path_name), frozenset(triggers))]


def test_a_call_only_workflow_without_the_prefix_is_flagged():
    assert _codes("deploy.yml", {"workflow_call"}) == ["WF002"]


def test_a_call_only_workflow_with_the_prefix_is_clean():
    assert _codes("_deploy.yml", {"workflow_call"}) == []


def test_a_dual_mode_workflow_keeps_its_plain_name():
    # workflow_call plus a human trigger is deliberate: the `_` prefix would hide a
    # workflow someone is meant to be able to dispatch.
    assert _codes("create-release-branch.yml", {"workflow_call", "workflow_dispatch"}) == []


def test_a_prefixed_workflow_nobody_can_call_is_flagged():
    assert _codes("_helper.yml", {"push"}) == ["WF003"]


def test_a_plain_workflow_with_ordinary_triggers_is_clean():
    assert _codes("test-unit.yml", {"pull_request", "push"}) == []


@pytest.mark.parametrize(
    "raw, expected",
    [
        ({"on": "push"}, {"push"}),
        ({"on": ["push", "pull_request"]}, {"push", "pull_request"}),
        ({"on": {"workflow_call": None}}, {"workflow_call"}),
        ({True: {"pull_request": None}}, {"pull_request"}),
        ({"jobs": {}}, set()),
        ("not a mapping", set()),
    ],
)
def test_triggers_reads_every_shape_the_on_key_takes(raw, expected):
    # YAML 1.1 turns a bare `on:` key into the boolean True, which is why the loaded
    # document has to be read both ways.
    assert hygiene._triggers(raw) == frozenset(expected)


def test_the_repo_as_it_stands_holds_only_workflows_in_the_workflow_dir():
    assert [f.subject for f in hygiene._strays()] == []


def test_the_repo_as_it_stands_names_every_reusable_workflow_with_the_prefix():
    assert [f.subject for f in hygiene._misnamed()] == []
