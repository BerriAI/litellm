"""A compat cell must drive its own deployment, not the shared one.

Markerless harness tests. The shared virtual names are registered once per
session and per xdist worker, so a call to one belongs to no test and cannot be
cached. These cover the per-cell copy that replaces them, and the one thing that
makes it worth anything: that the driver actually sends the copy's name.
"""

from __future__ import annotations

import subprocess
from typing import Dict, List, Optional, Tuple

import pytest

from claude_code import _compat_aliases
from claude_code._compat_aliases import AliasRegistry
from claude_code._compat_models import CompatDeployment
from claude_code.cli_driver import run_claude
from claude_code.rate_limiter import RateLimiter

_SHARED_NAME = "claude-sonnet-4-5-bedrock-invoke"
_NODE = "claude_code/tool_use/test_bedrock_invoke.py::test_tool_use_bedrock_invoke"


class _FakeRegistrar:
    def __init__(self) -> None:
        self.created: List[Tuple[str, object]] = []
        self.deleted: List[str] = []

    def create_model(self, model_name: str, litellm_params: object) -> str:
        self.created.append((model_name, litellm_params))
        return f"id-{len(self.created)}"

    def delete_model(self, model_id: str) -> None:
        self.deleted.append(model_id)


@pytest.fixture(name="deployments")
def _deployments() -> Dict[str, CompatDeployment]:
    params = CompatDeployment(model_name=_SHARED_NAME, litellm_params=None)  # pyright: ignore[reportArgumentType]  # only identity matters here
    return {_SHARED_NAME: params}


def test_two_cells_get_different_names_for_the_same_shared_model(
    deployments: Dict[str, CompatDeployment]
) -> None:
    registrar = _FakeRegistrar()
    mine = AliasRegistry(registrar, deployments, _NODE).alias(_SHARED_NAME)
    theirs = AliasRegistry(registrar, deployments, _NODE.replace("tool_use", "vision")).alias(_SHARED_NAME)

    assert mine != theirs, "two cells sharing one name is what leaves the router load-balancing between them"
    assert mine.startswith(_SHARED_NAME)


def test_the_same_cell_gets_the_same_name_in_a_later_build(
    deployments: Dict[str, CompatDeployment]
) -> None:
    first = AliasRegistry(_FakeRegistrar(), deployments, _NODE).alias(_SHARED_NAME)
    second = AliasRegistry(_FakeRegistrar(), deployments, _NODE).alias(_SHARED_NAME)

    assert first == second, "a name that moves between builds takes the cache key with it"


def test_a_cell_registers_its_copy_once_however_often_it_asks(
    deployments: Dict[str, CompatDeployment]
) -> None:
    registrar = _FakeRegistrar()
    registry = AliasRegistry(registrar, deployments, _NODE)

    names = {registry.alias(_SHARED_NAME) for _ in range(5)}

    assert len(names) == 1
    assert len(registrar.created) == 1


def test_a_name_the_matrix_does_not_own_is_left_alone(
    deployments: Dict[str, CompatDeployment]
) -> None:
    registrar = _FakeRegistrar()

    assert AliasRegistry(registrar, deployments, _NODE).alias("somebody-elses-model") == "somebody-elses-model"
    assert registrar.created == []


def test_teardown_deletes_every_copy_the_cell_registered(
    deployments: Dict[str, CompatDeployment]
) -> None:
    registrar = _FakeRegistrar()
    registry = AliasRegistry(registrar, deployments, _NODE)
    registry.alias(_SHARED_NAME)

    registry.teardown()

    assert registrar.deleted == ["id-1"]


def test_the_driver_sends_the_cells_own_name(deployments: Dict[str, CompatDeployment], tmp_path: object) -> None:
    sent: List[str] = []

    def fake_run(cmd: List[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        sent.append(cmd[cmd.index("--model") + 1])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    registry = AliasRegistry(_FakeRegistrar(), deployments, _NODE)
    _compat_aliases.install(registry)
    try:
        run_claude(
            prompt="hi",
            model=_SHARED_NAME,
            base_url="http://127.0.0.1:1",
            api_key="stub",
            runner=fake_run,
            rate_limiter=RateLimiter(state_dir=tmp_path / "limiter"),  # pyright: ignore[reportAttributeAccessIssue]  # tmp_path is a Path
        )
    finally:
        _compat_aliases.install(None)

    assert sent == [registry.alias(_SHARED_NAME)]
    assert sent != [_SHARED_NAME]


def test_the_driver_sends_the_plain_name_when_no_cell_owns_it(tmp_path: object) -> None:
    sent: List[str] = []

    def fake_run(cmd: List[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        sent.append(cmd[cmd.index("--model") + 1])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    _compat_aliases.install(None)
    run_claude(
        prompt="hi",
        model=_SHARED_NAME,
        base_url="http://127.0.0.1:1",
        api_key="stub",
        runner=fake_run,
        rate_limiter=RateLimiter(state_dir=tmp_path / "limiter"),  # pyright: ignore[reportAttributeAccessIssue]  # tmp_path is a Path
    )

    assert sent == [_SHARED_NAME]
