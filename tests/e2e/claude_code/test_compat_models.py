"""Harness tests for compat deployment registration.

No `e2e` marker and no proxy: `register_deployments` /
`unregister_deployments` take the "register one" and "delete one"
callables as parameters, so the concurrency and the errors-as-values
contract are pinned here with fakes.
"""

from __future__ import annotations

import threading
from typing import List

import pytest

from claude_code._compat_models import (
    CompatDeployment,
    DeploymentFailure,
    RegistrationOutcome,
    register_deployments,
    unregister_deployments,
)
from models import LiteLLMParamsBody


BARRIER_TIMEOUT_SECONDS = 10.0


def _deployments(count: int) -> tuple[CompatDeployment, ...]:
    return tuple(
        CompatDeployment(
            model_name=f"model-{index}",
            litellm_params=LiteLLMParamsBody(model=f"anthropic/model-{index}"),
        )
        for index in range(count)
    )


def test_deployments_register_concurrently() -> None:
    """Registration is one POST plus a poll per deployment, so they all go
    out at once: the barrier only trips if every registration is in
    flight together."""
    deployments = _deployments(4)
    barrier = threading.Barrier(len(deployments), timeout=BARRIER_TIMEOUT_SECONDS)

    def register(deployment: CompatDeployment) -> str:
        barrier.wait()
        return f"id-{deployment.model_name}"

    outcome = register_deployments(deployments, register)

    assert outcome.failures == ()
    assert sorted(outcome.model_ids) == [f"id-model-{index}" for index in range(4)]


def test_one_unservable_deployment_does_not_abort_the_batch() -> None:
    """A deployment the proxy has no credential for must come back as a
    failure value; the cells that target it fail loudly on their own,
    and every other deployment still registers."""
    deployments = _deployments(3)

    def register(deployment: CompatDeployment) -> str:
        if deployment.model_name == "model-1":
            raise AssertionError("no credential on the proxy")
        return f"id-{deployment.model_name}"

    outcome = register_deployments(deployments, register)

    assert outcome.failures == (
        DeploymentFailure(
            model_name="model-1",
            reason="AssertionError: no credential on the proxy",
        ),
    )
    assert sorted(outcome.model_ids) == ["id-model-0", "id-model-2"]


def test_registering_nothing_touches_the_proxy_not_at_all() -> None:
    def register(deployment: CompatDeployment) -> str:
        raise AssertionError(f"must not register {deployment.model_name}")

    assert register_deployments((), register) == RegistrationOutcome(
        model_ids=(), failures=()
    )


def test_deletes_run_concurrently() -> None:
    model_ids = ("id-1", "id-2", "id-3")
    barrier = threading.Barrier(len(model_ids), timeout=BARRIER_TIMEOUT_SECONDS)

    def delete(model_id: str) -> None:
        barrier.wait()

    assert unregister_deployments(model_ids, delete) == ()


def test_a_failed_delete_is_reported_and_the_rest_still_run() -> None:
    deleted: List[str] = []
    lock = threading.Lock()

    def delete(model_id: str) -> None:
        with lock:
            deleted.append(model_id)
        if model_id == "id-2":
            raise RuntimeError("already gone")

    failures = unregister_deployments(("id-1", "id-2", "id-3"), delete)

    assert sorted(deleted) == ["id-1", "id-2", "id-3"]
    assert failures == (
        DeploymentFailure(model_name="id-2", reason="RuntimeError: already gone"),
    )


def test_deleting_nothing_is_a_no_op() -> None:
    def delete(model_id: str) -> None:
        raise AssertionError(f"must not delete {model_id}")

    assert unregister_deployments((), delete) == ()


@pytest.mark.parametrize("count", [1, 5])
def test_every_deployment_is_registered_exactly_once(count: int) -> None:
    registered: List[str] = []
    lock = threading.Lock()

    def register(deployment: CompatDeployment) -> str:
        with lock:
            registered.append(deployment.model_name)
        return f"id-{deployment.model_name}"

    outcome = register_deployments(_deployments(count), register)

    assert sorted(registered) == [f"model-{index}" for index in range(count)]
    assert len(outcome.model_ids) == count
