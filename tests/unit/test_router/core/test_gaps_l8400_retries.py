"""
Gaps in the retry loop and the per-attempt bookkeeping of ``PythonCore`` (router_python_core.py
lines 8400-11100): the ``UnsupportedEndpointError`` skip-and-stop rule, the retry breadcrumb cap and
``base_model``, and the request the async failure callback counts toward group usage.
"""

from typing import Final

import pytest

import litellm
from tests.unit.test_router.core.retries_health_helpers import (
    BackgroundSuccessLoggingHost,
    MakeCore,
    MetadataRecorder,
    attempt,
)
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment, rate_limit, server_error


def unsupported_endpoint() -> litellm.UnsupportedEndpointError:
    return litellm.UnsupportedEndpointError(message="endpoint not supported", llm_provider="openai", model="gpt-4o")


# --- UnsupportedEndpointError: retried on the other deployments, stopped once every one refused ---------


@pytest.mark.asyncio
async def test_unsupported_endpoint_is_retried_once_per_deployment_then_raised(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    """A 400 is normally never retried; ``UnsupportedEndpointError`` is, but each deployment that refused
    is skipped and the loop stops as soon as every deployment of the group refused (num_retries=3 unused)."""
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b")], num_retries=3)
    host.answer("*", lambda **_: unsupported_endpoint())

    error: Final = await attempt(core)

    assert isinstance(error, litellm.UnsupportedEndpointError)
    called: Final = host.deployments_called("acompletion")
    assert sorted(called) == ["a", "b"]
    assert host.sleeps == [0]


@pytest.mark.asyncio
async def test_unsupported_endpoint_on_a_lone_deployment_is_raised_without_a_retry(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], num_retries=3)
    host.answer("*", lambda **_: unsupported_endpoint())

    error: Final = await attempt(core)

    assert isinstance(error, litellm.UnsupportedEndpointError)
    assert host.deployments_called("acompletion") == ["a"]
    assert host.sleeps == []


# --- retry breadcrumbs ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_previous_models_keeps_only_the_four_most_recent_failures(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    recorder: Final = MetadataRecorder()
    core: Final = make_core(model_list=[deployment("g", "a")], num_retries=6)
    host.answer("*", *([recorder.then(server_error)] * 6), recorder.then(None))

    response: Final = await attempt(core)

    assert not isinstance(response, Exception)
    trails: Final = [snapshot["previous_models"] for snapshot in recorder.snapshots]
    assert trails[0] is None
    assert [len(trail) for trail in trails[1:]] == [1, 2, 3, 4, 4, 4]  # pyright: ignore[reportArgumentType]
    assert [entry["attempted_retries"] for entry in trails[-1]] == [2, 3, 4, 5]  # pyright: ignore[reportGeneralTypeIssues, reportOptionalIterable]
    assert [snapshot["request_retry_count"] for snapshot in recorder.snapshots] == [None, 1, 2, 3, 4, 5, 6]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"base_model": "azure/gpt-4o"}, "azure/gpt-4o"),
        ({"model_info": {"base_model": "azure/gpt-4o-mini"}}, "azure/gpt-4o-mini"),
        ({}, None),
    ],
    ids=["litellm-params-base-model", "model-info-base-model", "none"],
)
async def test_a_retry_breadcrumb_names_the_deployments_base_model(
    make_core: MakeCore, host: ScriptedHost, params: dict[str, object], expected: str | None
) -> None:
    recorder: Final = MetadataRecorder()
    core: Final = make_core(model_list=[deployment("g", "a", **params)], num_retries=1)
    host.answer("*", recorder.then(server_error), recorder.then(None))

    await attempt(core)

    (entry,) = recorder.snapshots[1]["previous_models"]  # pyright: ignore[reportGeneralTypeIssues]
    assert entry["base_model"] == expected  # pyright: ignore[reportIndexIssue]


# --- the async failure callback counts the failed request ---------------------------------------


@pytest.mark.asyncio
async def test_a_failed_call_counts_a_request_but_no_tokens_in_group_usage(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", rpm=5)], num_retries=0)
    host.answer("*", rate_limit())

    error: Final = await attempt(core)

    assert isinstance(error, litellm.RateLimitError)
    assert await core.health().usage("g") == (None, 1)


class TestUsageWhenTheHostLogsSuccessAfterTheResponse:
    """litellm runs the async success callbacks after the endpoint returned (``ScriptedHost`` runs them
    before, which lets the success callback and the router's own count both add the same response)."""

    @pytest.fixture
    def host(self) -> BackgroundSuccessLoggingHost:
        return BackgroundSuccessLoggingHost()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("failures", "requests"), [(0, 1), (1, 2), (2, 3)])
    async def test_every_failed_attempt_and_the_final_success_count_as_requests(
        self, make_core: MakeCore, host: BackgroundSuccessLoggingHost, failures: int, requests: int
    ) -> None:
        core: Final = make_core(model_list=[deployment("g", "a", rpm=50)], num_retries=2)
        host.answer("*", *([server_error()] * failures), None)

        await attempt(core)
        await host.drain()

        assert await core.health().usage("g") == (10, requests)
