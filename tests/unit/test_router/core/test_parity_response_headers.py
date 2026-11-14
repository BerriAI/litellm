"""
Parity of the Rust core with ``router_python_core.py`` on the router's response headers, what a sync
pick filters, and the names the no-deployments errors carry.
"""

import asyncio
import warnings
from collections.abc import Callable, Mapping
from typing import Final

import pytest
from pydantic import ValidationError

import litellm
from litellm.exceptions import BadRequestError, EndpointNotSupportedByModelError
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.router import RouterRateLimitError
from tests.unit.test_router.core.fallbacks_helpers import MESSAGES, MakeCore, complete, complete_sync, headers
from tests.unit.test_router.core.scripted_host import CORE_BUILDERS, ScriptedHost, completion_response, deployment, server_error

GROUP_LIMITS: Final = {"rpm": 5, "tpm": 1000}
TEAM_OWNED: Final = {"team_id": "T1"}
UNREADABLE_LIMIT: Final = {"rpm": "abc"}
RETRY_COUNT: Final = "x-litellm-attempted-retries"
FALLBACK_COUNT: Final = "x-litellm-attempted-fallbacks"
GUARDRAILS: Final = [{"guardrail_name": "gr", "litellm_params": {"guardrail": "aporia", "mode": "pre_call"}}]
BLOCKED_RESPONSE: Final = {
    "id": "chatcmpl-blocked",
    "object": "chat.completion",
    "model": "gpt-4o",
    "created": 1,
    "choices": [{"index": 0, "finish_reason": "content_filter", "message": {"role": "assistant", "content": "no"}}],
}
# The dumps of a deployment `_create_deployment` makes, each of which warns about a mocked ModelResponse.
PYTHON_DUMPS_OF_A_DEPLOYMENT: Final = 4


# --- x-ratelimit-* and the counts ------------------------------------------------------------------


def test_a_sync_call_reads_the_remaining_usage_of_its_group(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1", **GROUP_LIMITS)], num_retries=0)
    stamped: Final = headers(complete_sync(core, "g"))
    assert stamped["x-ratelimit-limit-requests"] == 5
    assert stamped["x-ratelimit-remaining-requests"] == 4
    assert stamped["x-ratelimit-limit-tokens"] == 1000
    assert stamped[RETRY_COUNT] == 0
    assert stamped[FALLBACK_COUNT] == 0


async def test_an_unreadable_group_limit_leaves_the_counts_off_the_response(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    answered: Final = completion_response()
    host.answer("openai/d1", answered)
    core: Final = make_core(model_list=[deployment("g", "d1", model_info=UNREADABLE_LIMIT)], num_retries=0)
    with pytest.raises(ValidationError):
        await complete(core, "g")
    stamped: Final = headers(answered)
    assert stamped["x-litellm-model-group"] == "g"
    assert RETRY_COUNT not in stamped
    assert FALLBACK_COUNT not in stamped


def test_a_sync_call_reads_the_group_limit_too(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1", model_info=UNREADABLE_LIMIT)], num_retries=0)
    with pytest.raises(ValidationError):
        complete_sync(core, "g")


def test_a_sync_call_opens_at_most_one_event_loop(make_core: MakeCore, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each loop of its own hands the GIL to the logging thread before the headers are on the response."""
    opened: Final[list[object]] = []  # mutable-ok: filled by the patched factory
    real: Final[Callable[[], asyncio.AbstractEventLoop]] = asyncio.new_event_loop

    def counted() -> asyncio.AbstractEventLoop:
        loop: Final = real()
        opened.append(loop)
        return loop

    core: Final = make_core(model_list=[deployment("g", "d1", **GROUP_LIMITS)], num_retries=0)
    monkeypatch.setattr(asyncio, "new_event_loop", counted)
    _ = complete_sync(core, "g")
    assert len(opened) <= 1


class _FallbackCountAtLog(CustomLogger):
    def __init__(self, response: object) -> None:
        super().__init__()
        self._response: Final = response
        self.seen: Final[list[object]] = []  # mutable-ok: filled as the events log

    async def log_success_fallback_event(
        self, original_model_group: str, kwargs: dict, original_exception: Exception
    ) -> None:
        self.seen.append(headers(self._response).get(FALLBACK_COUNT))


async def test_a_fallback_counts_itself_before_its_success_is_logged(
    make_core: MakeCore, host: ScriptedHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_async_fallback`` stamps the response, then awaits ``log_success_fallback_event``."""
    answered: Final = completion_response()
    logger: Final = _FallbackCountAtLog(answered)
    monkeypatch.setattr(litellm, "callbacks", [logger])
    host.answer("openai/a1", server_error())
    host.answer("openai/b1", answered)
    core: Final = make_core(
        model_list=[deployment("a", "a1"), deployment("b", "b1")], num_retries=0, fallbacks=[{"a": ["b"]}]
    )
    _ = await complete(core, "a")
    assert logger.seen == [1]


# --- guardrails ------------------------------------------------------------------------------------


async def test_a_guardrail_result_carries_the_router_headers(make_core: MakeCore) -> None:
    answered: Final = completion_response()

    async def guard(**_kwargs: object) -> object:  # kwargs-ok: the guardrail's keyword payload
        return answered

    core: Final = make_core(model_list=[deployment("g", "d1")], guardrail_list=GUARDRAILS)
    result: Final = await core.aguardrail(guardrail_name="gr", original_function=guard)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint
    assert headers(result) == {"x-litellm-model-group": "gr", RETRY_COUNT: 0, FALLBACK_COUNT: 0}


# --- the team filter -------------------------------------------------------------------------------


# S2 and Y4 are ruled to differ: the Rust core filters every pick, so these run on it only.
def test_the_rust_core_filters_a_team_owned_deployment_on_a_sync_pick(host: ScriptedHost) -> None:
    core: Final = CORE_BUILDERS["rust"](host, model_list=[deployment("own", "o1", model_info=TEAM_OWNED)], num_retries=0)
    with pytest.raises(RouterRateLimitError):
        _ = complete_sync(core, "own")
    assert host.deployments_called("completion") == []


async def test_an_async_pick_runs_the_team_filter(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("own", "o1", model_info=TEAM_OWNED)], num_retries=0)
    with pytest.raises(RouterRateLimitError):
        await complete(core, "own")


@pytest.mark.parametrize(
    "request_fields",
    [
        pytest.param({}, id="no-team"),
        pytest.param({"metadata": {"user_api_key_team_id": "T2"}}, id="another-team"),
    ],
)
async def test_the_rust_core_refuses_a_team_owned_deployment_named_by_its_id(
    host: ScriptedHost, request_fields: Mapping[str, object]
) -> None:
    core: Final = CORE_BUILDERS["rust"](host, model_list=[deployment("own", "o1", model_info=TEAM_OWNED)], num_retries=0)
    with pytest.raises(RouterRateLimitError):
        _ = await complete(core, "o1", **request_fields)
    assert host.deployments_called("acompletion") == []


# --- what the errors name --------------------------------------------------------------------------


async def test_a_wildcard_refused_for_a_team_names_the_requested_model(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("anthropic/*", "w1", "anthropic/*", model_info=TEAM_OWNED)], num_retries=0
    )
    with pytest.raises(RouterRateLimitError) as refused:
        await complete(core, "anthropic/claude-x", metadata={"user_api_key_team_id": "T2"})
    assert refused.value.model == "anthropic/claude-x"


def test_a_pass_through_refusal_names_the_group_an_alias_resolved_to(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("target", "t1")], num_retries=0, model_group_alias={"alias": "target"}
    )
    with pytest.raises(BadRequestError) as refused:
        core.get_available_deployment_for_pass_through("alias", [])  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # the legacy member
    assert refused.value.model == "target"


def test_a_sync_completion_is_refused_by_a_deployment_that_serves_no_chat(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            deployment(
                "g", "e1", model_info={"enforce_supported_endpoints": True, "supported_endpoints": ["/v1/embeddings"]}
            )
        ],
        num_retries=0,
    )
    with pytest.raises(EndpointNotSupportedByModelError):
        core.completion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint


# --- construction ----------------------------------------------------------------------------------


def test_a_mocked_model_response_warns_once_per_dump_python_makes(make_core: MakeCore) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _ = make_core(model_list=[deployment("g", "d1", mock_response=BLOCKED_RESPONSE)], num_retries=0)
    serializer: Final = [w for w in caught if "Pydantic serializer warnings" in str(w.message)]
    assert len(serializer) == PYTHON_DUMPS_OF_A_DEPLOYMENT

