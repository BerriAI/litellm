import asyncio
import contextvars
import gc
import gzip
import json
import logging
import sys
import time
import traceback
import weakref
from collections.abc import AsyncIterator, Callable
from copy import deepcopy
from typing import Final

import h2.config
import h2.connection
import h2.events
import httpcore
import httpx
import pytest
import respx
from pydantic import SecretStr, ValidationError

import litellm
from litellm._logging import verbose_router_logger
from litellm.caching.evicted_client_closer import EvictedClientCloser
from litellm.litellm_core_utils.prompt_templates.factory import resolve_structured_messages
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.nace.decisions.transformation import nace_client_header
from litellm.router_api import RoutingContext as ApiRoutingContext
from litellm.router_strategy.complexity_router import complexity_router as router_module
from litellm.router_strategy.complexity_router import drex_route_classifier as drex_module
from litellm.router_strategy.complexity_router.complexity_router import ComplexityRouter
from litellm.router_strategy.complexity_router.drex_route_classifier import (
    DrexRouteCandidateProfile,
    DrexRouteClassifier,
    DrexRouteClassifierConfig,
    DrexRouteConfigError,
    DrexRouteError,
)
from litellm.types.router import RoutingContext
from litellm.types.utils import LlmProviders

_PRE_312: Final = pytest.mark.skipif(
    sys.version_info >= (3, 12),
    reason="3.12+ routes through asyncio.wait_for, which runs the plugin inline in the caller's task",
)


_PROFILE_DICT: Final = {
    "id": "custom-vendor/cheap",
    "provider": "custom-vendor",
    "price_in": 0.1,
    "price_out": 0.2,
    "context": 131072,
    "tags": ("cheap",),
}

_WIRE_PROFILE: Final = {
    "id": "custom-vendor/cheap",
    "provider": "custom-vendor",
    "price_in": 0.1,
    "price_out": 0.2,
    "context": 131072,
    "tags": ["cheap"],
}

_WIRE_CANDIDATES: Final = ["openai/gpt-6-luna", "anthropic/claude-sonnet-5.5", _WIRE_PROFILE]

_TOOLS: Final = [
    {
        "type": "function",
        "function": {"name": "lookup", "parameters": {"type": "object", "properties": {}}},
    }
]


def _config_dict(**overrides: object) -> dict[str, object]:
    return {
        "candidates": {
            "fast": "openai/gpt-6-luna",
            "careful": "anthropic/claude-sonnet-5.5",
            "custom": DrexRouteCandidateProfile.model_validate(_PROFILE_DICT),
        },
        "api_key": "test-key",
        "api_base": "https://drex.test",
        **overrides,
    }


def _handler(respond: Callable[[httpx.Request], httpx.Response]) -> AsyncHTTPHandler:
    """A handler built through AsyncHTTPHandler.create_client, so the underlying client
    carries production behavior (blocked cookie jar, redirects on by default) with a mock
    transport underneath."""
    return AsyncHTTPHandler(transport=httpx.MockTransport(respond))


def _selecting(candidate_id: str, requests: list[httpx.Request]) -> Callable[[httpx.Request], httpx.Response]:
    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"selected": candidate_id, "probabilities": {candidate_id: 0.9}, "request_id": "req_test"},
        )

    return respond


def _context(messages: list[dict[str, object]] | None = None) -> RoutingContext:
    raw: Final = messages if messages is not None else [{"role": "user", "content": "hello"}]
    return RoutingContext(
        raw_messages=raw,
        structured_messages=raw,
        candidate_models=["fast-model", "careful-model", "custom-model"],
    )


def _image_messages() -> list[dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this"},
                {"type": "image_url", "image_url": {"url": "https://example.test/light.png"}},
            ],
        }
    ]


def _auto_model_list() -> list[dict[str, object]]:
    return [
        {
            "model_name": "auto",
            "litellm_params": {
                "model": "auto_router/complexity_router",
                "complexity_router_config": {
                    "classifier_type": "custom",
                    "classifier_plugin": DrexRouteClassifier(_config_dict()),
                    "classifier_fallback": "default_model",
                    "tiers": {"fast": "fast-model", "careful": "careful-model", "custom": "custom-model"},
                },
                "complexity_router_default_model": "fast-model",
            },
        },
        {
            "model_name": "fast-model",
            "litellm_params": {"model": "openai/gpt-6-luna", "mock_response": "answer from fast"},
        },
        {
            "model_name": "careful-model",
            "litellm_params": {"model": "anthropic/claude-sonnet-5.5", "mock_response": "answer from careful"},
        },
        {
            "model_name": "custom-model",
            "litellm_params": {"model": "openai/gpt-6-luna", "mock_response": "answer from custom"},
        },
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tier", "candidate_id"),
    [
        ("fast", "openai/gpt-6-luna"),
        ("careful", "anthropic/claude-sonnet-5.5"),
        ("custom", "custom-vendor/cheap"),
    ],
)
async def test_selected_candidate_maps_to_its_tier(tier: str, candidate_id: str) -> None:
    requests: list[httpx.Request] = []
    handler: Final = _handler(_selecting(candidate_id, requests))
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)

    assert await classifier.classify(_context()) == tier

    assert len(requests) == 1
    await handler.close()


@pytest.mark.asyncio
async def test_route_request_carries_text_turns_candidates_priority_and_attribution() -> None:
    requests: list[httpx.Request] = []
    handler: Final = _handler(_selecting("anthropic/claude-sonnet-5.5", requests))
    classifier: Final = DrexRouteClassifier({**_config_dict(), "selection_priority": "cheapest"}, http_client=handler)

    assert await classifier.classify(_context()) == "careful"

    request: Final = requests[0]
    assert str(request.url) == "https://drex.test/v1/route"
    body: Final = json.loads(request.content)
    assert body == {
        "messages": [{"role": "user", "content": "hello"}],
        "candidates": _WIRE_CANDIDATES,
        "selection_priority": "cheapest",
    }
    assert "cost_weight" not in body
    assert request.headers["Authorization"] == "Bearer test-key"
    assert request.headers["X-Nace-Client"] == nace_client_header()["X-Nace-Client"]
    assert request.headers["Accept-Encoding"] == "identity"
    await handler.close()


@pytest.mark.asyncio
async def test_api_base_with_v1_suffix_reaches_the_same_route() -> None:
    requests: list[httpx.Request] = []
    handler: Final = _handler(_selecting("openai/gpt-6-luna", requests))
    classifier: Final = DrexRouteClassifier(
        {**_config_dict(), "api_base": "https://drex.test/v1/"}, http_client=handler
    )

    assert await classifier.classify(_context()) == "fast"

    assert str(requests[0].url) == "https://drex.test/v1/route"
    await handler.close()


@pytest.mark.asyncio
async def test_only_text_from_supported_roles_is_sent() -> None:
    requests: list[httpx.Request] = []
    handler: Final = _handler(_selecting("openai/gpt-6-luna", requests))
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)
    context: Final = _context(
        [
            {"role": "system", "content": "be terse"},
            {"role": "developer", "content": "use metric units"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is the speed of light"},
                    {"type": "image_url", "image_url": {"url": "https://example.test/light.png"}},
                ],
            },
            {"role": "tool", "content": "299792458"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
            },
            {"role": "user", "content": "thanks, and in km/s?"},
        ]
    )

    assert await classifier.classify(context) == "fast"

    assert json.loads(requests[0].content)["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "system", "content": "use metric units"},
        {"role": "user", "content": "what is the speed of light"},
        {"role": "user", "content": "thanks, and in km/s?"},
    ]
    await handler.close()


@pytest.mark.asyncio
async def test_image_request_offers_only_vision_tiers() -> None:
    requests: list[httpx.Request] = []
    handler: Final = _handler(_selecting("anthropic/claude-sonnet-5.5", requests))
    classifier: Final = DrexRouteClassifier(
        {**_config_dict(), "vision_tiers": frozenset({"careful"})}, http_client=handler
    )

    assert await classifier.classify(_context(_image_messages())) == "careful"
    assert json.loads(requests[0].content)["candidates"] == ["anthropic/claude-sonnet-5.5"]

    assert await classifier.classify(_context()) == "careful"
    assert json.loads(requests[1].content)["candidates"] == _WIRE_CANDIDATES
    await handler.close()


@pytest.mark.asyncio
async def test_image_request_without_vision_tiers_available_declines_without_calling_drex() -> None:
    requests: list[httpx.Request] = []
    handler: Final = _handler(_selecting("anthropic/claude-sonnet-5.5", requests))
    classifier: Final = DrexRouteClassifier({**_config_dict(), "vision_tiers": frozenset()}, http_client=handler)

    assert await classifier.classify(_context(_image_messages())) is None

    assert requests == []
    await handler.close()


@pytest.mark.asyncio
async def test_conversation_over_budget_sends_newest_turns_only() -> None:
    requests: list[httpx.Request] = []
    handler: Final = _handler(_selecting("openai/gpt-6-luna", requests))
    conversation: Final = [
        {"role": "user", "content": "draft a long legal disclaimer, please"},
        {"role": "assistant", "content": "here you go, a very long draft"},
        {"role": "user", "content": "make it shorter: café"},
    ]
    newest_only: Final = DrexRouteClassifier({**_config_dict(), "max_input_bytes": 30}, http_client=handler)

    assert await newest_only.classify(_context(conversation)) == "fast"
    newest: Final = json.loads(requests[0].content)["messages"]
    assert newest == [{"role": "user", "content": "make it shorter: café"}]
    assert sum(len(message["content"].encode()) for message in newest) <= 30

    truncated: Final = DrexRouteClassifier({**_config_dict(), "max_input_bytes": 11}, http_client=handler)
    assert await truncated.classify(_context(conversation)) == "fast"
    tail: Final = json.loads(requests[1].content)["messages"]
    assert tail == [{"role": "user", "content": "rter: café"}]
    assert sum(len(message["content"].encode()) for message in tail) <= 11

    blank_tail: Final = DrexRouteClassifier({**_config_dict(), "max_input_bytes": 1}, http_client=handler)
    assert await blank_tail.classify(_context(conversation)) is None
    assert len(requests) == 2
    await handler.close()


@pytest.mark.asyncio
async def test_request_without_text_declines_without_calling_drex() -> None:
    requests: list[httpx.Request] = []
    handler: Final = _handler(_selecting("openai/gpt-6-luna", requests))
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)
    context: Final = _context(
        [
            {"role": "tool", "content": "299792458"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
            },
        ]
    )

    assert await classifier.classify(context) is None

    assert requests == []
    await handler.close()


@pytest.mark.asyncio
async def test_missing_api_key_declines_without_calling_drex(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NACE_API_KEY", raising=False)
    requests: list[httpx.Request] = []
    handler: Final = _handler(_selecting("openai/gpt-6-luna", requests))
    config: Final = {key: value for key, value in _config_dict().items() if key != "api_key"}
    classifier: Final = DrexRouteClassifier(config, http_client=handler)

    assert await classifier.classify(_context()) is None

    assert requests == []
    await handler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", ["unknown-vendor/mystery", "careful"])
async def test_selection_outside_the_offered_candidates_is_never_returned(selected: str) -> None:
    requests: list[httpx.Request] = []
    handler: Final = _handler(_selecting(selected, requests))
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)

    assert await classifier.classify(_context()) is None

    assert len(requests) == 1
    await handler.close()


@pytest.mark.asyncio
async def test_selection_filtered_out_by_vision_tiers_is_never_returned() -> None:
    requests: list[httpx.Request] = []
    handler: Final = _handler(_selecting("openai/gpt-6-luna", requests))
    classifier: Final = DrexRouteClassifier(
        {**_config_dict(), "vision_tiers": frozenset({"careful"})}, http_client=handler
    )

    assert await classifier.classify(_context(_image_messages())) is None

    assert len(requests) == 1
    await handler.close()


@pytest.mark.asyncio
async def test_concurrent_classifications_do_not_mix_results() -> None:
    candidate_by_marker: Final = {
        "route fast": "openai/gpt-6-luna",
        "route carefully": "anthropic/claude-sonnet-5.5",
        "route custom": "custom-vendor/cheap",
    }
    tier_by_marker: Final = {"route fast": "fast", "route carefully": "careful", "route custom": "custom"}

    def respond(request: httpx.Request) -> httpx.Response:
        body: Final = json.loads(request.content)
        marker: Final = body["messages"][-1]["content"]
        candidate_id: Final = candidate_by_marker[marker]
        return httpx.Response(
            200, json={"selected": candidate_id, "probabilities": {candidate_id: 1.0}, "request_id": "req_test"}
        )

    handler: Final = _handler(respond)
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)
    markers: Final = tuple(("route fast", "route carefully", "route custom")[index % 3] for index in range(20))

    results: Final = await asyncio.gather(
        *(classifier.classify(_context([{"role": "user", "content": marker}])) for marker in markers)
    )

    assert tuple(results) == tuple(tier_by_marker[marker] for marker in markers)
    await handler.close()


@pytest.mark.parametrize(
    "candidates",
    [
        {"fast": "m/one", "careful": "m/one"},
        {},
        {f"tier-{index}": f"m/{index}" for index in range(256)},
    ],
)
def test_config_rejects_unusable_candidate_maps(candidates: dict[str, str]) -> None:
    with pytest.raises(DrexRouteConfigError):
        DrexRouteClassifierConfig.model_validate({"candidates": candidates})


def test_config_rejects_unknown_vision_tier() -> None:
    with pytest.raises(DrexRouteConfigError, match="vision_tiers names tiers with no candidate: spectral"):
        DrexRouteClassifierConfig.model_validate({**_config_dict(), "vision_tiers": frozenset({"spectral"})})


def test_config_rejects_negative_profile_price() -> None:
    with pytest.raises(ValidationError):
        DrexRouteCandidateProfile.model_validate({**_PROFILE_DICT, "price_in": -0.1})


@pytest.mark.asyncio
async def test_redirects_are_never_followed_even_when_the_client_enables_them() -> None:
    """The injected handler's client is created with follow_redirects=True, the shared
    default; the per-request setting must win so no cross-host resend can happen."""
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(307, headers={"Location": "https://evil.test/v1/route"})
        return httpx.Response(200, json={"selected": "openai/gpt-6-luna", "probabilities": {}})

    handler: Final = _handler(respond)
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)

    with pytest.raises(DrexRouteError, match="status 307"):
        await classifier.classify(_context())

    assert [str(request.url) for request in requests] == ["https://drex.test/v1/route"]
    await handler.close()


@pytest.mark.asyncio
async def test_responses_api_tool_text_is_stripped_from_the_route_request() -> None:
    """A continued /v1/responses request, converted by litellm's own resolver: the
    web_search_call becomes assistant content and the tool output becomes a tool message.
    Only genuine user text may reach Drex."""
    requests: list[httpx.Request] = []
    handler: Final = _handler(_selecting("openai/gpt-6-luna", requests))
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)
    structured: Final = resolve_structured_messages(
        None,
        {
            "input": [
                {"role": "user", "content": "find the best fondue in Geneva"},
                {
                    "type": "web_search_call",
                    "id": "ws_1",
                    "status": "completed",
                    "action": {"type": "search", "query": "best fondue in Geneva"},
                },
                {"type": "function_call", "id": "fc_1", "name": "lookup", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "fc_1", "output": "tool result text"},
                {"role": "user", "content": "now summarize what you found"},
            ]
        },
    )
    assert structured is not None
    assert any(
        message.get("role") == "assistant" and str(message.get("content", "")).startswith("Hosted web search: ")
        for message in structured
    )
    context: Final = RoutingContext(
        raw_messages=[],
        structured_messages=structured,
        candidate_models=["fast-model", "careful-model", "custom-model"],
    )

    assert await classifier.classify(context) == "fast"

    assert json.loads(requests[0].content)["messages"] == [
        {"role": "user", "content": "find the best fondue in Geneva"},
        {"role": "user", "content": "now summarize what you found"},
    ]
    await handler.close()


@pytest.mark.asyncio
async def test_user_text_starting_with_the_hosted_search_prefix_is_kept() -> None:
    """Only converted assistant tool content is filtered; a user instruction that happens
    to start with the prefix is a real request and must reach Drex."""
    requests: list[httpx.Request] = []
    handler: Final = _handler(_selecting("openai/gpt-6-luna", requests))
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)

    assert (
        await classifier.classify(_context([{"role": "user", "content": "Hosted web search: compare two providers"}]))
        == "fast"
    )

    assert json.loads(requests[0].content)["messages"] == [
        {"role": "user", "content": "Hosted web search: compare two providers"}
    ]
    await handler.close()


@pytest.mark.asyncio
async def test_remote_strings_never_reach_logs(caplog: pytest.LogCaptureFixture) -> None:
    """Drex-controlled fields must not be logged verbatim: an unknown selected ID and a
    request_id are only ever reflected as a length or not at all, in decline and success."""
    marker: Final = "PROMPT-ECHO-MARKER"
    requests: list[httpx.Request] = []

    def echoing(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200, json={"selected": f"unknown-vendor/{marker}", "probabilities": {}, "request_id": marker}
        )

    def echoing_success(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"selected": "openai/gpt-6-luna", "probabilities": {"openai/gpt-6-luna": 0.9}, "request_id": marker},
        )

    declining_handler: Final = _handler(echoing)
    declining: Final = DrexRouteClassifier(_config_dict(), http_client=declining_handler)
    with caplog.at_level(logging.DEBUG, logger=verbose_router_logger.name):
        assert await declining.classify(_context([{"role": "user", "content": marker}])) is None

    succeeding_handler: Final = _handler(echoing_success)
    succeeding: Final = DrexRouteClassifier(_config_dict(), http_client=succeeding_handler)
    with caplog.at_level(logging.DEBUG, logger=verbose_router_logger.name):
        assert await succeeding.classify(_context()) == "fast"

    assert marker not in caplog.text
    await declining_handler.close()
    await succeeding_handler.close()


def test_config_validation_errors_never_include_the_api_key() -> None:
    with pytest.raises(DrexRouteConfigError) as excinfo:
        DrexRouteClassifier({"candidates": {}, "api_key": "test-key", "api_base": "https://drex.test"})
    assert "test-key" not in str(excinfo.value)

    with pytest.raises(DrexRouteConfigError, match="api_key must be a string"):
        DrexRouteClassifier({**_config_dict(), "api_key": 123})


def test_direct_config_construction_never_echoes_the_api_key() -> None:
    with pytest.raises(DrexRouteConfigError) as excinfo:
        DrexRouteClassifierConfig(api_key="test-key", candidates={})
    assert "test-key" not in str(excinfo.value)

    with pytest.raises(ValidationError) as missing:
        DrexRouteClassifierConfig(api_key="test-key")
    assert "test-key" not in str(missing.value)
    assert all("test-key" not in str(error.get("input", "")) for error in missing.value.errors())


@pytest.mark.asyncio
async def test_http_failures_raise_sanitized_errors_without_request_details() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "nope"})

    handler: Final = _handler(respond)
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)

    with pytest.raises(DrexRouteError) as excinfo:
        await classifier.classify(_context())

    assert str(excinfo.value) == "drex route call returned status 500"
    assert "test-key" not in repr(excinfo.value)
    assert "nope" not in repr(excinfo.value)
    assert not hasattr(excinfo.value, "request")
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    await handler.close()


@pytest.mark.asyncio
async def test_oversized_response_is_rejected_without_selecting_a_tier() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "selected": "openai/gpt-6-luna",
                "probabilities": {"openai/gpt-6-luna": 0.9},
                "request_id": "req_test",
                "ignored": "x" * 70_000,
            },
        )

    handler: Final = _handler(respond)
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)

    with pytest.raises(DrexRouteError, match="exceeded"):
        await classifier.classify(_context())

    await handler.close()


@pytest.mark.asyncio
async def test_compressed_responses_are_rejected_before_reading() -> None:
    """The request asks for identity encoding; a server that ignores it and answers with a
    compressed body is refused before any byte is decoded, so the cap bounds wire bytes."""
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            content=gzip.compress(
                json.dumps({"selected": "openai/gpt-6-luna", "probabilities": {}, "request_id": "req"}).encode()
            ),
            headers={"Content-Encoding": "gzip"},
        )

    handler: Final = _handler(respond)
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)

    with pytest.raises(DrexRouteError, match="content encoding"):
        await classifier.classify(_context())

    assert len(requests) == 1
    await handler.close()


@pytest.mark.asyncio
async def test_image_in_normalized_messages_narrows_vision_tiers_when_raw_is_empty() -> None:
    """An input-based request can arrive with empty raw_messages and the image only in
    the normalized list; the vision filter must still narrow the offered tiers."""
    requests: list[httpx.Request] = []
    handler: Final = _handler(_selecting("anthropic/claude-sonnet-5.5", requests))
    classifier: Final = DrexRouteClassifier(
        {**_config_dict(), "vision_tiers": frozenset({"careful"})}, http_client=handler
    )
    context: Final = RoutingContext(
        raw_messages=[],
        structured_messages=_image_messages(),
        candidate_models=["fast-model", "careful-model", "custom-model"],
    )

    assert await classifier.classify(context) == "careful"

    assert json.loads(requests[0].content)["candidates"] == ["anthropic/claude-sonnet-5.5"]
    await handler.close()


@pytest.mark.asyncio
async def test_connection_error_makes_exactly_one_attempt() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise httpx.ConnectError("connection refused")

    handler: Final = _handler(respond)
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)

    with pytest.raises(DrexRouteError, match="ConnectError") as excinfo:
        await classifier.classify(_context())

    assert len(requests) == 1
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    assert not hasattr(excinfo.value, "request")
    await handler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [201, 202, 204, 307])
async def test_only_http_200_is_accepted(status_code: int) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            json={"selected": "openai/gpt-6-luna", "probabilities": {"openai/gpt-6-luna": 0.9}, "request_id": "req"},
        )

    handler: Final = _handler(respond)
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)

    with pytest.raises(DrexRouteError, match=f"status {status_code}"):
        await classifier.classify(_context())

    await handler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("probabilities", [{"openai/gpt-6-luna": True}, {"openai/gpt-6-luna": "0.75"}])
async def test_non_float_probabilities_are_rejected(probabilities: dict[str, object]) -> None:
    body: Final = {"selected": "openai/gpt-6-luna", "probabilities": probabilities, "request_id": "req"}

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    handler: Final = _handler(respond)
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)

    with pytest.raises(ValidationError, match="expected shape") as excinfo:
        await classifier.classify(_context())

    assert json.dumps(body) not in str(excinfo.value)
    await handler.close()


@pytest.mark.asyncio
async def test_probabilities_outside_the_contract_are_rejected() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"selected": "openai/gpt-6-luna", "probabilities": {"openai/gpt-6-luna": 2.0}, "request_id": "req"},
        )

    handler: Final = _handler(respond)
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)

    with pytest.raises(ValidationError, match="expected shape"):
        await classifier.classify(_context())

    await handler.close()


@pytest.mark.asyncio
async def test_large_history_projects_only_the_newest_turns_within_the_budget() -> None:
    requests: list[httpx.Request] = []
    handler: Final = _handler(_selecting("openai/gpt-6-luna", requests))
    classifier: Final = DrexRouteClassifier({**_config_dict(), "max_input_bytes": 120}, http_client=handler)
    history: Final = [{"role": "user", "content": f"turn {index}"} for index in range(2000)]

    assert await classifier.classify(_context(history)) == "fast"

    sent: Final = json.loads(requests[0].content)["messages"]
    assert sent == [{"role": "user", "content": f"turn {index}"} for index in range(1987, 2000)]
    assert sum(len(message["content"].encode()) for message in sent) <= 120
    await handler.close()


@pytest.mark.asyncio
async def test_tool_heavy_histories_still_route_the_newest_eligible_turns() -> None:
    """A suffix of tool results costs one role lookup per message, never a fallback: the
    turn beyond it stays the newest eligible text and must reach Drex."""
    requests: list[httpx.Request] = []
    handler: Final = _handler(_selecting("openai/gpt-6-luna", requests))
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)

    assert (
        await classifier.classify(
            _context(
                [{"role": "user", "content": "summarize the fondue search"}]
                + [{"role": "tool", "content": f"result {index}"} for index in range(1024)]
            )
        )
        == "fast"
    )
    assert json.loads(requests[0].content)["messages"] == [{"role": "user", "content": "summarize the fondue search"}]

    assert (
        await classifier.classify(
            _context(
                [
                    {"role": "user", "content": "old ask"},
                    *[{"role": "tool", "content": f"result {index}"} for index in range(1100)],
                    {"role": "user", "content": "new ask"},
                ]
            )
        )
        == "fast"
    )
    assert json.loads(requests[1].content)["messages"] == [
        {"role": "user", "content": "old ask"},
        {"role": "user", "content": "new ask"},
    ]
    await handler.close()


@pytest.mark.asyncio
async def test_credentials_resolve_once_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NACE_API_KEY", "construction-key")
    requests: list[httpx.Request] = []
    handler: Final = _handler(_selecting("openai/gpt-6-luna", requests))
    config: Final = {name: value for name, value in _config_dict().items() if name != "api_key"}
    classifier: Final = DrexRouteClassifier(config, http_client=handler)
    monkeypatch.setenv("NACE_API_KEY", "later-key")

    assert await classifier.classify(_context()) == "fast"

    assert requests[0].headers["Authorization"] == "Bearer construction-key"
    await handler.close()


@pytest.mark.asyncio
async def test_transport_cancelled_without_a_pending_cancel_converts_to_a_sanitized_error() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError()

    handler: Final = _handler(respond)
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)

    with pytest.raises(DrexRouteError, match="cancelled before completing"):
        await classifier.classify(_context())

    await handler.close()


@pytest.mark.asyncio
async def test_cancelling_the_caller_still_propagates_cancellation() -> None:
    never: Final = asyncio.Event()

    async def respond(request: httpx.Request) -> httpx.Response:
        await never.wait()
        return httpx.Response(200, json={"selected": "openai/gpt-6-luna", "probabilities": {}})

    handler: Final = _handler(respond)
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)
    task = asyncio.create_task(classifier.classify(_context()))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    await handler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule", ["in_responder", "call_soon"])
async def test_caller_cancellation_outranks_a_simultaneous_exchange_failure(schedule: str) -> None:
    """A 503 landing on the same loop turn as the caller's cancellation must surface the
    cancellation, not the sanitized failure, and must not leave the failure unretrieved."""
    requests: list[httpx.Request] = []
    tasks: list[asyncio.Task[str | None]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if schedule == "in_responder":
            tasks[0].cancel()
        else:
            asyncio.get_running_loop().call_soon(tasks[0].cancel)
        return httpx.Response(503, json={"error": "nope"})

    handler: Final = _handler(respond)
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)
    tasks.append(asyncio.ensure_future(classifier.classify(_context())))

    with pytest.raises(asyncio.CancelledError):
        await tasks[0]

    assert len(requests) == 1
    await handler.close()


class _SlowClosingStream(httpx.AsyncByteStream):
    """A response body whose transport close waits for the test, recording the closing task."""

    def __init__(self) -> None:
        self.closing: Final = asyncio.Event()
        self.allow_close: Final = asyncio.Event()
        self.released: Final = asyncio.Event()
        self.closer: asyncio.Task[object] | None = None

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield json.dumps({"selected": "openai/gpt-6-luna", "probabilities": {}}).encode()

    async def aclose(self) -> None:
        self.closer = asyncio.current_task()
        self.closing.set()
        await self.allow_close.wait()
        self.released.set()


def _slow_closing_handler(stream: _SlowClosingStream, close: str) -> AsyncHTTPHandler:
    headers: Final = {"Content-Encoding": "gzip"} if close == "after_rejection" else None
    return _handler(lambda request: httpx.Response(200, stream=stream, headers=headers))


async def _let_cancellations_land() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("close", ["at_eof", "after_rejection"])
@pytest.mark.parametrize("cancels", [1, 2])
async def test_cancelling_during_the_response_close_still_releases_the_stream(close: str, cancels: int) -> None:
    """HTTPX marks a response closed before it awaits the transport's close, so an
    interrupted close is never retried. A caller cancellation that lands mid-close, and a
    second one while the caller drains the exchange, waits for the close and then
    propagates."""
    stream: Final = _SlowClosingStream()
    handler: Final = _slow_closing_handler(stream, close)
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)
    task: Final = asyncio.create_task(classifier.classify(_context()))
    try:
        await asyncio.wait_for(stream.closing.wait(), 2)
        for _ in range(cancels):
            task.cancel()
            await _let_cancellations_land()
        assert not task.done(), "classify returned while the response stream was still closing"
        stream.allow_close.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.released.is_set()
        assert stream.closer is not None and stream.closer.done()
    finally:
        stream.allow_close.set()
        await asyncio.gather(task, return_exceptions=True)
        await handler.close()


@pytest.mark.asyncio
async def test_plugin_deadline_during_the_response_close_still_releases_the_stream() -> None:
    stream: Final = _SlowClosingStream()
    handler: Final = _slow_closing_handler(stream, "at_eof")
    router: Final = _plugin_router(handler, classifier_plugin_timeout_ms=50)
    routing: Final = asyncio.create_task(
        router.async_pre_routing_hook(model="auto", request_kwargs={}, messages=[{"role": "user", "content": "hello"}])
    )
    try:
        await asyncio.wait_for(stream.closing.wait(), 2)
        await asyncio.sleep(0.2)
        assert not routing.done(), "routing finished while the response stream was still closing"
        stream.allow_close.set()
        response: Final = await routing
        assert response is not None
        assert response.model == "backup"
        assert response.routing_decision is not None
        assert response.routing_decision["classifier_failure_reason"] == "timeout"
        assert stream.released.is_set()
    finally:
        stream.allow_close.set()
        await asyncio.gather(routing, return_exceptions=True)
        await handler.close()


def _self_cancelling(request: httpx.Request) -> httpx.Response:
    """A transport that cancels its own task and returns without suspending, so the task
    ends cancelled although its coroutine never saw a CancelledError."""
    task: Final = asyncio.current_task()
    assert task is not None
    _ = task.cancel()
    return httpx.Response(200, json={"selected": "openai/gpt-6-luna", "probabilities": {}})


@pytest.mark.asyncio
async def test_exchange_cancelling_itself_without_suspending_raises_a_sanitized_error() -> None:
    handler: Final = _handler(_self_cancelling)
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)

    with pytest.raises(DrexRouteError, match="cancelled before completing") as excinfo:
        await classifier.classify(_context())

    assert "test-key" not in repr(excinfo.value)
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    await handler.close()


@pytest.mark.asyncio
async def test_exchange_cancelling_itself_without_suspending_falls_back_to_the_default_model() -> None:
    handler: Final = _handler(_self_cancelling)
    router: Final = _plugin_router(handler)

    response: Final = await router.async_pre_routing_hook(
        model="auto", request_kwargs={}, messages=[{"role": "user", "content": "hello"}]
    )

    assert response is not None
    assert response.model == "backup"
    assert response.routing_decision is not None
    assert response.routing_decision["classifier_failure_reason"] == "classifier_error"
    await handler.close()


_SECRET_KEY: Final = "sk-drex-secret-key-material"
_REMOTE_BODY: Final = "remote-secret-body-text"
_INPUT_TEXT: Final = "private-user-input-text"


def _frame_locals(error: BaseException) -> list[object]:
    """The locals of every traceback frame of the error and its chain, except this test
    module's own frames."""
    found: list[object] = []
    for link in (error, error.__context__, error.__cause__):
        tb = link.__traceback__ if link is not None else None
        while tb is not None:
            if tb.tb_frame.f_code.co_filename != __file__:
                found.extend(tb.tb_frame.f_locals.values())
            tb = tb.tb_next
    return found


def _rendered_traceback(error: BaseException) -> str:
    """The traceback as standard formatting with capture_locals renders it, as a logger or
    error reporter would, without this test module's own frames."""
    rendered: Final = traceback.TracebackException.from_exception(error, capture_locals=True)
    rendered.stack = traceback.StackSummary.from_list([frame for frame in rendered.stack if frame.filename != __file__])
    return "".join(rendered.format())


def _reachable(error: BaseException) -> list[object]:
    """Every object reachable from the error through its exception chain, args, and
    attributes, recursively."""
    seen: list[object] = []
    visited: set[int] = set()
    pending: list[object] = [error]
    while pending:
        item = pending.pop()
        if id(item) in visited:
            continue
        visited.add(id(item))
        seen.append(item)
        if isinstance(item, BaseException):
            pending.extend(link for link in (item.__context__, item.__cause__) if link is not None)
            pending.extend(item.args)
            pending.extend(getattr(item, "__dict__", {}).values())
        elif isinstance(item, (tuple, list, set, frozenset)):
            pending.extend(item)
        elif isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
    return seen


class _FailingReadStream(httpx.AsyncByteStream):
    """A body whose read fails with a ReadError carrying the request, and whose close is
    then interrupted by a CancelledError, chaining that ReadError onto it."""

    def __init__(self, request: httpx.Request) -> None:
        self.request: Final = request

    async def __aiter__(self) -> AsyncIterator[bytes]:
        raise httpx.ReadError(_REMOTE_BODY, request=self.request)
        yield b""

    async def aclose(self) -> None:
        raise asyncio.CancelledError()


def _leaky_responder(scenario: str) -> Callable[[httpx.Request], httpx.Response]:
    def respond(request: httpx.Request) -> httpx.Response:
        if scenario == "read_error_then_cancelled_close":
            return httpx.Response(200, stream=_FailingReadStream(request))
        if scenario == "connect_error":
            raise httpx.ConnectError(_REMOTE_BODY, request=request)
        if scenario == "read_timeout":
            raise httpx.ReadTimeout(_REMOTE_BODY, request=request)
        if scenario == "self_cancel":
            return _self_cancelling(request)
        if scenario == "status_500":
            return httpx.Response(500, json={"error": _REMOTE_BODY})
        return httpx.Response(200, content=f'{{"selected": "{_REMOTE_BODY}"'.encode())

    return respond


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario",
    [
        "read_error_then_cancelled_close",
        "connect_error",
        "read_timeout",
        "self_cancel",
        "status_500",
        "malformed_body",
        "lone_surrogate_input",
    ],
)
async def test_sanitized_errors_reach_no_key_body_or_input(scenario: str) -> None:
    """Following __context__, __cause__, args, attributes, and the locals of every
    traceback frame outside this test from the raised error never reaches the key, the
    remote body, the conversation text, or an httpx request or response."""
    handler: Final = _handler(_leaky_responder(scenario))
    classifier: Final = DrexRouteClassifier(_config_dict(api_key=_SECRET_KEY), http_client=handler)
    text: Final = f"{_INPUT_TEXT}\ud800" if scenario == "lone_surrogate_input" else _INPUT_TEXT

    with pytest.raises((DrexRouteError, ValidationError)) as excinfo:
        await classifier.classify(_context([{"role": "user", "content": text}]))

    for item in _reachable(excinfo.value):
        assert not isinstance(item, (httpx.Request, httpx.Response))
        rendered = repr(item)
        assert _SECRET_KEY not in rendered
        assert _REMOTE_BODY not in rendered
        assert _INPUT_TEXT not in rendered
    for value in _frame_locals(excinfo.value):
        assert not isinstance(value, (httpx.Request, httpx.Response))
    rendered_traceback: Final = _rendered_traceback(excinfo.value)
    assert _SECRET_KEY not in rendered_traceback
    assert _REMOTE_BODY not in rendered_traceback
    assert _INPUT_TEXT not in rendered_traceback
    await handler.close()


_NON_TOKEN_KEYS: Final = (
    "sk-drex-é-secret",
    "sk-drex-\r\nX-Injected: 1",
    "sk-drex-\u2028-secret",
    "sk-drex-secret-with-a-trailing-space ",
    " sk-drex-secret-with-a-leading-space",
    "\tsk-drex-secret-with-a-tab",
    "sk-drex-secret-with-a-newline\n",
    "sk-drex=secret-with-inner-padding",
    "sk-drex-secret,with-a-comma",
)


@pytest.mark.parametrize("key", _NON_TOKEN_KEYS)
def test_api_key_that_cannot_be_a_header_is_rejected_at_construction_without_echo(
    key: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Only RFC 6750 Bearer tokens are accepted. Anything else could be rejected by the
    HTTP stack, which logs the rejected header value at debug level, so it is refused before
    any request exists, and neither the error, its traceback frames, nor any log carries it."""
    caplog.set_level(logging.DEBUG)
    for build in (
        lambda: DrexRouteClassifier(_config_dict(api_key=key)),
        lambda: DrexRouteClassifier(_config_dict(api_key=SecretStr(key))),
        lambda: DrexRouteClassifierConfig(**_config_dict(api_key=key)),
        lambda: DrexRouteClassifier(DrexRouteClassifierConfig(**_config_dict(api_key=SecretStr(key)))),
    ):
        with pytest.raises(DrexRouteConfigError) as excinfo:
            _ = build()
        assert key not in repr(excinfo.value)
        assert "secret" not in str(excinfo.value)
        assert excinfo.value.__context__ is None
        assert "secret" not in _rendered_traceback(excinfo.value)
        assert all("secret" not in repr(value) for value in _frame_locals(excinfo.value))
    assert "secret" not in caplog.text


def test_api_key_from_the_environment_is_checked_at_construction(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    for key in _NON_TOKEN_KEYS:
        monkeypatch.setenv("NACE_API_KEY", key)

        with pytest.raises(DrexRouteConfigError) as excinfo:
            _ = DrexRouteClassifier(_config_dict(api_key=None))

        assert "secret" not in str(excinfo.value)
        assert "secret" not in _rendered_traceback(excinfo.value)
        assert all("secret" not in repr(value) for value in _frame_locals(excinfo.value))
    assert "secret" not in caplog.text


@pytest.mark.parametrize("key", ["sk-drex-secret_key.v1~x", "c2stZHJleC1zZWNyZXQ+/x==", "a"])
def test_api_key_in_bearer_token_syntax_is_sent_as_the_bearer_credential(key: str) -> None:
    requests: list[httpx.Request] = []
    classifier: Final = DrexRouteClassifier(
        _config_dict(api_key=key), http_client=_handler(_selecting("openai/gpt-6-luna", requests))
    )

    assert asyncio.run(classifier.classify(_context())) == "fast"
    assert requests[0].headers["authorization"] == f"Bearer {key}"


@pytest.mark.parametrize("field", ["timeout_s", "cleanup_grace_s"])
@pytest.mark.parametrize("value", [float("inf"), float("nan"), 1e9])
def test_unbounded_durations_are_rejected(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        _ = DrexRouteClassifierConfig(**_config_dict(**{field: value}))


class _SlowCloseNetworkStream(httpcore.AsyncMockStream):
    """A network connection whose close waits for the test."""

    def __init__(self, buffer: list[bytes], closing: asyncio.Event, allow: asyncio.Event, done: asyncio.Event) -> None:
        super().__init__(buffer)
        self.closing: Final = closing
        self.allow: Final = allow
        self.done: Final = done

    async def aclose(self) -> None:
        self.closing.set()
        await self.allow.wait()
        await super().aclose()
        self.done.set()


class _SlowCloseBackend(httpcore.AsyncMockBackend):
    def __init__(self, buffer: list[bytes]) -> None:
        super().__init__(buffer)
        self.closing: Final = asyncio.Event()
        self.allow_close: Final = asyncio.Event()
        self.released: Final = asyncio.Event()
        self.connects: int = 0

    async def connect_tcp(self, *args: object, **kwargs: object) -> httpcore.AsyncNetworkStream:
        del args, kwargs
        self.connects += 1
        return _SlowCloseNetworkStream(list(self._buffer), self.closing, self.allow_close, self.released)


def _httpcore_handler(backend: httpcore.AsyncNetworkBackend, **pool_options: object) -> AsyncHTTPHandler:
    """A handler on HTTPX's real HTTP/1.1 transport over a scripted network backend."""
    transport: Final = httpx.AsyncHTTPTransport()
    transport._pool = httpcore.AsyncConnectionPool(network_backend=backend, **pool_options)  # pyright: ignore[reportPrivateUsage, reportArgumentType]  # HTTPX exposes no public network-backend hook
    return AsyncHTTPHandler(transport=transport)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_bytes",
    [
        pytest.param(
            b"HTTP/1.1 307 Temporary Redirect\r\nLocation: http://drex.test:abc/\r\nContent-Length: 4\r\n\r\n",
            id="close_inside_send_on_invalid_redirect",
        ),
        pytest.param(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{", id="close_inside_read_on_truncated_body"),
    ],
)
async def test_cancelling_during_a_transport_initiated_close_still_releases_the_connection(
    response_bytes: bytes,
) -> None:
    """HTTPX (on an invalid redirect inside send()) and httpcore (on a truncated body) start
    their own closes below the adapter's stream wrapper. A caller cancellation that lands
    there must wait for the release instead of interrupting it."""
    backend: Final = _SlowCloseBackend([response_bytes])
    handler: Final = _httpcore_handler(backend)
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)
    task: Final = asyncio.create_task(classifier.classify(_context()))
    try:
        await asyncio.wait_for(backend.closing.wait(), 2)
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done(), "classify returned while the connection was still being released"
        backend.allow_close.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert backend.released.is_set()
    finally:
        backend.allow_close.set()
        await asyncio.gather(task, return_exceptions=True)
        await handler.close()


@pytest.mark.asyncio
async def test_stuck_close_is_detached_after_the_cleanup_grace() -> None:
    stream: Final = _SlowClosingStream()
    handler: Final = _slow_closing_handler(stream, "at_eof")
    classifier: Final = DrexRouteClassifier(_config_dict(cleanup_grace_s=0.05), http_client=handler)
    task: Final = asyncio.create_task(classifier.classify(_context()))
    try:
        await asyncio.wait_for(stream.closing.wait(), 2)
        task.cancel()
        await asyncio.sleep(0.01)
        assert not task.done()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        assert stream.closer is not None and not stream.closer.done()
        assert stream.closer in drex_module._DETACHED_EXCHANGES  # pyright: ignore[reportPrivateUsage]  # the strong reference is the guarantee under test
        stream.allow_close.set()
        await asyncio.wait_for(stream.released.wait(), 1)
        await asyncio.sleep(0)
        assert stream.closer.done()
        assert stream.closer not in drex_module._DETACHED_EXCHANGES  # pyright: ignore[reportPrivateUsage]
    finally:
        stream.allow_close.set()
        await asyncio.gather(task, return_exceptions=True)
        await handler.close()


@pytest.mark.asyncio
async def test_stuck_close_at_the_plugin_deadline_still_falls_back_after_the_grace() -> None:
    stream: Final = _SlowClosingStream()
    handler: Final = _slow_closing_handler(stream, "at_eof")
    router: Final = _plugin_router(
        handler,
        classifier_plugin=DrexRouteClassifier(_config_dict(cleanup_grace_s=0.05), http_client=handler),
        classifier_plugin_timeout_ms=30,
    )
    try:
        response: Final = await asyncio.wait_for(
            router.async_pre_routing_hook(
                model="auto", request_kwargs={}, messages=[{"role": "user", "content": "hello"}]
            ),
            2,
        )
        assert response is not None
        assert response.model == "backup"
        assert response.routing_decision is not None
        assert response.routing_decision["classifier_failure_reason"] == "timeout"
    finally:
        stream.allow_close.set()
        await asyncio.wait_for(stream.released.wait(), 1)
        await handler.close()


class _ScriptedPlugin:
    """A classifier plugin whose completion and cleanup the test controls."""

    def __init__(self, *, finish: bool) -> None:
        self.finish: Final = finish
        self.cleaning_up: Final = asyncio.Event()
        self.allow_cleanup: Final = asyncio.Event()
        self.on_done: Callable[[], None] | None = None
        self.task: asyncio.Task[object] | None = None

    async def classify(self, context: RoutingContext) -> str | None:
        del context
        self.task = asyncio.current_task()
        if self.finish:
            on_done: Final = self.on_done
            if on_done is not None:
                self.task.add_done_callback(lambda _: on_done())
            return "fast"
        try:
            await asyncio.Future()
        finally:
            self.cleaning_up.set()
            await self.allow_cleanup.wait()
        return None


def _scripted_router(plugin: _ScriptedPlugin, timeout_ms: int = 5000) -> ComplexityRouter:
    return ComplexityRouter(
        model_name="auto",
        ctx=ApiRoutingContext(litellm.Router(model_list=[])),
        default_model="backup",
        complexity_router_config={
            "classifier_type": "custom",
            "classifier_plugin": plugin,
            "classifier_plugin_timeout_ms": timeout_ms,
            "classifier_fallback": "default_model",
            "tiers": {"fast": "fast-model", "careful": "careful-model"},
        },
    )


def _start_routing(router: ComplexityRouter) -> "asyncio.Task[object]":
    return asyncio.create_task(
        router.async_pre_routing_hook(model="auto", request_kwargs={}, messages=[{"role": "user", "content": "hello"}])
    )


@_PRE_312
@pytest.mark.asyncio
async def test_plugin_deadline_never_returns_a_result_after_the_caller_cancelled() -> None:
    """CPython 3.10 and 3.11 asyncio.wait_for returned the plugin's finished result when the
    caller's cancellation landed as the plugin completed."""
    plugin: Final = _ScriptedPlugin(finish=True)
    routing: Final = _start_routing(_scripted_router(plugin))

    def cancel_routing() -> None:
        _ = routing.cancel()

    plugin.on_done = cancel_routing

    with pytest.raises(asyncio.CancelledError):
        await routing


@_PRE_312
@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["caller", "deadline"])
async def test_plugin_deadline_waits_for_plugin_cleanup_through_a_further_cancellation(trigger: str) -> None:
    """CPython 3.10 and 3.11 asyncio.wait_for returned while the plugin was still cleaning
    up when a further cancellation arrived (a second caller cancel, or the first one after
    the deadline started cleanup)."""
    plugin: Final = _ScriptedPlugin(finish=False)
    routing: Final = _start_routing(_scripted_router(plugin, timeout_ms=20 if trigger == "deadline" else 5000))
    try:
        if trigger == "caller":
            await asyncio.sleep(0.01)
            routing.cancel()
        await asyncio.wait_for(plugin.cleaning_up.wait(), 2)
        routing.cancel()
        await asyncio.sleep(0.05)
        assert not routing.done(), "routing returned while the plugin was still cleaning up"
        plugin.allow_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await routing
        assert plugin.task is not None and plugin.task.done()
    finally:
        plugin.allow_cleanup.set()
        await asyncio.gather(routing, return_exceptions=True)


@pytest.mark.asyncio
async def test_evicting_the_shared_handler_mid_request_still_closes_the_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eviction drops the cache's reference mid-request, and the collector may run in the
    same turn. Because the classifier holds the handler itself (not only its client) until
    the request ends, the evicted handler survives that collection and the closer closes
    its transport once the request is done, instead of the client being orphaned open."""

    class _EvictableTransport(httpx.MockTransport):
        closed = False
        in_flight = False

        def has_request_in_flight(self) -> bool:
            return self.in_flight

        async def aclose(self) -> None:
            self.closed = True
            await super().aclose()

    cache: Final = litellm.in_memory_llm_clients_cache
    closer: Final = EvictedClientCloser(grace_seconds=0, clock=lambda: 0.0)
    monkeypatch.setattr(cache, "evicted_client_closer", closer)
    handler_refs: list[weakref.ref[AsyncHTTPHandler]] = []
    held: list[AsyncHTTPHandler] = []
    alive_mid_request: list[bool] = []
    evicted_keys: list[object] = []

    def respond(request: httpx.Request) -> httpx.Response:
        transport.in_flight = True
        for key, value in tuple(cache.cache_dict.items()):
            if value is handler_refs[0]():
                evicted_keys.append(key)
                cache._remove_key(key)
                break
        del key, value
        held.clear()
        gc.collect()
        alive_mid_request.append(handler_refs[0]() is not None)
        resurrected: Final = handler_refs[0]()
        if resurrected is not None:
            held.append(resurrected)
        return httpx.Response(200, json={"selected": "openai/gpt-6-luna", "probabilities": {}})

    transport: Final = _EvictableTransport(respond)

    def fresh_handler(llm_provider: object, **_: object) -> AsyncHTTPHandler:
        handler = AsyncHTTPHandler(transport=transport)
        handler_refs.append(weakref.ref(handler))
        held.append(handler)
        cache.set_cache("async_httpx_client-test-drex-eviction", handler, litellm_owned_client=True)
        return handler

    monkeypatch.setattr(drex_module, "get_async_httpx_client", fresh_handler)
    classifier: Final = DrexRouteClassifier(_config_dict())

    assert await classifier.classify(_context()) == "fast"

    assert len(evicted_keys) == 1
    assert alive_mid_request == [True]
    assert len(held) == 1
    transport.in_flight = False
    closer.reap()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert transport.closed
    held.clear()


@pytest.mark.asyncio
async def test_oversized_multipart_turn_sends_only_the_bounded_tail() -> None:
    """A multipart turn larger than the budget is truncated to its last bytes, and a
    multipart hosted-tool replay is dropped however large it is."""
    requests: list[httpx.Request] = []
    handler: Final = _handler(_selecting("openai/gpt-6-luna", requests))
    tight: Final = DrexRouteClassifier({**_config_dict(), "max_input_bytes": 11}, http_client=handler)

    assert (
        await tight.classify(
            _context(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "make it shorter: "},
                            {"type": "text", "text": "café"},
                        ],
                    }
                ]
            )
        )
        == "fast"
    )
    assert json.loads(requests[0].content)["messages"] == [{"role": "user", "content": "rter: café"}]

    roomy: Final = DrexRouteClassifier({**_config_dict(), "max_input_bytes": 200}, http_client=handler)
    assert (
        await roomy.classify(
            _context(
                [
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Hosted web search: "},
                            {"type": "text", "text": json.dumps({"query": "best fondue"}) * 100},
                        ],
                    },
                    {"role": "user", "content": "now summarize"},
                ]
            )
        )
        == "fast"
    )
    assert json.loads(requests[1].content)["messages"] == [{"role": "user", "content": "now summarize"}]
    await handler.close()


@pytest.mark.asyncio
async def test_candidate_map_is_frozen_at_construction() -> None:
    requests: list[httpx.Request] = []
    handler: Final = _handler(_selecting("anthropic/claude-sonnet-5.5", requests))
    validated: Final = DrexRouteClassifierConfig.model_validate(_config_dict())
    from_instance: Final = DrexRouteClassifier(validated, http_client=handler)
    validated.candidates["careful"] = "mutated/other-model"

    assert await from_instance.classify(_context()) == "careful"
    assert json.loads(requests[0].content)["candidates"] == _WIRE_CANDIDATES

    source: Final = _config_dict()
    from_mapping: Final = DrexRouteClassifier(source, http_client=handler)
    source["candidates"]["careful"] = "mutated/other-model"

    assert await from_mapping.classify(_context()) == "careful"
    assert json.loads(requests[1].content)["candidates"] == _WIRE_CANDIDATES
    await handler.close()


@pytest.mark.asyncio
async def test_classifier_stays_deep_copyable_after_a_call() -> None:
    """Router.set_model_list deep-copies deployments; nothing the classifier holds after a
    call may be a live HTTP client."""
    requests: list[httpx.Request] = []
    handler: Final = _handler(_selecting("openai/gpt-6-luna", requests))
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)

    assert await classifier.classify(_context()) == "fast"
    copied: Final = deepcopy(classifier)
    assert await copied.classify(_context()) == "fast"

    assert len(requests) == 2
    await handler.close()


@pytest.mark.asyncio
async def test_response_cookies_are_never_replayed() -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"selected": "openai/gpt-6-luna", "probabilities": {}, "request_id": "req"},
            headers={"Set-Cookie": "session=abc; Path=/"},
        )

    handler: Final = _handler(respond)
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)

    assert await classifier.classify(_context()) == "fast"
    assert await classifier.classify(_context()) == "fast"

    assert requests[1].headers.get("cookie") is None
    await handler.close()


@pytest.mark.asyncio
async def test_default_path_uses_litellms_shared_nace_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without an injected handler the classifier must ride LiteLLM's shared, per-loop
    NACE handler, whose client carries litellm's TLS, proxy and cookie configuration."""
    requests: list[httpx.Request] = []
    handler: Final = AsyncHTTPHandler(transport=httpx.MockTransport(_selecting("openai/gpt-6-luna", requests)))
    seen: list[object] = []

    def fake_shared_handler(llm_provider: object, **_: object) -> AsyncHTTPHandler:
        seen.append(llm_provider)
        return handler

    monkeypatch.setattr(drex_module, "get_async_httpx_client", fake_shared_handler)
    classifier: Final = DrexRouteClassifier(_config_dict())

    assert await classifier.classify(_context()) == "fast"

    assert seen == [LlmProviders.NACE]
    assert len(requests) == 1
    await handler.close()


def test_two_separate_event_loops_each_serve_the_same_classifier(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared handler is resolved per call and scoped per event loop, so one
    classifier instance works across sequential asyncio.run loops instead of reusing a
    dead loop's connection pool."""
    classifier: Final = DrexRouteClassifier(_config_dict())
    real_factory: Final = drex_module.get_async_httpx_client
    handlers: list[AsyncHTTPHandler] = []

    def spying_factory(*args: object, **kwargs: object) -> AsyncHTTPHandler:
        handler = real_factory(*args, **kwargs)  # pyright: ignore[reportAny]  # passthrough spy on the shared factory
        handlers.append(handler)
        return handler

    monkeypatch.setattr(drex_module, "get_async_httpx_client", spying_factory)

    def classify_once() -> str | None:
        async def call() -> str | None:
            with respx.mock(using="httpx", assert_all_called=True) as drex:
                drex.post("https://drex.test/v1/route").respond(
                    json={
                        "selected": "openai/gpt-6-luna",
                        "probabilities": {"openai/gpt-6-luna": 0.9},
                        "request_id": "req",
                    }
                )
                return await classifier.classify(_context())

        return asyncio.run(call())

    assert classify_once() == "fast"
    assert classify_once() == "fast"

    assert len(handlers) == 2
    assert handlers[0] is not handlers[1]


def _plugin_router(handler: AsyncHTTPHandler, **config_overrides: object) -> ComplexityRouter:
    return ComplexityRouter(
        model_name="auto",
        ctx=ApiRoutingContext(litellm.Router(model_list=[])),
        default_model="backup",
        complexity_router_config={
            "classifier_type": "custom",
            "classifier_plugin": DrexRouteClassifier(_config_dict(), http_client=handler),
            "classifier_fallback": "default_model",
            "tiers": {"fast": "fast-model", "careful": "careful-model", "custom": "custom-model"},
            **config_overrides,
        },
    )


@pytest.mark.asyncio
async def test_drex_selection_routes_to_the_configured_model() -> None:
    requests: list[httpx.Request] = []
    handler: Final = _handler(_selecting("anthropic/claude-sonnet-5.5", requests))
    router: Final = _plugin_router(handler)

    response: Final = await router.async_pre_routing_hook(
        model="auto", request_kwargs={}, messages=[{"role": "user", "content": "hello"}]
    )

    assert response is not None
    assert response.model == "careful-model"
    assert response.routing_decision is not None
    assert response.routing_decision["cause"] == "classifier_plugin"
    await handler.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ("status_402", "classifier_error"),
        ("status_422", "classifier_error"),
        ("status_429", "classifier_error"),
        ("status_503", "classifier_error"),
        ("status_529", "classifier_error"),
        ("status_202", "classifier_error"),
        ("oversized", "classifier_error"),
        ("compressed", "classifier_error"),
        ("read_timeout", "timeout"),
        ("missing_selected", "invalid_response"),
        ("bad_probability", "invalid_response"),
        ("non_float_probability", "invalid_response"),
        ("unknown_selected", "declined"),
    ],
)
async def test_routing_failure_falls_back_to_the_default_model(failure: str, reason: str) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if failure.startswith("status_"):
            return httpx.Response(int(failure.removeprefix("status_")), json={"error": "nope"})
        if failure == "read_timeout":
            raise httpx.ReadTimeout("read timed out")
        if failure == "missing_selected":
            return httpx.Response(200, json={"probabilities": {}, "request_id": "req_test"})
        if failure == "oversized":
            return httpx.Response(
                200,
                json={
                    "selected": "openai/gpt-6-luna",
                    "probabilities": {},
                    "request_id": "req_test",
                    "ignored": "x" * 70_000,
                },
            )
        if failure == "compressed":
            return httpx.Response(
                200,
                content=gzip.compress(json.dumps({"selected": "openai/gpt-6-luna", "probabilities": {}}).encode()),
                headers={"Content-Encoding": "gzip"},
            )
        if failure == "bad_probability":
            return httpx.Response(
                200, json={"selected": "openai/gpt-6-luna", "probabilities": {"openai/gpt-6-luna": 2.0}}
            )
        if failure == "non_float_probability":
            return httpx.Response(
                200, json={"selected": "openai/gpt-6-luna", "probabilities": {"openai/gpt-6-luna": "0.75"}}
            )
        return httpx.Response(
            200, json={"selected": "unknown-vendor/mystery", "probabilities": {}, "request_id": "req_test"}
        )

    handler: Final = _handler(respond)
    router: Final = _plugin_router(handler)

    response: Final = await router.async_pre_routing_hook(
        model="auto", request_kwargs={}, messages=[{"role": "user", "content": "hello"}]
    )

    assert response is not None
    assert response.model == "backup"
    assert response.routing_decision is not None
    assert response.routing_decision["classifier_failure_reason"] == reason
    await handler.close()


@pytest.mark.asyncio
async def test_router_failure_logs_carry_no_remote_response_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker: Final = "RESPONSE-ECHO-MARKER"

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"echo": marker, "request_id": marker})

    handler: Final = _handler(respond)
    router: Final = _plugin_router(handler)

    with caplog.at_level(logging.WARNING, logger=verbose_router_logger.name):
        response: Final = await router.async_pre_routing_hook(
            model="auto", request_kwargs={}, messages=[{"role": "user", "content": "hello"}]
        )

    assert response is not None
    assert response.model == "backup"
    assert marker not in caplog.text
    await handler.close()


@pytest.mark.asyncio
async def test_slow_drex_call_is_cancelled_at_the_plugin_timeout() -> None:
    never: Final = asyncio.Event()
    cancellation_seen: Final = asyncio.Event()

    async def respond(request: httpx.Request) -> httpx.Response:
        try:
            await never.wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            raise
        return httpx.Response(200, json={"selected": "openai/gpt-6-luna", "probabilities": {}})

    handler: Final = _handler(respond)
    router: Final = _plugin_router(handler, classifier_plugin_timeout_ms=20)

    response: Final = await router.async_pre_routing_hook(
        model="auto", request_kwargs={}, messages=[{"role": "user", "content": "hello"}]
    )

    assert response is not None
    assert response.model == "backup"
    assert response.routing_decision is not None
    assert response.routing_decision["classifier_failure_reason"] == "timeout"
    assert cancellation_seen.is_set()
    await handler.close()


@pytest.mark.asyncio
async def test_tools_stream_and_messages_are_left_unchanged() -> None:
    requests: list[httpx.Request] = []
    handler: Final = _handler(_selecting("anthropic/claude-sonnet-5.5", requests))
    router: Final = _plugin_router(handler)
    messages: Final = _image_messages()
    request_kwargs: Final = {
        "tools": _TOOLS,
        "stream": True,
        "metadata": {"user_api_key_hash": "test-caller-hash"},
    }
    messages_before: Final = deepcopy(messages)
    kwargs_before: Final = deepcopy(request_kwargs)

    response: Final = await router.async_pre_routing_hook(
        model="auto", request_kwargs=request_kwargs, messages=messages
    )

    assert response is not None
    assert response.model == "careful-model"
    assert messages == messages_before
    assert request_kwargs == kwargs_before
    assert response.messages is None or response.messages == messages_before
    await handler.close()


@pytest.mark.asyncio
async def test_streamed_completion_is_answered_by_the_selected_deployment() -> None:
    """The full Router path, Drex call included: the marker deployment deep-copies
    model_list, so the classifier rides along with no injected handler and resolves the
    shared NACE client per call."""
    router: Final = litellm.Router(model_list=_auto_model_list())
    with respx.mock(using="httpx", assert_all_called=True) as drex:
        drex.post("https://drex.test/v1/route").respond(
            json={
                "selected": "anthropic/claude-sonnet-5.5",
                "probabilities": {"anthropic/claude-sonnet-5.5": 0.9},
                "request_id": "req_test",
            }
        )
        stream = await router.acompletion(
            model="auto",
            messages=[{"role": "user", "content": "hello"}],
            tools=_TOOLS,
            stream=True,
        )
        chunks: Final = [chunk async for chunk in stream]
        text: Final = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)

    assert text == "answer from careful"


@pytest.mark.asyncio
async def test_set_model_list_succeeds_after_a_routed_completion() -> None:
    """Reloading the model list deep-copies the old deployments; a classifier that has
    already served a call must survive that copy, and the reloaded router must route."""
    router: Final = litellm.Router(model_list=_auto_model_list())
    with respx.mock(using="httpx", assert_all_called=False) as drex:
        drex.post("https://drex.test/v1/route").respond(
            json={
                "selected": "anthropic/claude-sonnet-5.5",
                "probabilities": {"anthropic/claude-sonnet-5.5": 0.9},
                "request_id": "req_test",
            }
        )
        response: Final = await router.acompletion(model="auto", messages=[{"role": "user", "content": "hello"}])
    assert response.choices[0].message.content == "answer from careful"

    router.set_model_list(_auto_model_list())

    with respx.mock(using="httpx", assert_all_called=False) as drex:
        drex.post("https://drex.test/v1/route").respond(
            json={
                "selected": "openai/gpt-6-luna",
                "probabilities": {"openai/gpt-6-luna": 0.9},
                "request_id": "req_test",
            }
        )
        reloaded: Final = await router.acompletion(model="auto", messages=[{"role": "user", "content": "hello"}])
    assert reloaded.choices[0].message.content == "answer from fast"


_TRUNCATED_RESPONSE: Final = b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{"
_SELECTING_BODY: Final = json.dumps({"selected": "openai/gpt-6-luna", "probabilities": {}}).encode()
_KEEP_ALIVE_RESPONSE: Final = (
    b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(_SELECTING_BODY)).encode() + b"\r\n\r\n" + _SELECTING_BODY
)


@pytest.mark.asyncio
async def test_detached_closes_are_capped_and_the_next_call_falls_back_without_a_new_exchange() -> None:
    """With a one-connection pool, every stuck close holds the only connection. Once
    max_detached_exchanges closes are pending, classify falls back without touching the
    transport, so neither detached tasks nor queued pool requests accumulate."""
    backend: Final = _SlowCloseBackend([_TRUNCATED_RESPONSE])
    handler: Final = _httpcore_handler(backend, max_connections=1)
    classifier: Final = DrexRouteClassifier(
        _config_dict(cleanup_grace_s=0.05, max_detached_exchanges=1), http_client=handler
    )
    first: Final = asyncio.create_task(classifier.classify(_context()))
    try:
        await asyncio.wait_for(backend.closing.wait(), 2)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(first, 1)
        assert len(drex_module._DETACHED_EXCHANGES) == 1  # pyright: ignore[reportPrivateUsage]  # the bound is the guarantee under test

        for _ in range(3):
            with pytest.raises(DrexRouteError, match="still closing"):
                _ = await asyncio.wait_for(classifier.classify(_context()), 1)
        assert backend.connects == 1
        assert len(drex_module._DETACHED_EXCHANGES) == 1  # pyright: ignore[reportPrivateUsage]

        backend.allow_close.set()
        await asyncio.wait_for(backend.released.wait(), 1)
        await _let_cancellations_land()
        assert not drex_module._DETACHED_EXCHANGES  # pyright: ignore[reportPrivateUsage]
        with pytest.raises(DrexRouteError, match="RemoteProtocolError"):
            _ = await asyncio.wait_for(classifier.classify(_context()), 1)
        assert backend.connects == 2
    finally:
        backend.allow_close.set()
        await asyncio.gather(first, return_exceptions=True)
        await handler.close()


def test_max_detached_exchanges_is_bounded() -> None:
    for value in (0, 65):
        with pytest.raises(ValidationError):
            _ = DrexRouteClassifierConfig(**_config_dict(max_detached_exchanges=value))


@pytest.mark.asyncio
async def test_cancelling_during_an_injected_transport_close_on_an_invalid_redirect_still_releases_it() -> None:
    """AsyncClient.send closes a response with an invalid redirect location itself, before
    returning it. The adapter sends through the transport directly, so the close is always
    the adapter's own, and a cancellation that lands in it waits for the release."""
    stream: Final = _SlowClosingStream()
    handler: Final = _handler(
        lambda request: httpx.Response(307, headers={"Location": "http://drex.test:abc/"}, stream=stream)
    )
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)
    task: Final = asyncio.create_task(classifier.classify(_context()))
    try:
        await asyncio.wait_for(stream.closing.wait(), 2)
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done(), "classify returned while the response stream was still closing"
        stream.allow_close.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.released.is_set()
    finally:
        stream.allow_close.set()
        await asyncio.gather(task, return_exceptions=True)
        await handler.close()


@pytest.mark.asyncio
async def test_cancelling_while_httpcore_closes_an_expired_keep_alive_connection_still_releases_it() -> None:
    """httpcore closes an expired idle connection at the start of the next request, inside
    its own shielded scope and without a response_closed trace event. A cancellation that
    lands there waits for the release."""
    backend: Final = _SlowCloseBackend([_KEEP_ALIVE_RESPONSE])
    handler: Final = _httpcore_handler(backend, keepalive_expiry=0.01)
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)
    assert await classifier.classify(_context()) == "fast"
    await asyncio.sleep(0.05)
    task: Final = asyncio.create_task(classifier.classify(_context()))
    try:
        await asyncio.wait_for(backend.closing.wait(), 2)
        task.cancel()
        await asyncio.sleep(0.05)
        assert not task.done(), "classify returned while the expired connection was still closing"
        backend.allow_close.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert backend.released.is_set()
    finally:
        backend.allow_close.set()
        await asyncio.gather(task, return_exceptions=True)
        await handler.close()


class _RecordingH2Stream(httpcore.AsyncMockStream):
    """An HTTP/2 network connection replaying scripted server bytes and recording writes."""

    def __init__(self, buffer: list[bytes], sent: list[bytes]) -> None:
        super().__init__(buffer, http2=True)
        self.sent: Final = sent

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        del timeout
        self.sent.append(buffer)


class _GoawayBackend(httpcore.AsyncMockBackend):
    """Connection one answers stream 1 and then sends GOAWAY(last_stream_id=1), so the
    next request on it (stream 3) is unprocessed and httpcore's pool resends it on a new
    connection, whose writes are recorded."""

    def __init__(self) -> None:
        super().__init__([], http2=True)
        self.writes: Final[list[list[bytes]]] = []

    async def connect_tcp(self, *args: object, **kwargs: object) -> httpcore.AsyncNetworkStream:
        del args, kwargs
        sent: Final[list[bytes]] = []
        self.writes.append(sent)
        return _RecordingH2Stream(_goaway_script() if len(self.writes) == 1 else [], sent)


def _goaway_script() -> list[bytes]:
    client: Final = h2.connection.H2Connection()
    server: Final = h2.connection.H2Connection(config=h2.config.H2Configuration(client_side=False))
    client.initiate_connection()
    server.initiate_connection()
    preface: Final = server.data_to_send()
    client.send_headers(1, [(":method", "POST"), (":path", "/"), (":scheme", "https"), (":authority", "x")], True)
    _ = server.receive_data(client.data_to_send())
    _ = server.data_to_send()
    server.send_headers(1, [(":status", "200"), ("content-length", str(len(_SELECTING_BODY)))])
    server.send_data(1, _SELECTING_BODY, end_stream=True)
    answer: Final = server.data_to_send()
    server.close_connection(last_stream_id=1)
    return [preface, answer, server.data_to_send()]


def _request_headers_received(written: list[bytes]) -> int:
    server: Final = h2.connection.H2Connection(config=h2.config.H2Configuration(client_side=False))
    events: Final = server.receive_data(b"".join(written))
    return sum(isinstance(event, h2.events.RequestReceived) for event in events)


@pytest.mark.asyncio
async def test_a_request_the_http2_peer_left_unprocessed_is_not_sent_again() -> None:
    """After GOAWAY says a stream was not processed, httpcore's pool resends the request
    on a new connection. The adapter allows one wire attempt per call, so the call falls
    back and the new connection never receives the request."""
    backend: Final = _GoawayBackend()
    handler: Final = _httpcore_handler(backend, http1=False, http2=True)
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)
    try:
        assert await classifier.classify(_context()) == "fast"

        with pytest.raises(DrexRouteError, match="RetryRefused"):
            _ = await classifier.classify(_context())

        assert len(backend.writes) == 2, "httpcore did not retry on a new connection"
        assert _request_headers_received(backend.writes[1]) == 0
    finally:
        await handler.close()


class _AlwaysRetryingTransport(httpx.AsyncBaseTransport):
    """Reports a request-header send through the trace extension twice, as httpcore's pool
    does when it retries, and records each send it was allowed to make."""

    def __init__(self) -> None:
        self.sends: int = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        trace: Final = request.extensions["trace"]
        for _ in range(2):
            await trace("http11.send_request_headers.started", {"request": request})
            self.sends += 1
        return httpx.Response(200, content=_SELECTING_BODY)


@pytest.mark.asyncio
async def test_a_transport_retry_is_refused_before_the_second_send() -> None:
    transport: Final = _AlwaysRetryingTransport()
    handler: Final = AsyncHTTPHandler(transport=transport)
    classifier: Final = DrexRouteClassifier(_config_dict(), http_client=handler)

    with pytest.raises(DrexRouteError, match="RetryRefused"):
        _ = await classifier.classify(_context())

    assert transport.sends == 1
    await handler.close()


class _ContextPlugin:
    marker: Final = contextvars.ContextVar[str | None]("drex_test_marker", default=None)

    async def classify(self, context: RoutingContext) -> str | None:
        del context
        _ = self.marker.set("set by the plugin")
        return "fast"


@pytest.mark.asyncio
async def test_plugin_context_changes_reach_the_caller_exactly_as_with_asyncio_wait_for() -> None:
    """3.12+ asyncio.wait_for runs the plugin in the caller's task and context; on 3.10 and
    3.11 it runs in a new task with a copied context. The deadline helper keeps both."""
    _ = await _scripted_router(_ContextPlugin()).async_pre_routing_hook(  # pyright: ignore[reportArgumentType]
        model="auto", request_kwargs={}, messages=[{"role": "user", "content": "hello"}]
    )

    expected: Final = "set by the plugin" if sys.version_info >= (3, 12) else None
    assert _ContextPlugin.marker.get() == expected


class _LatePlugin:
    """Finishes its own work just before the deadline, while the event loop is blocked past
    both, so the completion and the deadline are ready in the same loop turn."""

    async def classify(self, context: RoutingContext) -> str | None:
        del context
        asyncio.get_running_loop().call_soon(time.sleep, 0.08)
        await asyncio.sleep(0.01)
        return "fast"


@pytest.mark.asyncio
async def test_plugin_finishing_with_its_deadline_routes_exactly_as_asyncio_wait_for_does() -> None:
    """3.12+ asyncio.wait_for reports the deadline when it and the plugin's completion are
    ready together; 3.10 and 3.11 return the finished result. The routed model is the same
    as with plain asyncio.wait_for on each version."""
    response: Final = await _scripted_router(_LatePlugin(), timeout_ms=40).async_pre_routing_hook(  # pyright: ignore[reportArgumentType]
        model="auto", request_kwargs={}, messages=[{"role": "user", "content": "hello"}]
    )

    assert response is not None
    assert response.routing_decision is not None
    if sys.version_info >= (3, 12):
        assert response.model == "backup"
        assert response.routing_decision["classifier_failure_reason"] == "timeout"
    else:
        assert response.model == "fast-model"
        assert response.routing_decision.get("classifier_failure_reason") is None


@pytest.mark.asyncio
async def test_a_cancelled_caller_waits_at_most_the_grace_for_an_unrelated_plugin_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A custom classifier whose cleanup never finishes holds a cancelled caller for at most
    the cleanup grace, then is left to finish on its own."""
    monkeypatch.setattr(router_module, "_PLUGIN_CLEANUP_GRACE_S", 0.1)
    plugin: Final = _ScriptedPlugin(finish=False)
    routing: Final = _start_routing(_scripted_router(plugin, timeout_ms=20))
    try:
        await asyncio.wait_for(plugin.cleaning_up.wait(), 2)
        routing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(routing, 1)
        if sys.version_info < (3, 12):
            assert plugin.task is not None and not plugin.task.done()
            assert plugin.task in router_module._DETACHED_PLUGINS  # pyright: ignore[reportPrivateUsage]  # the strong reference is the guarantee under test
            plugin.allow_cleanup.set()
            await _let_cancellations_land()
            assert plugin.task.done()
            assert plugin.task not in router_module._DETACHED_PLUGINS  # pyright: ignore[reportPrivateUsage]
    finally:
        plugin.allow_cleanup.set()
        await asyncio.gather(routing, return_exceptions=True)


@_PRE_312
@pytest.mark.asyncio
async def test_plugin_cleanup_after_the_deadline_falls_back_after_the_grace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(router_module, "_PLUGIN_CLEANUP_GRACE_S", 0.1)
    plugin: Final = _ScriptedPlugin(finish=False)
    routing: Final = _start_routing(_scripted_router(plugin, timeout_ms=20))
    try:
        response: Final = await asyncio.wait_for(routing, 1)
        assert response is not None
        assert response.model == "backup"
        assert response.routing_decision is not None
        assert response.routing_decision["classifier_failure_reason"] == "timeout"
        assert plugin.task is not None and not plugin.task.done()
    finally:
        plugin.allow_cleanup.set()
        await asyncio.gather(routing, return_exceptions=True)
