"""The native Messages route shapes the wire request the way the Python handler does.

Model capability expectations come from model_prices_and_context_window.json (Claude Sonnet 5 is an
adaptive-thinking model without sampling params; Claude Haiku 4.5 is a legacy-thinking model), read at
2026-09-24; the cost map is LiteLLM's own file.
"""

from collections.abc import Iterator
from typing import Final

import pytest

import litellm
from litellm.rust_bridge import catalog
from litellm.rust_bridge.catalog import Route, RouteRule
from litellm.rust_bridge.configuration import Rollout
from tests.test_litellm_rust.support.isolation import rebound
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import MESSAGES, MESSAGES_RESPONSE

pytestmark = pytest.mark.requires_rust_extension

ADAPTIVE_MODEL: Final = "anthropic/claude-sonnet-5"
LEGACY_THINKING_MODEL: Final = "anthropic/claude-haiku-4-5"


@pytest.fixture(autouse=True)
def opt_messages_into_rust() -> Iterator[None]:
    with rebound(catalog, "RULES", (RouteRule(Route.MESSAGES, Rollout.RUST_OPT_IN), *catalog.RULES)):
        yield


@pytest.fixture
def messages_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    return recording_server


def arguments(server: RecordingServer, **kwargs: object) -> dict[str, object]:
    return {
        "model": ADAPTIVE_MODEL,
        "messages": [dict(message) for message in MESSAGES],
        "max_tokens": 8192,
        "api_key": "test-key",
        "api_base": server.base_url,
        **kwargs,
    }


def sent(server: RecordingServer) -> tuple[dict[str, object], dict[str, str]]:
    assert len(server.requests) == 1
    request: Final = server.requests[0]
    assert not request.headers.get("user-agent", "").startswith("python-httpx")
    assert isinstance(request.body, dict)
    return request.body, request.headers


@pytest.mark.asyncio
async def test_reasoning_effort_becomes_adaptive_thinking_and_effort_on_the_wire(
    messages_server: RecordingServer,
) -> None:
    await litellm.anthropic.messages.acreate(**arguments(messages_server, reasoning_effort="high"))

    body, _ = sent(messages_server)
    assert "reasoning_effort" not in body
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "high"}


@pytest.mark.asyncio
async def test_claude_code_adaptive_payload_is_downgraded_to_a_capped_budget_for_a_legacy_model(
    messages_server: RecordingServer,
) -> None:
    await litellm.anthropic.messages.acreate(
        **arguments(
            messages_server,
            model=LEGACY_THINKING_MODEL,
            max_tokens=3000,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            temperature=0,
        )
    )

    body, _ = sent(messages_server)
    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2999}
    assert "output_config" not in body
    assert "temperature" not in body


@pytest.mark.asyncio
async def test_removed_sampling_params_are_dropped_under_drop_params(messages_server: RecordingServer) -> None:
    await litellm.anthropic.messages.acreate(
        **arguments(messages_server, temperature=0.2, top_p=0.9, top_k=5, drop_params=True)
    )

    body, _ = sent(messages_server)
    assert not {"temperature", "top_p", "top_k"} & body.keys()


@pytest.mark.asyncio
async def test_removed_sampling_params_are_rejected_without_drop_params(messages_server: RecordingServer) -> None:
    messages_server.expected_requests = 0

    with pytest.raises(litellm.BadRequestError, match="does not support top_k=5"):
        await litellm.anthropic.messages.acreate(**arguments(messages_server, top_k=5))

    assert messages_server.requests == []


@pytest.mark.asyncio
async def test_replayed_history_is_sanitized_before_it_reaches_the_provider(
    messages_server: RecordingServer,
) -> None:
    history: Final = [
        {"role": "user", "content": "run it"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": ""},
                {
                    "type": "tool_use",
                    "id": "functions.Bash:0",
                    "name": "Bash",
                    "input": {},
                    "provider_specific_fields": {"x": 1},
                },
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "functions.Bash:0", "content": "ok"}]},
    ]

    await litellm.anthropic.messages.acreate(**arguments(messages_server, messages=history))

    body, _ = sent(messages_server)
    assert body["messages"] == [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "functions_Bash_0", "name": "Bash", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "functions_Bash_0", "content": "ok"}]},
    ]


@pytest.mark.asyncio
async def test_feature_betas_merge_into_the_forwarded_beta_header(messages_server: RecordingServer) -> None:
    await litellm.anthropic.messages.acreate(
        **arguments(
            messages_server,
            output_format={"type": "json_schema", "schema": {"type": "object"}},
            extra_headers={"anthropic-beta": "web-search-2025-03-05"},
        )
    )

    _, headers = sent(messages_server)
    assert headers["anthropic-beta"] == "structured-outputs-2025-11-13,web-search-2025-03-05"


@pytest.mark.asyncio
async def test_oauth_token_authenticates_as_a_bearer_with_the_oauth_beta(messages_server: RecordingServer) -> None:
    await litellm.anthropic.messages.acreate(**arguments(messages_server, api_key="sk-ant-oat01-token"))

    _, headers = sent(messages_server)
    assert "x-api-key" not in headers
    assert headers["authorization"] == "Bearer sk-ant-oat01-token"
    assert headers["anthropic-beta"] == "oauth-2025-04-20"
    assert headers["anthropic-dangerous-direct-browser-access"] == "true"


@pytest.mark.asyncio
async def test_metadata_is_reduced_to_the_fields_anthropic_accepts(messages_server: RecordingServer) -> None:
    await litellm.anthropic.messages.acreate(
        **arguments(messages_server, metadata={"user_id": "u-1", "trace_id": "internal"})
    )

    body, _ = sent(messages_server)
    assert body["metadata"] == {"user_id": "u-1"}


@pytest.mark.asyncio
async def test_additional_drop_params_remove_nested_fields_from_the_wire(messages_server: RecordingServer) -> None:
    tools: Final = [{"name": "lookup", "input_schema": {"type": "object"}, "input_examples": [{"q": "x"}]}]

    await litellm.anthropic.messages.acreate(
        **arguments(messages_server, tools=tools, additional_drop_params=["tools[*].input_examples"])
    )

    body, _ = sent(messages_server)
    assert body["tools"] == [{"name": "lookup", "input_schema": {"type": "object"}}]


@pytest.mark.asyncio
async def test_provider_specific_headers_scoped_to_anthropic_reach_the_wire(messages_server: RecordingServer) -> None:
    await litellm.anthropic.messages.acreate(
        **arguments(
            messages_server,
            provider_specific_header=[
                {"custom_llm_provider": "anthropic, azure_ai", "extra_headers": {"x-scoped": "yes"}},
                {"custom_llm_provider": "openai", "extra_headers": {"x-other": "no"}},
            ],
        )
    )

    _, headers = sent(messages_server)
    assert headers["x-scoped"] == "yes"
    assert "x-other" not in headers


@pytest.mark.asyncio
async def test_scoped_headers_override_extra_headers_which_override_forwarded_headers(
    messages_server: RecordingServer,
) -> None:
    await litellm.anthropic.messages.acreate(
        **arguments(
            messages_server,
            headers={"x-priority": "forwarded", "x-forwarded-only": "kept"},
            extra_headers={"x-priority": "extra", "x-extra-only": "kept"},
            provider_specific_header={"custom_llm_provider": "anthropic", "extra_headers": {"x-priority": "scoped"}},
        )
    )

    _, headers = sent(messages_server)
    assert {name: headers.get(name) for name in ("x-priority", "x-forwarded-only", "x-extra-only")} == {
        "x-priority": "scoped",
        "x-forwarded-only": "kept",
        "x-extra-only": "kept",
    }


@pytest.mark.asyncio
async def test_non_string_metadata_user_id_is_rejected_before_the_provider_call(
    messages_server: RecordingServer,
) -> None:
    messages_server.expected_requests = 0

    with pytest.raises(litellm.BadRequestError, match=r"metadata\.user_id must be a string"):
        await litellm.anthropic.messages.acreate(**arguments(messages_server, metadata={"user_id": 123}))

    assert messages_server.requests == []
