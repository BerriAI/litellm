from typing import Final

import pytest

import litellm
from tests.test_litellm_rust.messages.data import MESSAGES, MESSAGES_RESPONSE, call
from tests.test_litellm_rust.support.recording_server import RecordingServer
from tests.test_litellm_rust.support.routing import assert_served_natively

pytestmark = pytest.mark.requires_rust_extension

CAPABILITY_MODEL: Final = "claude-test-runtime-capabilities"


def sent(server: RecordingServer) -> tuple[dict[str, object], dict[str, str]]:
    assert_served_natively(server)
    request: Final = server.requests[0]
    assert isinstance(request.body, dict)
    return request.body, request.headers


def capabilities(monkeypatch: pytest.MonkeyPatch, *, sampling: bool) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        CAPABILITY_MODEL,
        {"litellm_provider": "anthropic", "mode": "chat", "supports_sampling_params": sampling},
    )


@pytest.mark.asyncio
async def test_replayed_tool_ids_are_sanitized_before_they_reach_the_provider(server: RecordingServer) -> None:
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

    await call(server, messages=history)

    body, _ = sent(server)
    assert body["messages"] == [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "functions_Bash_0", "name": "Bash", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "functions_Bash_0", "content": "ok"}]},
    ]


@pytest.mark.asyncio
async def test_provider_specific_headers_scoped_to_anthropic_reach_the_wire(server: RecordingServer) -> None:
    await call(
        server,
        provider_specific_header=[
            {"custom_llm_provider": "anthropic, azure_ai", "extra_headers": {"x-scoped": "yes"}},
            {"custom_llm_provider": "openai", "extra_headers": {"x-other": "no"}},
        ],
    )

    _, headers = sent(server)
    assert headers["x-scoped"] == "yes"
    assert "x-other" not in headers


@pytest.mark.asyncio
async def test_scoped_headers_override_extra_headers_which_override_forwarded_headers(server: RecordingServer) -> None:
    await call(
        server,
        headers={"x-priority": "forwarded", "x-forwarded-only": "kept"},
        extra_headers={"x-priority": "extra", "x-extra-only": "kept"},
        provider_specific_header={"custom_llm_provider": "anthropic", "extra_headers": {"x-priority": "scoped"}},
    )

    _, headers = sent(server)
    assert {name: headers.get(name) for name in ("x-priority", "x-forwarded-only", "x-extra-only")} == {
        "x-priority": "scoped",
        "x-forwarded-only": "kept",
        "x-extra-only": "kept",
    }


@pytest.mark.asyncio
async def test_model_capabilities_are_read_from_model_cost_at_call_time(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.expected_requests = 2
    model: Final = f"anthropic/{CAPABILITY_MODEL}"

    capabilities(monkeypatch, sampling=True)
    await call(server, model=model, temperature=0.2, drop_params=True)
    capabilities(monkeypatch, sampling=False)
    await call(server, model=model, temperature=0.2, drop_params=True)

    assert_served_natively(server, count=2)
    first, second = (request.body for request in server.requests)
    assert isinstance(first, dict)
    assert first["temperature"] == 0.2
    assert second == {name: value for name, value in first.items() if name != "temperature"}


@pytest.mark.asyncio
async def test_global_drop_params_drops_unsupported_params(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    capabilities(monkeypatch, sampling=False)
    monkeypatch.setattr(litellm, "drop_params", True)

    await call(server, model=f"anthropic/{CAPABILITY_MODEL}", top_k=5)

    body, _ = sent(server)
    assert "top_k" not in body


@pytest.mark.asyncio
async def test_unsupported_params_are_rejected_before_io_without_drop_params(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    capabilities(monkeypatch, sampling=False)
    monkeypatch.setattr(litellm, "drop_params", False)
    server.expected_requests = 0

    with pytest.raises(litellm.BadRequestError, match="top_k"):
        await call(server, model=f"anthropic/{CAPABILITY_MODEL}", top_k=5)

    assert server.requests == []


@pytest.mark.asyncio
async def test_optional_positional_body_parameters_reach_the_wire(server: RecordingServer) -> None:
    metadata: Final = {"user_id": "caller"}
    args: Final = (16, list(MESSAGES), "anthropic/claude-test", metadata, None, False, "Be brief", 0.25)

    response: Final = await litellm.anthropic_messages(*args, api_key="test-key", api_base=server.base_url)

    assert isinstance(response, dict)
    assert response["id"] == MESSAGES_RESPONSE["id"]
    body, _ = sent(server)
    assert (body["metadata"], body["system"], body["temperature"]) == (args[3], args[6], args[7])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "workspace_field", ("workspace_id", "aws_workspace_id", "anthropic_workspace_id", "anthropic-workspace-id")
)
async def test_platform_workspace_and_unsupported_override_reach_the_native_adapter(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch, workspace_field: str
) -> None:
    monkeypatch.delenv("ANTHROPIC_AWS_WORKSPACE_ID", raising=False)
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    response: Final = await call(
        server,
        model="bedrock/claude_platform/claude-test",
        asynchronous=True,
        **{workspace_field: "deployment-workspace"},
        claude_platform_unsupported_params=["temperature"],
        temperature=0.3,
        tools=[
            {"name": "lookup", "input_schema": {"type": "object", "properties": {"temperature": {"type": "number"}}}}
        ],
    )
    assert_served_natively(server, response)
    body, headers = sent(server)
    assert headers["anthropic-workspace-id"] == "deployment-workspace"
    assert "temperature" not in body
    assert "claude_platform_unsupported_params" not in body
    assert workspace_field not in body
    assert body["tools"] == [
        {"name": "lookup", "input_schema": {"type": "object", "properties": {"temperature": {"type": "number"}}}}
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("max_tokens", (1.5, True, "16", None, [], {}))
async def test_malformed_max_tokens_is_rejected_natively_before_io(server: RecordingServer, max_tokens: object) -> None:
    server.expected_requests = 0
    with pytest.raises(
        litellm.BadRequestError, match=r"invalid Anthropic messages request|missing required field: max_tokens"
    ):
        await call(server, max_tokens=max_tokens)
    assert_served_natively(server, count=0)
