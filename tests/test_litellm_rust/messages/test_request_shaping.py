"""The native Messages route shapes the wire request the way the Python handler does.

Model capability expectations come from model_prices_and_context_window.json (Claude Sonnet 5 is an
adaptive-thinking model without sampling params; Claude Haiku 4.5 is a legacy-thinking model), read at
2026-09-24; the cost map is LiteLLM's own file.
"""

import asyncio
from collections.abc import Iterator, Mapping
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


@pytest.mark.asyncio
async def test_native_messages_observes_runtime_capabilities_and_separate_caller_settings(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    from litellm.rust_bridge.messages.entrypoints import NATIVE_AMESSAGES
    from litellm.rust_bridge.public_call import NativeCall

    native: Final = NATIVE_AMESSAGES.load()
    assert native is not None
    model: Final = "claude-test-runtime-capabilities"
    messages_server.expected_requests = 2
    request: Final = NativeCall(
        args=(),
        kwargs={"temperature": 0.2, "drop_params": True},
        bound={
            "model": model,
            "messages": MESSAGES,
            "max_tokens": 16,
            "stream": None,
            "api_key": "test-key",
            "api_base": messages_server.base_url,
            "custom_llm_provider": "anthropic",
            "temperature": 0.2,
            "drop_params": True,
        },
    )
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {
            "litellm_provider": "anthropic",
            "mode": "chat",
            "supports_sampling_params": True,
        },
    )
    first: Final = await native(request)
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {
            "litellm_provider": "anthropic",
            "mode": "chat",
            "supports_sampling_params": False,
        },
    )
    second: Final = await native(request)

    assert isinstance(first, dict)
    assert isinstance(second, dict)
    assert first["id"] == second["id"] == MESSAGES_RESPONSE["id"]
    assert len(messages_server.requests) == 2
    first_body: Final = messages_server.requests[0].body
    second_body: Final = messages_server.requests[1].body
    assert isinstance(first_body, dict)
    assert isinstance(second_body, dict)
    assert first_body["temperature"] == request.kwargs["temperature"]
    assert second_body == {name: value for name, value in first_body.items() if name != "temperature"}


@pytest.mark.asyncio
async def test_native_messages_reads_optional_positional_body_parameters(messages_server: RecordingServer) -> None:
    from litellm.messages.dispatch import _MESSAGES, _public_request
    from litellm.rust_bridge.messages.entrypoints import NATIVE_AMESSAGES

    native: Final = NATIVE_AMESSAGES.load()
    assert native is not None
    metadata: Final = {"user_id": "caller"}
    args: Final = (16, MESSAGES, "anthropic/claude-test", metadata, None, False, "Be brief", 0.25)
    kwargs: Final = {"api_key": "test-key", "api_base": messages_server.base_url}
    call: Final = _public_request(_MESSAGES, args, kwargs)
    assert call is not None

    await native(call)

    body, _ = sent(messages_server)
    assert body["temperature"] == args[7]
    assert body["system"] == args[6]
    assert body["metadata"] == metadata


@pytest.fixture
def provider_fields_model(monkeypatch: pytest.MonkeyPatch) -> str:
    model: Final = "claude-test-provider-fields"
    monkeypatch.setitem(
        litellm.model_cost,
        model,
        {
            "litellm_provider": "anthropic",
            "mode": "chat",
            "supports_sampling_params": False,
            "supports_reasoning": True,
        },
    )
    return model


async def invoke_native_messages(
    server: RecordingServer, asynchronous: bool, model: str, options: Mapping[str, object]
) -> object:
    from litellm.rust_bridge.messages.entrypoints import NATIVE_AMESSAGES, NATIVE_MESSAGES
    from litellm.rust_bridge.public_call import NativeCall

    bound: Final = {
        "model": model,
        "messages": MESSAGES,
        "max_tokens": 32,
        "stream": False,
        "api_key": "test-key",
        "api_base": server.base_url,
        "custom_llm_provider": "anthropic",
        **options,
    }
    request: Final = NativeCall(args=(), kwargs=bound, bound=bound)
    if asynchronous:
        native_async: Final = NATIVE_AMESSAGES.load()
        assert native_async is not None
        return await native_async(request)
    native_sync: Final = NATIVE_MESSAGES.load()
    assert native_sync is not None
    return await asyncio.to_thread(native_sync, request)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("drop_params", [False, True], ids=["keep", "drop"])
@pytest.mark.parametrize("in_extra_body", [False, True], ids=["kwargs", "extra_body"])
@pytest.mark.parametrize(
    "extension", [None, {"mode": "future", "values": [True, None, {"nested": 7}]}], ids=["null", "nested"]
)
async def test_unknown_provider_fields_reach_messages_transport(
    messages_server: RecordingServer,
    provider_fields_model: str,
    asynchronous: bool,
    drop_params: bool,
    in_extra_body: bool,
    extension: object,
) -> None:
    fields: Final = {"future_provider_option": extension}
    options: Final = {"drop_params": drop_params, **({"extra_body": fields} if in_extra_body else fields)}

    await invoke_native_messages(messages_server, asynchronous, provider_fields_model, options)

    body, _ = sent(messages_server)
    assert body["future_provider_option"] == extension
    assert fields == {"future_provider_option": extension}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("field", "extension"),
    [
        ("service_tier", "future_tier"),
        ("tools", [{"type": "future_tool", "future_config": {"mode": "new"}}]),
        ("thinking", {"type": "future_mode", "future_config": {"values": [True, None]}}),
        ("thinking", {"type": "disabled", "future_hint": {"mode": "new"}}),
    ],
    ids=["future_enum", "future_tool", "future_thinking", "known_variant_extension"],
)
async def test_extensible_provider_types_reach_messages_transport(
    messages_server: RecordingServer,
    provider_fields_model: str,
    asynchronous: bool,
    field: str,
    extension: object,
) -> None:
    await invoke_native_messages(messages_server, asynchronous, provider_fields_model, {field: extension})

    body, _ = sent(messages_server)
    assert body[field] == extension


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_messages_resolves_overrides_before_additional_drop_params(
    messages_server: RecordingServer, provider_fields_model: str, asynchronous: bool
) -> None:
    options: Final = {
        "future_provider_option": {"old": True},
        "extra_body": {"future_provider_option": {"keep": True, "remove": {"nested": 7}}},
        "additional_drop_params": ["future_provider_option.remove"],
    }

    await invoke_native_messages(messages_server, asynchronous, provider_fields_model, options)

    body, _ = sent(messages_server)
    assert body["future_provider_option"] == {"keep": True}
    assert options["extra_body"] == {"future_provider_option": {"keep": True, "remove": {"nested": 7}}}


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("drop_params", [False, True], ids=["reject", "drop"])
async def test_messages_applies_model_policy_to_effective_extra_body_values(
    messages_server: RecordingServer, provider_fields_model: str, asynchronous: bool, drop_params: bool
) -> None:
    options: Final = {
        "temperature": 1.0,
        "extra_body": {"top_k": 5, "future_provider_option": {"keep": True}},
        "drop_params": drop_params,
    }
    if not drop_params:
        messages_server.expected_requests = 0
        with pytest.raises(litellm.BadRequestError, match="does not support top_k=5"):
            await invoke_native_messages(messages_server, asynchronous, provider_fields_model, options)
        assert messages_server.requests == []
        return

    await invoke_native_messages(messages_server, asynchronous, provider_fields_model, options)

    body, _ = sent(messages_server)
    assert "top_k" not in body
    assert body["future_provider_option"] == {"keep": True}
    assert body["temperature"] == options["temperature"]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_messages_decodes_effective_typed_values_before_transport(
    messages_server: RecordingServer, provider_fields_model: str, asynchronous: bool
) -> None:
    messages_server.expected_requests = 0

    with pytest.raises(litellm.BadRequestError, match="invalid type"):
        await invoke_native_messages(
            messages_server, asynchronous, provider_fields_model, {"extra_body": {"max_tokens": "invalid"}}
        )

    assert messages_server.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_messages_excludes_opaque_controls_from_extra_body(
    messages_server: RecordingServer, provider_fields_model: str, asynchronous: bool
) -> None:
    opaque: Final = object()
    overrides: Final = {
        "callbacks": [opaque],
        "api_key": opaque,
        "litellm_metadata": opaque,
        "model": "ignored",
        "messages": ["ignored"],
        "future_provider_option": {"keep": True},
    }

    await invoke_native_messages(messages_server, asynchronous, provider_fields_model, {"extra_body": overrides})

    body, headers = sent(messages_server)
    assert body["model"] == provider_fields_model
    assert body["messages"] == list(MESSAGES)
    assert body["future_provider_option"] == {"keep": True}
    assert not {"callbacks", "api_key", "litellm_metadata", "extra_body"} & body.keys()
    assert headers["x-api-key"] == "test-key"
    assert overrides["api_key"] is opaque


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_public_messages_preserves_unknown_kwargs_and_extra_body(
    messages_server: RecordingServer, provider_fields_model: str, asynchronous: bool
) -> None:
    options: Final = arguments(
        messages_server,
        model=provider_fields_model,
        custom_llm_provider="anthropic",
        future_provider_option=None,
        extra_body={"another_future_option": {"values": [True, None, 0]}},
        drop_params=True,
    )
    if asynchronous:
        await litellm.anthropic.messages.acreate(**options)
    else:
        await asyncio.to_thread(litellm.anthropic.messages.create, **options)

    body, _ = sent(messages_server)
    assert body["future_provider_option"] is None
    assert body["another_future_option"] == {"values": [True, None, 0]}
