import asyncio
from base64 import b64encode
from copy import deepcopy
from threading import get_ident
from typing import Final

import httpx
import pytest
import respx
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.llms.anthropic.count_tokens.handler import AnthropicCountTokensHandler
from litellm.llms.anthropic.count_tokens.transformation import (
    AnthropicCountTokensConfig,
)
from litellm.llms.azure_ai.anthropic.count_tokens.handler import (
    AzureAIAnthropicCountTokensHandler,
)
from litellm.llms.azure_ai.anthropic.count_tokens.transformation import (
    AzureAIAnthropicCountTokensConfig,
)


@pytest.mark.parametrize(
    "config_type", (AnthropicCountTokensConfig, AzureAIAnthropicCountTokensConfig)
)
@pytest.mark.parametrize(
    ("image_url", "source"),
    (
        ("data:image/png;base64,aW1hZ2U=", {"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="}),
        ({"url": "data:image/png;base64,aW1hZ2U="}, {"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="}),
        (
            {"url": "data:image/png;base64,aW1hZ2U=", "format": "image/jpeg", "detail": "high"},
            {"type": "base64", "media_type": "image/jpeg", "data": "aW1hZ2U="},
        ),
    ),
)
def test_count_translates_openai_images_without_mutating_input(
    config_type: type[AnthropicCountTokensConfig],
    image_url: str | dict[str, JsonValue],
    source: dict[str, JsonValue],
) -> None:
    cache_control: Final[dict[str, JsonValue]] = {"type": "ephemeral"}
    messages: Final[list[dict[str, JsonValue]]] = [{
        "role": "user", "content": [
            {"type": "text", "text": "Count this image"},
            {"type": "image_url", "image_url": image_url, "cache_control": cache_control},
        ],
    }]
    original: Final = deepcopy(messages)
    result: Final = config_type().transform_request_to_count_tokens(
        model="claude-opus-5-5", messages=messages
    )

    assert result == {
        "model": "claude-opus-5-5", "messages": [{
            "role": "user", "content": [
                {"type": "text", "text": "Count this image"},
                {"type": "image", "source": source, "cache_control": cache_control},
            ],
        }],
    }
    assert messages == original


@pytest.mark.parametrize(
    "config_type", (AnthropicCountTokensConfig, AzureAIAnthropicCountTokensConfig)
)
def test_count_normalizes_nested_tool_images_and_preserves_native_fields(
    config_type: type[AnthropicCountTokensConfig],
) -> None:
    openai_image: Final[dict[str, JsonValue]] = {
        "type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}
    }
    native_image: Final[dict[str, JsonValue]] = {
        "type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="}
    }
    assistant: Final[dict[str, JsonValue]] = {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "inspect screenshot", "signature": "fixture-signature"},
        {"type": "tool_use", "id": "read-1", "name": "Read", "input": {"content": [openai_image]}},
    ]}
    tool_result: Final[dict[str, JsonValue]] = {
        "type": "tool_result", "tool_use_id": "read-1", "is_error": False,
        "content": [{"type": "text", "text": "Screenshot"}, native_image, openai_image],
        "cache_control": {"type": "ephemeral"},
    }
    text_result: Final[dict[str, JsonValue]] = {"type": "tool_result", "tool_use_id": "read-2", "content": "done"}
    messages: Final[list[dict[str, JsonValue]]] = [
        assistant, {"role": "user", "content": [native_image, tool_result, text_result]}
    ]
    tools: Final[list[dict[str, JsonValue]]] = [{
        "name": "Read", "input_schema": {"type": "object", "examples": [openai_image]}
    }]
    system: Final[JsonValue] = [{"type": "text", "text": "policy", "cache_control": {"type": "ephemeral"}}]
    options: Final[dict[str, JsonValue]] = {
        "thinking": {"type": "adaptive"}, "tool_choice": {"type": "auto"}, "output_config": {"effort": "high"}
    }
    original: Final = deepcopy((messages, tools, system, options))
    result: Final = config_type().transform_request_to_count_tokens(
        model="claude-opus-5-5", messages=messages, tools=tools, system=system, optional_params=options
    )

    assert result == {
        "model": "claude-opus-5-5", "system": system, "tools": tools, **options,
        "messages": [assistant, {"role": "user", "content": [native_image, {
            **tool_result, "content": [{"type": "text", "text": "Screenshot"}, native_image, native_image]
        }, text_result]}],
    }
    assert (messages, tools, system, options) == original


def test_transform_basic_request():
    """Test basic request with only model and messages."""
    config = AnthropicCountTokensConfig()

    result = config.transform_request_to_count_tokens(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
    )

    assert result == {
        "model": "claude-3-5-sonnet",
        "messages": [{"role": "user", "content": "Hello"}],
    }


def test_transform_includes_system():
    """Test that system prompt is included when provided."""
    config = AnthropicCountTokensConfig()

    result = config.transform_request_to_count_tokens(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        system="You are a helpful assistant.",
    )

    assert result["system"] == "You are a helpful assistant."
    assert result["model"] == "claude-3-5-sonnet"
    assert result["messages"] == [{"role": "user", "content": "Hello"}]


def test_transform_includes_tools():
    """Test that tools are included when provided."""
    config = AnthropicCountTokensConfig()

    tools = [
        {
            "name": "read_file",
            "description": "Read a file",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        }
    ]

    result = config.transform_request_to_count_tokens(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        tools=tools,
    )

    assert result["tools"] == tools


def test_transform_includes_system_and_tools():
    """Test that both system and tools are included together."""
    config = AnthropicCountTokensConfig()

    result = config.transform_request_to_count_tokens(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        system="Be helpful",
        tools=[{"name": "my_tool", "input_schema": {"type": "object"}}],
    )

    assert "system" in result
    assert "tools" in result
    assert "messages" in result
    assert "model" in result


def test_transform_no_system_no_tools():
    """Test that None system/tools are not included."""
    config = AnthropicCountTokensConfig()

    result = config.transform_request_to_count_tokens(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "Hello"}],
        system=None,
        tools=None,
    )

    assert "system" not in result
    assert "tools" not in result


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        (None, "https://api.anthropic.com/v1/messages/count_tokens"),
        ("", "https://api.anthropic.com/v1/messages/count_tokens"),
        ("https://gateway.example", "https://gateway.example/v1/messages/count_tokens"),
        ("https://gateway.example/", "https://gateway.example/v1/messages/count_tokens"),
        ("https://gateway.example/v1", "https://gateway.example/v1/messages/count_tokens"),
        ("https://gateway.example/anthropic/v1/messages", "https://gateway.example/anthropic/v1/messages/count_tokens"),
    ],
)
def test_endpoint_appends_count_tokens_path_to_deployment_api_base(api_base, expected, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    assert AnthropicCountTokensConfig().get_anthropic_count_tokens_endpoint(api_base) == expected


@pytest.mark.parametrize("env_name", ["ANTHROPIC_API_BASE", "ANTHROPIC_BASE_URL"])
@pytest.mark.parametrize("api_base", [None, ""])
def test_endpoint_without_deployment_api_base_follows_env_base(env_name, api_base, monkeypatch):
    """Chat and the federated exchange resolve an unset deployment base through the environment,
    so an env-only gateway must receive the count too, never Anthropic's public host."""
    monkeypatch.delenv("ANTHROPIC_API_BASE", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.setenv(env_name, "https://env-gateway.example/v1/messages/")
    assert (
        AnthropicCountTokensConfig().get_anthropic_count_tokens_endpoint(api_base)
        == "https://env-gateway.example/v1/messages/count_tokens"
    )


def test_endpoint_prefers_deployment_api_base_over_env_base(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_BASE", "https://env-gateway.example")
    assert (
        AnthropicCountTokensConfig().get_anthropic_count_tokens_endpoint("https://gateway.example/v1")
        == "https://gateway.example/v1/messages/count_tokens"
    )


@pytest.fixture
def httpx_transport_clients(monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    client_cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    if client_cache is not None:
        client_cache.flush_cache()
    yield
    if client_cache is not None:
        client_cache.flush_cache()


@pytest.mark.asyncio
async def test_handler_posts_to_count_tokens_path_under_deployment_api_base(httpx_transport_clients):
    """A deployment api_base names the chat host, so a handler that posts to it verbatim lands on
    the host root, gets a 404, and the official count silently degrades to the local tokenizer."""
    with respx.mock:
        route = respx.post("https://gateway.example/v1/messages/count_tokens").mock(
            return_value=httpx.Response(200, json={"input_tokens": 7})
        )
        result = await AnthropicCountTokensHandler().handle_count_tokens_request(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            auth_header={"x-api-key": "sk-ant-api03-test-key"},
            api_base="https://gateway.example",
        )

    assert route.called
    assert result == {"input_tokens": 7}


@pytest.mark.asyncio
@pytest.mark.usefixtures("httpx_transport_clients")
@pytest.mark.parametrize(
    "handler_type", (AnthropicCountTokensHandler, AzureAIAnthropicCountTokensHandler)
)
@pytest.mark.parametrize("scheme", ("http", "https"))
@pytest.mark.parametrize("dict_url", (False, True))
async def test_remote_image_fetch_keeps_counting_handler_event_loop_responsive(
    handler_type: type[AnthropicCountTokensHandler] | type[AzureAIAnthropicCountTokensHandler],
    scheme: str,
    dict_url: bool,
) -> None:
    loop: Final = asyncio.get_running_loop()
    loop_thread: Final = get_ident()
    witness: Final = asyncio.Event()
    image_bytes: Final = b"\x89PNG\r\n\x1a\ncount-image"
    image_url: Final = f"{scheme}://1.1.1.1/{handler_type.__name__}-{dict_url}.png"
    model: Final = "claude-opus-5-5"
    api_base: Final = "https://gateway.example/anthropic"
    image: Final[dict[str, JsonValue]] = {
        "type": "image_url", "image_url": {"url": image_url} if dict_url else image_url,
        "cache_control": {"type": "ephemeral"},
    }
    messages: Final[list[dict[str, JsonValue]]] = [{
        "role": "user", "content": [image, {"type": "tool_result", "tool_use_id": "read-1", "content": [image]}]
    }]
    original: Final = deepcopy(messages)

    async def run_witness() -> None:
        witness.set()

    def image_response(_request: httpx.Request) -> httpx.Response:
        assert get_ident() != loop_thread, "image fetch blocked the counting handler's event loop"
        asyncio.run_coroutine_threadsafe(run_witness(), loop).result(timeout=5)
        return httpx.Response(200, content=image_bytes, headers={"Content-Type": "image/png"})

    with respx.mock:
        image_route: Final = respx.get(image_url).mock(side_effect=image_response)
        count_route: Final = respx.post(f"{api_base}/v1/messages/count_tokens").mock(
            return_value=httpx.Response(200, json={"input_tokens": 7})
        )
        handler: Final = handler_type()
        result: Final = await (
            handler.handle_count_tokens_request(
                model=model, messages=messages, api_base=api_base, auth_header={"x-api-key": "test-key"}
            ) if isinstance(handler, AnthropicCountTokensHandler) else handler.handle_count_tokens_request(
                model=model, messages=messages, api_base=api_base, api_key="test-key"
            )
        )

    assert witness.is_set()
    assert image_route.call_count == count_route.call_count == 1
    assert result == {"input_tokens": 7}
    native_image: Final[dict[str, JsonValue]] = {
        "type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": b64encode(image_bytes).decode()
        }, "cache_control": {"type": "ephemeral"},
    }
    assert TypeAdapter(dict[str, JsonValue]).validate_json(count_route.calls.last.request.content) == {
        "model": model, "messages": [{"role": "user", "content": [
            native_image, {"type": "tool_result", "tool_use_id": "read-1", "content": [native_image]}
        ]}],
    }
    assert messages == original


@pytest.mark.parametrize(
    "config_type", (AnthropicCountTokensConfig, AzureAIAnthropicCountTokensConfig)
)
def test_count_lifts_the_leading_system_run_into_system(
    config_type: type[AnthropicCountTokensConfig],
) -> None:
    """A Responses ``instructions`` arrives as a leading system-role message. count_tokens answers 400
    on that role at the head of ``messages`` and only takes the initial prompt in ``system``, so the
    leading run moves there with its cache_control, empty text dropped, and a later reminder stays."""
    cache_control: Final[dict[str, JsonValue]] = {"type": "ephemeral"}
    messages: Final[list[dict[str, JsonValue]]] = [
        {"role": "system", "content": "Be terse", "cache_control": cache_control},
        {"role": "system", "content": [{"type": "text", "text": "Answer in French"}, {"type": "text", "text": ""}]},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "system", "content": "later reminder"},
        {"role": "assistant", "content": "Bonjour."},
    ]
    original: Final = deepcopy(messages)
    result: Final = config_type().transform_request_to_count_tokens(model="claude-opus-5-5", messages=messages)

    assert result == {
        "model": "claude-opus-5-5",
        "system": [
            {"type": "text", "text": "Be terse", "cache_control": cache_control},
            {"type": "text", "text": "Answer in French"},
        ],
        "messages": [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "system", "content": "later reminder"},
            {"role": "assistant", "content": "Bonjour."},
        ],
    }
    assert messages == original


@pytest.mark.parametrize(
    ("system", "expected_system"),
    (
        (None, [{"type": "text", "text": "Be terse"}]),
        ("", [{"type": "text", "text": "Be terse"}]),
        ("Answer in French", [{"type": "text", "text": "Answer in French"}, {"type": "text", "text": "Be terse"}]),
        (
            [{"type": "text", "text": "Answer in French", "cache_control": {"type": "ephemeral"}}],
            [
                {"type": "text", "text": "Answer in French", "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": "Be terse"},
            ],
        ),
    ),
    ids=["absent", "empty", "string", "blocks"],
)
def test_count_keeps_the_callers_system_ahead_of_the_lifted_run(
    system: JsonValue, expected_system: list[dict[str, JsonValue]]
) -> None:
    result: Final = AnthropicCountTokensConfig().transform_request_to_count_tokens(
        model="claude-opus-5-5",
        messages=[{"role": "system", "content": "Be terse"}, {"role": "user", "content": "hi"}],
        system=system,
    )

    assert result == {
        "model": "claude-opus-5-5",
        "system": expected_system,
        "messages": [{"role": "user", "content": "hi"}],
    }


def test_count_leaves_a_non_text_system_and_its_messages_as_sent() -> None:
    """A malformed ``system`` is the provider's to reject, so nothing is rearranged around it."""
    messages: Final[list[dict[str, JsonValue]]] = [
        {"role": "system", "content": "Be terse"},
        {"role": "user", "content": "hi"},
    ]
    result: Final = AnthropicCountTokensConfig().transform_request_to_count_tokens(
        model="claude-opus-5-5", messages=messages, system=5
    )

    assert result == {"model": "claude-opus-5-5", "system": 5, "messages": messages}


def test_count_drops_a_leading_system_message_without_text() -> None:
    result: Final = AnthropicCountTokensConfig().transform_request_to_count_tokens(
        model="claude-opus-5-5",
        messages=[{"role": "system", "content": ""}, {"role": "user", "content": "hi"}],
    )

    assert result == {"model": "claude-opus-5-5", "messages": [{"role": "user", "content": "hi"}]}


@pytest.mark.asyncio
async def test_handler_sends_the_leading_system_run_as_system_not_as_a_message(httpx_transport_clients):
    """The wire body is what the provider judges: ``system`` carries the prompt and no message has
    ``role: "system"``, so a Responses ``instructions`` is counted by Anthropic instead of 400ing."""
    with respx.mock:
        route = respx.post("https://gateway.example/v1/messages/count_tokens").mock(
            return_value=httpx.Response(200, json={"input_tokens": 21})
        )
        result = await AnthropicCountTokensHandler().handle_count_tokens_request(
            model="claude-opus-5-5",
            messages=[{"role": "system", "content": "Be terse"}, {"role": "user", "content": "Hello, how are you?"}],
            auth_header={"x-api-key": "sk-ant-api03-test-key"},
            api_base="https://gateway.example",
        )

    assert result == {"input_tokens": 21}
    assert TypeAdapter(dict[str, JsonValue]).validate_json(route.calls.last.request.content) == {
        "model": "claude-opus-5-5",
        "system": [{"type": "text", "text": "Be terse"}],
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
    }
