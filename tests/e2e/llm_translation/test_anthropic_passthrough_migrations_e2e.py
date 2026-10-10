from typing import Final, cast

import pytest
from anthropic.types import Message, MessageParam, TextBlock, ToolUnionParam
from e2e_config import unique_marker
from e2e_metadata import Domain, Mode, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import NO_PROXY_CACHE, SdkClients

BEDROCK_MODEL_ID: Final = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
BEDROCK_BACKENDS: Final = (
    pytest.param(f"bedrock/converse/{BEDROCK_MODEL_ID}", id="bedrock-converse"),
    pytest.param(f"bedrock/invoke/{BEDROCK_MODEL_ID}", id="bedrock-invoke"),
)
ANTHROPIC_MODEL: Final = "anthropic/claude-sonnet-4-5-20250929"
TOOL_SEARCH_BACKENDS: Final = (
    pytest.param(ANTHROPIC_MODEL, id="anthropic"),
    *BEDROCK_BACKENDS,
)
def _tool_search_tools() -> list[ToolUnionParam]:
    return [
        cast(
            ToolUnionParam,
            {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
        ),
        cast(
            ToolUnionParam,
            {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
                "defer_loading": True,
            },
        ),
        cast(
            ToolUnionParam,
            {
                "name": "get_stock_price",
                "description": "Get the current stock price for a ticker symbol",
                "input_schema": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"],
                },
                "defer_loading": True,
            },
        ),
    ]
CACHED_DOCUMENT: Final = "This agreement describes payment terms, renewal dates, and service obligations. " * 120

pytestmark = pytest.mark.e2e


def _register_deployment(
    proxy: ProxyClient,
    resources: ResourceManager,
    params: LiteLLMParamsBody,
    prefix: str,
) -> tuple[str, str]:
    model_name: Final = f"{prefix}-{unique_marker()}"
    model_id: Final = proxy.create_model(model_name, params)
    resources.defer(lambda: proxy.delete_model(model_id))
    return model_name, resources.key()


def _deployment_params(backend: str) -> LiteLLMParamsBody:
    if not backend.startswith("bedrock/"):
        return LiteLLMParamsBody(model=backend, api_key="os.environ/ANTHROPIC_API_KEY")
    return LiteLLMParamsBody(
        model=backend,
        aws_access_key_id="os.environ/AWS_BEDROCK_TEST_ACCESS_KEY_ID",
        aws_secret_access_key="os.environ/AWS_BEDROCK_TEST_SECRET_ACCESS_KEY",
        aws_region_name="us-east-1",
    )


def _cached_messages(marker: str) -> list[MessageParam]:
    return [
        cast(
            MessageParam,
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{marker} {CACHED_DOCUMENT}",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": "What are the payment terms?"},
                ],
            },
        )
    ]


def _message_text(message: Message) -> str:
    return "".join(block.text for block in message.content if isinstance(block, TextBlock))


@pytest.mark.parametrize("backend", BEDROCK_BACKENDS)
@pytest.mark.covers(
    "llm.messages.bedrock_converse.prompt_cache_5m.nonstream.works",
    "llm.messages.bedrock_invoke.prompt_cache_5m.nonstream.works",
)
@meta(
    Subject(
        domain=Domain.LLM_TRANSLATION,
        route=Route.MESSAGES,
        providers=(Provider.BEDROCK,),
        models=(BEDROCK_MODEL_ID,),
        mode=Mode.NONSTREAM,
    )
)
def test_bedrock_messages_prompt_caching_creates_cache(
    proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients, backend: str
) -> None:
    model, key = _register_deployment(proxy, resources, _deployment_params(backend), "e2e-prompt-cache-create")
    client = sdk.anthropic(key)
    message = client.messages.create(
        model=model,
        max_tokens=100,
        messages=_cached_messages(unique_marker()),
        extra_body=NO_PROXY_CACHE,
    )

    assert message.role == "assistant", f"unexpected role: {message.role!r}"
    assert message.usage.cache_creation_input_tokens > 0 or message.usage.cache_read_input_tokens > 0, (
        f"prompt cache was not created or read: {message.usage!r}"
    )


@pytest.mark.parametrize("backend", BEDROCK_BACKENDS)
@pytest.mark.covers(
    "llm.messages.bedrock_converse.prompt_cache_5m.nonstream.works",
    "llm.messages.bedrock_invoke.prompt_cache_5m.nonstream.works",
)
@meta(
    Subject(
        domain=Domain.LLM_TRANSLATION,
        route=Route.MESSAGES,
        providers=(Provider.BEDROCK,),
        models=(BEDROCK_MODEL_ID,),
        mode=Mode.NONSTREAM,
    )
)
def test_bedrock_messages_prompt_caching_reads_cache_on_second_call(
    proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients, backend: str
) -> None:
    model, key = _register_deployment(proxy, resources, _deployment_params(backend), "e2e-prompt-cache-read")
    client = sdk.anthropic(key)
    messages: Final = _cached_messages(unique_marker())

    client.messages.create(model=model, max_tokens=100, messages=messages, extra_body=NO_PROXY_CACHE)
    message = client.messages.create(model=model, max_tokens=100, messages=messages, extra_body=NO_PROXY_CACHE)

    assert message.usage.cache_read_input_tokens > 0, f"cache read was not reported: {message.usage!r}"


@pytest.mark.parametrize("backend", BEDROCK_BACKENDS)
@pytest.mark.covers(
    "llm.messages.bedrock_converse.basic.stream.works",
    "llm.messages.bedrock_invoke.basic.stream.works",
)
@meta(
    Subject(
        domain=Domain.LLM_TRANSLATION,
        route=Route.MESSAGES,
        providers=(Provider.BEDROCK,),
        models=(BEDROCK_MODEL_ID,),
        mode=Mode.STREAM,
    )
)
def test_bedrock_messages_streaming_prompt_caching_reads_cache(
    proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients, backend: str
) -> None:
    model, key = _register_deployment(proxy, resources, _deployment_params(backend), "e2e-prompt-cache-stream")
    client = sdk.anthropic(key)
    messages: Final = _cached_messages(unique_marker())
    client.messages.create(model=model, max_tokens=100, messages=messages, extra_body=NO_PROXY_CACHE)

    with client.messages.stream(
        model=model,
        max_tokens=100,
        messages=messages,
        extra_body=NO_PROXY_CACHE,
    ) as stream:
        message: Final = stream.get_final_message()

    assert message.usage.cache_read_input_tokens > 0, f"streaming cache read was not reported: {message.usage!r}"
    assert _message_text(message).strip(), f"streaming response contained no text: {message.content!r}"


@pytest.mark.parametrize("backend", TOOL_SEARCH_BACKENDS)
@pytest.mark.covers(
    "llm.messages.anthropic.tool_search.nonstream.works",
    "llm.messages.bedrock_converse.tool_search.nonstream.works",
    "llm.messages.bedrock_invoke.tool_search.nonstream.works",
)
@meta(
    Subject(
        domain=Domain.LLM_TRANSLATION,
        route=Route.MESSAGES,
        providers=(Provider.ANTHROPIC, Provider.BEDROCK),
        models=(ANTHROPIC_MODEL, BEDROCK_MODEL_ID),
        mode=Mode.NONSTREAM,
    )
)
def test_anthropic_tool_search_returns_a_message(
    proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients, backend: str
) -> None:
    model, key = _register_deployment(proxy, resources, _deployment_params(backend), "e2e-tool-search")
    message = sdk.anthropic(key).messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": f"Find the weather tool for San Francisco. {unique_marker()}"}],
        tools=_tool_search_tools(),
        extra_headers={"anthropic-beta": "tool-search-tool-2025-10-19,advanced-tool-use-2025-11-20"},
        extra_body=NO_PROXY_CACHE,
    )

    assert message.role == "assistant", f"unexpected role: {message.role!r}"
    assert message.content, "tool search response contained no content"
    assert message.usage.input_tokens > 0, f"tool search response had no input usage: {message.usage!r}"


@meta(
    Subject(
        domain=Domain.LLM_TRANSLATION,
        route=Route.MESSAGES,
        providers=(Provider.ANTHROPIC, Provider.BEDROCK),
        models=(ANTHROPIC_MODEL, BEDROCK_MODEL_ID),
        mode=Mode.STREAM,
    )
)
@pytest.mark.parametrize("backend", TOOL_SEARCH_BACKENDS)
@pytest.mark.covers(
    "llm.messages.anthropic.tool_search.nonstream.works",
    "llm.messages.bedrock_converse.tool_search.nonstream.works",
    "llm.messages.bedrock_invoke.tool_search.nonstream.works",
)
def test_anthropic_tool_search_streaming_returns_a_message(
    proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients, backend: str
) -> None:
    model, key = _register_deployment(proxy, resources, _deployment_params(backend), "e2e-tool-search-stream")
    client = sdk.anthropic(key)

    with client.messages.stream(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": f"Find the weather tool for Tokyo. {unique_marker()}"}],
        tools=_tool_search_tools(),
        extra_headers={"anthropic-beta": "tool-search-tool-2025-10-19,advanced-tool-use-2025-11-20"},
        extra_body=NO_PROXY_CACHE,
    ) as stream:
        message: Final = stream.get_final_message()

    assert message.role == "assistant", f"unexpected role: {message.role!r}"
    assert message.content, "streaming tool search response contained no content"
    assert message.usage.input_tokens > 0, f"streaming tool search response had no input usage: {message.usage!r}"


@pytest.mark.covers("llm.messages.bedrock_invoke.basic.nonstream.works")
@meta(
    Subject(
        domain=Domain.LLM_TRANSLATION,
        route=Route.MESSAGES,
        providers=(Provider.BEDROCK,),
        models=(BEDROCK_MODEL_ID,),
        mode=Mode.NONSTREAM,
    )
)
def test_bedrock_invoke_messages_accepts_forwarded_headers(
    proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
) -> None:
    params: Final = _deployment_params(f"bedrock/invoke/{BEDROCK_MODEL_ID}")
    model, key = _register_deployment(proxy, resources, params, "e2e-bedrock-forwarded-headers")
    message = sdk.anthropic(key).messages.create(
        model=model,
        max_tokens=32,
        messages=[{"role": "user", "content": f"Reply with one word. {unique_marker()}"}],
        extra_headers={
            "x-forwarded-for": "10.11.232.194",
            "x-forwarded-port": "443",
            "x-forwarded-proto": "https",
            "x-app": "cli",
        },
        extra_body=NO_PROXY_CACHE,
    )

    assert message.role == "assistant", f"unexpected role: {message.role!r}"
    assert _message_text(message).strip(), f"forwarded-header response contained no text: {message.content!r}"
