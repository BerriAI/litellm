from __future__ import annotations

from typing import cast

import anthropic
import pytest
from anthropic.types import Message, MessageParam, TextBlock, ThinkingConfigParam, ToolParam
from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import NO_PROXY_CACHE, SdkClients

pytestmark = pytest.mark.e2e

EVERYTHING_BACKEND = "bedrock/global.anthropic.claude-fable-5-1"
DISPLAY_ONLY_BACKEND = "bedrock/us.anthropic.claude-sonnet-5"
UPDATES = cast(ThinkingConfigParam, {"type": "adaptive", "display": "updates"})
DEFERRED_TOOL: ToolParam = {
    "name": "mcp__linear__list_issues",
    "description": "List Linear issues",
    "defer_loading": True,
    "input_schema": {"type": "object", "properties": {}},
}


READ_TOOL: ToolParam = {
    "name": "Read",
    "description": "Read a file",
    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
}


def _register(proxy: ProxyClient, resources: ResourceManager, backend: str, *, drop_params: bool) -> tuple[str, str]:
    model = f"e2e-bedrock-msgs-ext-{unique_marker()}"
    model_id = proxy.create_model(
        model,
        LiteLLMParamsBody(
            model=backend,
            aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
            aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
            aws_region_name="us-east-1",
            drop_params=drop_params,
        ),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model, resources.key()


def _effort_messages() -> list[MessageParam]:
    return [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi! What can I do for you?"},
        cast(MessageParam, {"role": "system", "content": [], "output_config": {"effort": "low"}}),
        {"role": "user", "content": "Say hi in five words."},
    ]


def _tool_addition_messages() -> list[MessageParam]:
    return [
        {"role": "user", "content": "Which tools do you have for Linear? Answer in one sentence."},
        cast(
            MessageParam,
            {
                "role": "system",
                "content": [
                    {"type": "tool_addition", "tool": {"type": "tool_reference", "name": "mcp__linear__list_issues"}}
                ],
            },
        ),
    ]


def _text(message: Message) -> str:
    return "".join(block.text for block in message.content if isinstance(block, TextBlock))


def _assert_actionable_error(error: anthropic.BadRequestError, *, path: str, knob: str) -> None:
    assert path in str(error), str(error)
    assert knob in str(error), str(error)


class TestBedrockMessagesNativeExtensions:
    @pytest.mark.covers("llm.messages.bedrock_invoke.native_extensions.nonstream.works")
    def test_supported_model_answers_with_every_extension_forwarded(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources, EVERYTHING_BACKEND, drop_params=False)
        client = sdk.anthropic(key)

        message = client.messages.create(
            model=model,
            max_tokens=300,
            thinking=UPDATES,
            tools=[READ_TOOL, DEFERRED_TOOL],
            messages=[*_effort_messages(), *_tool_addition_messages()[1:]],
            extra_body=NO_PROXY_CACHE,
        )
        assert message.role == "assistant", f"unexpected role: {message.role!r}"
        assert _text(message).strip(), f"/v1/messages returned no text: {message.content!r}"

    @pytest.mark.covers("llm.messages.bedrock_invoke.native_extensions.nonstream.works")
    def test_thinking_display_updates_is_forwarded_where_supported(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources, DISPLAY_ONLY_BACKEND, drop_params=False)
        client = sdk.anthropic(key)

        message = client.messages.create(
            model=model,
            max_tokens=300,
            thinking=UPDATES,
            messages=[{"role": "user", "content": "what is 2+2? think briefly"}],
            extra_body=NO_PROXY_CACHE,
        )
        assert message.role == "assistant", f"unexpected role: {message.role!r}"
        assert _text(message).strip(), f"/v1/messages returned no text: {message.content!r}"

    @pytest.mark.covers("llm.messages.bedrock_invoke.native_extensions.nonstream.works")
    def test_unsupported_per_message_effort_rejected_with_actionable_error(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources, DISPLAY_ONLY_BACKEND, drop_params=False)
        client = sdk.anthropic(key)

        with pytest.raises(anthropic.BadRequestError) as exc_info:
            client.messages.create(model=model, max_tokens=300, messages=_effort_messages(), extra_body=NO_PROXY_CACHE)
        _assert_actionable_error(exc_info.value, path="messages[2].output_config", knob="drop_params")

    @pytest.mark.covers("llm.messages.bedrock_invoke.native_extensions.nonstream.works")
    def test_drop_params_drops_unsupported_per_message_effort(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources, DISPLAY_ONLY_BACKEND, drop_params=True)
        client = sdk.anthropic(key)

        message = client.messages.create(
            model=model, max_tokens=300, messages=_effort_messages(), extra_body=NO_PROXY_CACHE
        )
        assert message.role == "assistant", f"unexpected role: {message.role!r}"
        assert _text(message).strip(), f"/v1/messages returned no text: {message.content!r}"

    @pytest.mark.covers("llm.messages.bedrock_invoke.native_extensions.nonstream.works")
    def test_unsupported_tool_addition_rejected_with_actionable_error(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources, DISPLAY_ONLY_BACKEND, drop_params=False)
        client = sdk.anthropic(key)

        with pytest.raises(anthropic.BadRequestError) as exc_info:
            client.messages.create(
                model=model,
                max_tokens=300,
                tools=[READ_TOOL, DEFERRED_TOOL],
                messages=_tool_addition_messages(),
                extra_body=NO_PROXY_CACHE,
            )
        _assert_actionable_error(
            exc_info.value, path="messages[1].content[0] (type 'tool_addition')", knob="modify_params"
        )

    @pytest.mark.covers("llm.messages.bedrock_invoke.native_extensions.nonstream.works")
    def test_drop_params_alone_does_not_remove_tool_addition_block(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources, DISPLAY_ONLY_BACKEND, drop_params=True)
        client = sdk.anthropic(key)

        with pytest.raises(anthropic.BadRequestError) as exc_info:
            client.messages.create(
                model=model,
                max_tokens=300,
                tools=[READ_TOOL, DEFERRED_TOOL],
                messages=_tool_addition_messages(),
                extra_body=NO_PROXY_CACHE,
            )
        _assert_actionable_error(
            exc_info.value, path="messages[1].content[0] (type 'tool_addition')", knob="modify_params"
        )
