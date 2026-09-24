from __future__ import annotations

from typing import cast

import anthropic
import pytest
from anthropic.types import Message, MessageParam, TextBlock, ThinkingConfigParam
from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import NO_PROXY_CACHE, SdkClients

pytestmark = pytest.mark.e2e

BEDROCK_INVOKE_BACKEND = "bedrock/invoke/us.anthropic.claude-sonnet-5"
TOOL_ADDITION_BLOCK = {"type": "tool_addition", "tool_reference": {"type": "tool_reference", "tool_name": "Read"}}


def _register(proxy: ProxyClient, resources: ResourceManager) -> tuple[str, str]:
    model = f"e2e-bedrock-msgs-ext-{unique_marker()}"
    model_id = proxy.create_model(
        model,
        LiteLLMParamsBody(
            model=BEDROCK_INVOKE_BACKEND,
            aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
            aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
            aws_region_name="us-east-1",
        ),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model, resources.key()


def _output_config_messages() -> list[MessageParam]:
    return [
        {"role": "user", "content": "read the file /tmp/a.txt"},
        cast(
            MessageParam,
            {
                "role": "assistant",
                "output_config": {"effort": "high"},
                "content": [
                    {"type": "text", "text": "Reading it now."},
                    {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "/tmp/a.txt"}},
                ],
            },
        ),
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "hello world"}],
        },
    ]


def _tool_addition_messages() -> list[MessageParam]:
    return [
        {"role": "user", "content": "read /tmp/a.txt"},
        cast(MessageParam, {"role": "assistant", "content": [TOOL_ADDITION_BLOCK, {"type": "text", "text": "ok"}]}),
        {"role": "user", "content": "continue"},
    ]


def _only_tool_addition_messages() -> list[MessageParam]:
    return [
        {"role": "user", "content": "read /tmp/a.txt"},
        cast(MessageParam, {"role": "assistant", "content": [TOOL_ADDITION_BLOCK, TOOL_ADDITION_BLOCK]}),
        {"role": "user", "content": "continue"},
    ]


def _text(message: Message) -> str:
    return "".join(block.text for block in message.content if isinstance(block, TextBlock))


def _assert_answered(message: Message) -> None:
    assert message.role == "assistant", f"unexpected role: {message.role!r}"
    assert _text(message).strip(), f"/v1/messages returned no text: {message.content!r}"


class TestBedrockMessagesNativeExtensions:
    @pytest.mark.covers("llm.messages.bedrock_invoke.native_extensions.nonstream.works")
    def test_nested_output_config_is_stripped_before_bedrock_invoke(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources)
        message = sdk.anthropic(key).messages.create(
            model=model, max_tokens=300, messages=_output_config_messages(), extra_body=NO_PROXY_CACHE
        )
        _assert_answered(message)

    @pytest.mark.covers("llm.messages.bedrock_invoke.native_extensions.nonstream.works")
    def test_tool_addition_block_is_stripped_before_bedrock_invoke(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources)
        message = sdk.anthropic(key).messages.create(
            model=model, max_tokens=300, messages=_tool_addition_messages(), extra_body=NO_PROXY_CACHE
        )
        _assert_answered(message)

    @pytest.mark.covers("llm.messages.bedrock_invoke.native_extensions.nonstream.works")
    def test_thinking_display_updates_is_mapped_before_bedrock_invoke(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources)
        message = sdk.anthropic(key).messages.create(
            model=model,
            max_tokens=300,
            thinking=cast(ThinkingConfigParam, {"type": "adaptive", "display": "updates"}),
            messages=[{"role": "user", "content": "what is 2+2? think briefly"}],
            extra_body=NO_PROXY_CACHE,
        )
        _assert_answered(message)

    @pytest.mark.covers("llm.messages.bedrock_invoke.native_extensions.nonstream.works")
    def test_message_emptied_by_stripping_is_rejected_naming_the_message(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources)
        with pytest.raises(anthropic.BadRequestError) as exc_info:
            sdk.anthropic(key).messages.create(
                model=model, max_tokens=300, messages=_only_tool_addition_messages(), extra_body=NO_PROXY_CACHE
            )
        assert "messages[1]" in str(exc_info.value), str(exc_info.value)
