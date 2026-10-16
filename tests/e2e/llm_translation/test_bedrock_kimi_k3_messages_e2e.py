from __future__ import annotations

from typing import Final

import pytest
from anthropic.types import Message, MessageParam
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import SdkClients

pytestmark = pytest.mark.e2e

BEDROCK_MODEL: Final = "bedrock/us.moonshotai.kimi-k3"
MODEL_NAME: Final = "claude-kimi-k3"
MODEL_SWITCH_BETA: Final = (
    "claude-code-20250219,context-1m-2025-08-07,interleaved-thinking-2025-05-14,"
    "redact-thinking-2026-02-12,context-management-2025-06-27,"
    "prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07"
)


def _register_model(proxy: ProxyClient, resources: ResourceManager) -> tuple[str, str]:
    model_id: Final = proxy.create_model(
        MODEL_NAME,
        LiteLLMParamsBody(
            model=BEDROCK_MODEL,
            aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
            aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
            aws_region_name="os.environ/AWS_REGION",
            drop_params=True,
        ),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return MODEL_NAME, resources.key(models=[MODEL_NAME])


class TestBedrockKimiK3Messages:
    @pytest.mark.covers("llm.messages.bedrock_converse.basic.nonstream.works_for_kimi_k3_max_tokens_1")
    def test_model_switch_probe_max_tokens_1_returns_message(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register_model(proxy, resources)
        client: Final = sdk.anthropic(key)
        message_param: Final[MessageParam] = {
            "role": "user",
            "content": [{"type": "text", "text": "Hi", "cache_control": {"type": "ephemeral"}}],
        }
        message: Final = client.messages.create(
            model=model,
            max_tokens=1,
            stream=False,
            system="You are Claude Code, an assistant for software development.",
            messages=[message_param],
            extra_headers={"anthropic-beta": MODEL_SWITCH_BETA},
            extra_query={"beta": "true"},
        )

        assert isinstance(message, Message)
        assert message.role == "assistant"
