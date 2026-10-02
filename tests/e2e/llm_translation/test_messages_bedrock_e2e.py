from __future__ import annotations

from typing import Final

import pytest
from anthropic.types import RawContentBlockDeltaEvent, RawMessageDeltaEvent, TextBlock, TextDelta
from e2e_config import unique_marker
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import NO_PROXY_CACHE, SdkClients
from structured_output import SENTIMENT_OUTPUT_FORMAT, SENTIMENT_PROMPT, assert_sentiment_json

pytestmark = pytest.mark.e2e

CONVERSE_CLAUDE_BACKEND: Final = "bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0"
NOVA_BACKEND: Final = "bedrock/us.amazon.nova-2-lite-v1:0"


def _register(proxy: ProxyClient, resources: ResourceManager, backend: str) -> str:
    model = f"e2e-messages-bedrock-{unique_marker()}"
    model_id = proxy.create_model(
        model,
        LiteLLMParamsBody(
            model=backend,
            aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
            aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
            aws_region_name="os.environ/AWS_REGION",
        ),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model


class TestBedrockMessages:
    def test_converse_output_format_returns_schema_json_text(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register(proxy, resources, CONVERSE_CLAUDE_BACKEND)
        client = sdk.anthropic(resources.key())

        message = client.messages.create(
            model=model,
            max_tokens=128,
            messages=[{"role": "user", "content": SENTIMENT_PROMPT}],
            extra_body={**NO_PROXY_CACHE, "output_format": SENTIMENT_OUTPUT_FORMAT},
        )
        texts = tuple(block.text for block in message.content if isinstance(block, TextBlock))
        assert len(texts) == len(message.content), f"structured output came back as non-text blocks: {message!r}"
        assert_sentiment_json("".join(texts))

    @pytest.mark.covers("llm.messages.bedrock_converse.basic.stream.works")
    def test_nova_stream_relays_text_usage_and_stop(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model = _register(proxy, resources, NOVA_BACKEND)
        client = sdk.anthropic(resources.key())

        events = tuple(
            client.messages.create(
                model=model,
                max_tokens=64,
                stream=True,
                messages=[{"role": "user", "content": "Say hello in one short sentence."}],
                extra_body=NO_PROXY_CACHE,
            )
        )
        types = tuple(event.type for event in events)
        text = "".join(
            event.delta.text
            for event in events
            if isinstance(event, RawContentBlockDeltaEvent) and isinstance(event.delta, TextDelta)
        )
        assert text.strip(), f"streamed Nova reply carried no text: {types}"
        assert types[0] == "message_start" and types[-1] == "message_stop", (
            f"stream must open with message_start and end with message_stop: {types}"
        )
        deltas = tuple(event for event in events if isinstance(event, RawMessageDeltaEvent))
        assert len(deltas) == 1, f"expected exactly one message_delta: {types}"
        assert deltas[0].delta.stop_reason is not None, f"message_delta carried no stop_reason: {deltas[0]!r}"
        assert deltas[0].usage.output_tokens > 0, f"message_delta reported no output tokens: {deltas[0]!r}"
