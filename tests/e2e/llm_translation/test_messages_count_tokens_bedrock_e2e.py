"""Live e2e: `/v1/messages/count_tokens` on a Bedrock Claude model that bedrock-runtime
cannot count.

Claude Opus 4.8 is offered only through cross-region inference, and bedrock-runtime's
CountTokens answers 400 for it. The proxy then has to count through bedrock-mantle's
Anthropic count_tokens, and the answer must sit within a few percent of what `/v1/messages`
bills as `usage.input_tokens`. The local tokenizer fallback undercounts these models by
about 40%, so this is the line that proves the real count is served
"""

from __future__ import annotations

from typing import Final

import pytest
from e2e_config import unique_marker
from e2e_http import unwrap
from e2e_metadata import Domain, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import AnthropicMessagesBody, ChatMessage, CountTokensBody, LiteLLMParamsBody
from proxy_client import ProxyClient

pytestmark = pytest.mark.e2e

CROSS_REGION_ONLY_CLAUDE_BACKEND: Final = "bedrock/global.anthropic.claude-opus-4-8"
COUNT_TOLERANCE: Final = 0.05


def _register(proxy: ProxyClient, resources: ResourceManager, backend: str) -> str:
    model = f"e2e-count-tokens-bedrock-{unique_marker()}"
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


class TestBedrockMessagesCountTokens:
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.COUNT_TOKENS,
            providers=(Provider.BEDROCK,),
            models=(CROSS_REGION_ONLY_CLAUDE_BACKEND,),
        )
    )
    def test_count_matches_billed_input_tokens_for_a_model_bedrock_runtime_cannot_count(
        self, proxy: ProxyClient, scoped_key: str, resources: ResourceManager
    ) -> None:
        model = _register(proxy, resources, CROSS_REGION_ONLY_CLAUDE_BACKEND)
        prompt = f"{unique_marker()} " + "The quick brown fox jumps over the lazy dog. " * 40
        message = ChatMessage(role="user", content=prompt)

        counted = unwrap(proxy.count_tokens(scoped_key, CountTokensBody(model=model, messages=[message])))
        answered = unwrap(
            proxy.messages(scoped_key, AnthropicMessagesBody(model=model, messages=[message], max_tokens=1))
        )

        assert answered.usage is not None and answered.usage.input_tokens, answered
        billed = answered.usage.input_tokens
        assert abs(counted.input_tokens - billed) <= billed * COUNT_TOLERANCE, (counted, answered.usage)
