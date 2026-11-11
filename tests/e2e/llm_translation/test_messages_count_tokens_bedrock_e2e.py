"""Live e2e: `/v1/messages/count_tokens` on a Bedrock Claude model that bedrock-runtime
cannot count.

Claude Opus 4.8 is offered only through cross-region inference, and bedrock-runtime's
CountTokens answers 400 for it. The proxy then has to count through bedrock-mantle's
Anthropic count_tokens, and the answer must sit within a few percent of what `/v1/messages`
bills as `usage.input_tokens`. The local tokenizer fallback undercounts these models by
about 40%, so this is the line that proves the real count is served. Both calls go through
the real Anthropic SDK, the client customers count with
"""

from __future__ import annotations

from typing import Final

import pytest
from anthropic.types import MessageParam
from e2e_config import unique_marker
from e2e_metadata import Domain, Provider, Route, Subject, meta
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import NO_PROXY_CACHE, SdkClients

pytestmark = pytest.mark.e2e

CROSS_REGION_ONLY_CLAUDE_BACKEND: Final = "bedrock/global.anthropic.claude-opus-4-8"
COUNT_TOLERANCE: Final = 0.05


def _register(proxy: ProxyClient, resources: ResourceManager, backend: str) -> tuple[str, str]:
    model: Final = f"e2e-count-tokens-bedrock-{unique_marker()}"
    model_id: Final = proxy.create_model(
        model,
        LiteLLMParamsBody(
            model=backend,
            aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
            aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
            aws_region_name="os.environ/AWS_REGION",
        ),
    )
    resources.defer(lambda: proxy.delete_model(model_id))
    return model, resources.key()


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
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(proxy, resources, CROSS_REGION_ONLY_CLAUDE_BACKEND)
        client: Final = sdk.anthropic(key)
        prompt: Final = f"{unique_marker()} " + "The quick brown fox jumps over the lazy dog. " * 40
        message: Final[MessageParam] = {"role": "user", "content": prompt}

        counted: Final = client.messages.count_tokens(model=model, messages=[message])
        answered: Final = client.messages.create(
            model=model, max_tokens=1, messages=[message], extra_body=NO_PROXY_CACHE
        )

        billed: Final = answered.usage.input_tokens
        assert billed, answered.usage
        assert abs(counted.input_tokens - billed) <= billed * COUNT_TOLERANCE, (counted, answered.usage)
