from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.llms.bedrock.count_tokens.bedrock_token_counter import BedrockTokenCounter

COUNT_TOKENS_URL_IN_THE_PREFIX_REGION: Final = (
    "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-haiku-4-5-20251001-v1%3A0/count-tokens"
)


@pytest.mark.asyncio
async def test_count_tokens_calls_the_region_named_in_the_model_prefix_with_the_bare_model_id(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("AWS_REGION_NAME", "us-west-2")
    monkeypatch.delenv("AWS_BEDROCK_RUNTIME_ENDPOINT", raising=False)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    with respx.mock:
        respx.post(COUNT_TOKENS_URL_IN_THE_PREFIX_REGION).mock(
            return_value=httpx.Response(200, json={"inputTokens": 7})
        )
        result = await BedrockTokenCounter().count_tokens(
            model_to_use="us-east-1/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            messages=[{"role": "user", "content": "hi"}],
            contents=None,
            deployment={"litellm_params": {"aws_access_key_id": "fake", "aws_secret_access_key": "fake"}},
        )

    assert result is not None
    assert result.total_tokens == 7
