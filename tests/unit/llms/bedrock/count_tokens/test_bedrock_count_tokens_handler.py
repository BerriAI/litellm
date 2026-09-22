import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest
from botocore.credentials import RefreshableCredentials

from litellm.llms.bedrock.count_tokens.handler import BedrockCountTokensHandler
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from tests.test_litellm.llms.bedrock.event_loop_probe import EventLoopProbe


class _ProbedCountTokensHandler(BedrockCountTokensHandler):
    def __init__(self, probe: EventLoopProbe) -> None:
        super().__init__()
        self._probe = probe

    def get_credentials(
        self,
        **kwargs: object,  # kwargs-ok: mirrors the base resolver's keyword contract, which the probe ignores
    ) -> RefreshableCredentials:
        return self._probe.credentials()


@pytest.mark.asyncio
async def test_handle_count_tokens_request_signs_off_the_event_loop(monkeypatch):
    """Regression for issue #40165: the count_tokens handler signed on the loop, so botocore's blocking
    credential refresh inside SigV4 stalled every other request on the worker."""
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    probe = EventLoopProbe()
    client = AsyncMock(spec=AsyncHTTPHandler)
    client.post = AsyncMock(
        return_value=httpx.Response(
            200,
            json={"inputTokens": 7},
            request=httpx.Request("POST", "https://bedrock-runtime.us-west-2.amazonaws.com/"),
        )
    )
    release = asyncio.create_task(probe.release_refresh_from_the_loop())

    result = await _ProbedCountTokensHandler(probe).handle_count_tokens_request(
        request_data={
            "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "messages": [{"role": "user", "content": "hi"}],
        },
        litellm_params={"aws_region_name": "us-west-2"},
        resolved_model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        client=client,
    )
    await release

    assert result == {"input_tokens": 7}
    assert client.post.call_args.kwargs["headers"]["Authorization"].startswith("AWS4-HMAC-SHA256")
    assert probe.served_during_refresh is True
