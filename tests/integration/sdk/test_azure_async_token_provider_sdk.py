import asyncio
import json
from typing import Final

import pytest
from integration._support.wire import Reply, Request, wire_server

import litellm

_TOKEN: Final = "async-provider-token"
_COMPLETION: Final = json.dumps(
    {
        "id": "chatcmpl-async-provider",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
).encode()


def _azure(_: Request) -> Reply:
    return Reply(body=_COMPLETION)


async def _async_token_provider() -> str:
    await asyncio.sleep(0)
    return _TOKEN


@pytest.mark.parametrize("api_version", ["2024-10-21", "v1"])
def test_an_async_azure_ad_token_provider_authorizes_the_upstream_call(
    api_version: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("AZURE_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_AD_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    with wire_server(_azure) as azure:
        response: Final = asyncio.run(
            litellm.acompletion(
                model="azure/gpt-4o",
                api_base=azure.url,
                api_version=api_version,
                azure_ad_token_provider=_async_token_provider,
                messages=[{"role": "user", "content": "hi"}],
                max_retries=0,
            )
        )
        authorizations: Final = tuple(request.headers.get("authorization") for request in azure.drain())

    assert response.choices[0].message.content == "ok"
    assert authorizations == (f"Bearer {_TOKEN}",)
