import json
import ssl
from typing import Final
from unittest.mock import patch

import certifi
import httpx
import pytest

import litellm


@pytest.mark.asyncio
@pytest.mark.parametrize("verify", [False, certifi.where(), None], ids=["disabled", "custom-ca", "default"])
async def test_ssl_verify_http_client(verify: bool | str | None):
    def respond(request: httpx.Request) -> httpx.Response:
        payload: Final = json.loads(request.content)
        assert "ssl_verify" not in payload
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-3.5-turbo",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
                ],
            },
        )

    transport: Final = httpx.MockTransport(respond)
    with patch("litellm.llms.openai.common_utils.AsyncHTTPHandler") as handler:
        handler._create_async_transport.return_value = transport
        handler._create_httpx_proxy_mounts.return_value = {}
        response: Final = await litellm.acompletion(
            model="openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            ssl_verify=verify,
            api_key=f"sk-test-{verify}",
            max_retries=0,
        )
        assert response.choices[0].message.content == "hello"
        transport_kwargs: Final = handler._create_async_transport.call_args.kwargs
        if verify is False:
            assert transport_kwargs["ssl_verify"] is False
            assert transport_kwargs["ssl_context"] is None
        else:
            context: Final = transport_kwargs["ssl_context"]
            assert isinstance(context, ssl.SSLContext)
            assert context.verify_mode == ssl.CERT_REQUIRED
            assert context.check_hostname
