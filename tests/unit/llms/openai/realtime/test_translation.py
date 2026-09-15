import gzip
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import AsyncOpenAI

from litellm.llms.azure.realtime.http_transformation import AzureRealtimeHTTPConfig
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.llms.openai.realtime.http_transformation import OpenAIRealtimeHTTPConfig
from litellm.types.realtime import RealtimeSessionConfig


def test_realtime_session_config_supports_translation_and_live_transcription_fields():
    session = RealtimeSessionConfig(
        type="translation",
        model="gpt-realtime-translate",
        audio={
            "input": {
                "transcription": {
                    "model": "gpt-live-transcribe",
                    "delay": "minimal",
                    "languages": ["en", "fr"],
                    "keywords": ["LiteLLM"],
                }
            },
            "output": {"language": "es"},
        },
    )

    assert session.audio is not None
    assert session.audio.input is not None
    assert session.audio.input.transcription is not None
    assert session.audio.input.transcription.delay == "minimal"
    assert session.audio.input.transcription.languages == ["en", "fr"]
    assert session.audio.input.transcription.keywords == ["LiteLLM"]
    assert session.audio.output is not None
    assert session.audio.output.language == "es"


@pytest.mark.parametrize(
    "api_base,expected",
    [
        (
            "https://api.openai.com",
            "https://api.openai.com/v1/realtime/translations/client_secrets",
        ),
        (
            "https://api.openai.com/v1",
            "https://api.openai.com/v1/realtime/translations/client_secrets",
        ),
    ],
)
def test_openai_translation_client_secret_url(api_base: str, expected: str):
    config = OpenAIRealtimeHTTPConfig()
    assert config.get_translation_client_secret_url(api_base, "gpt-realtime-translate") == expected


def test_openai_translation_calls_url():
    config = OpenAIRealtimeHTTPConfig()
    assert (
        config.get_translation_calls_url("https://api.openai.com/v1", "gpt-realtime-translate")
        == "https://api.openai.com/v1/realtime/translations/calls"
    )


def test_azure_translation_urls_use_ga_paths():
    config = AzureRealtimeHTTPConfig()
    assert (
        config.get_translation_client_secret_url("https://example.openai.azure.com", "translate-deployment")
        == "https://example.openai.azure.com/openai/v1/realtime/translations/client_secrets"
    )
    assert (
        config.get_translation_calls_url("https://example.openai.azure.com", "translate-deployment")
        == "https://example.openai.azure.com/openai/v1/realtime/translations/calls"
    )


@pytest.mark.asyncio
async def test_translation_client_secret_uses_custom_translation_path():
    client = MagicMock(spec=AsyncHTTPHandler)
    client.post = AsyncMock(
        return_value=httpx.Response(
            200,
            json={"value": "ek_test"},
            request=httpx.Request("POST", "https://api.openai.com/v1/realtime/translations/client_secrets"),
        )
    )
    logging_obj = MagicMock()
    handler = BaseLLMHTTPHandler()
    request_data = {"session": {"type": "translation", "model": "gpt-realtime-translate"}}

    response = await handler.async_realtime_translation_client_secret_handler(
        api_base="https://api.openai.com",
        api_key="sk-test",
        request_data=request_data,
        logging_obj=logging_obj,
        timeout=10,
        provider_config=OpenAIRealtimeHTTPConfig(),
        model="gpt-realtime-translate",
        client=client,
    )

    assert response.status_code == 200
    call = client.post.call_args.kwargs
    assert call["url"] == "https://api.openai.com/v1/realtime/translations/client_secrets"
    assert call["json"] == request_data


@pytest.mark.asyncio
async def test_azure_translation_client_secret_supports_entra_bearer_auth():
    client = MagicMock(spec=AsyncHTTPHandler)
    client.post = AsyncMock(
        return_value=httpx.Response(
            200,
            json={"value": "ek_test"},
            request=httpx.Request(
                "POST",
                "https://example.openai.azure.com/openai/v1/realtime/translations/client_secrets",
            ),
        )
    )
    handler = BaseLLMHTTPHandler()

    await handler.async_realtime_translation_client_secret_handler(
        api_base="https://example.openai.azure.com",
        api_key="",
        request_data={"session": {"type": "translation", "model": "translate-deployment"}},
        logging_obj=MagicMock(),
        timeout=10,
        provider_config=AzureRealtimeHTTPConfig(),
        model="translate-deployment",
        extra_headers={"Authorization": "Bearer entra-token"},
        client=client,
    )

    headers = client.post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer entra-token"
    assert "api-key" not in headers


@pytest.mark.asyncio
async def test_translation_calls_use_translation_session_and_path():
    client = MagicMock(spec=AsyncHTTPHandler)
    client.post = AsyncMock(
        return_value=httpx.Response(
            201,
            content=b"v=0\r\n",
            request=httpx.Request("POST", "https://api.openai.com/v1/realtime/translations/calls"),
        )
    )
    logging_obj = MagicMock()
    handler = BaseLLMHTTPHandler()

    response = await handler.async_realtime_calls_handler(
        api_base="https://api.openai.com",
        openai_ephemeral_key="ek_test",
        sdp_body=b"v=0\r\n",
        logging_obj=logging_obj,
        timeout=10,
        provider_config=OpenAIRealtimeHTTPConfig(),
        model="gpt-realtime-translate",
        client=client,
        translation=True,
    )

    assert response.status_code == 201
    call = client.post.call_args.kwargs
    assert call["url"] == "https://api.openai.com/v1/realtime/translations/calls"
    assert call["headers"]["Content-Type"] == "application/sdp"
    assert call["content"] == "v=0\r\n"


@pytest.mark.asyncio
async def test_standard_client_secret_uses_openai_sdk_resource():
    async def send_response(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/realtime/client_secrets"
        return httpx.Response(
            200,
            content=gzip.compress(b'{"value":"ek_test"}'),
            headers={"content-encoding": "gzip"},
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(send_response))
    openai_client = AsyncOpenAI(api_key="sk-test", base_url="https://example.com/v1", http_client=http_client)
    handler = BaseLLMHTTPHandler()
    logging_obj = MagicMock()

    response = await handler.async_realtime_client_secret_handler(
        api_base="https://example.com",
        api_key="sk-test",
        request_data={"session": {"type": "realtime", "model": "gpt-realtime-2.1"}},
        logging_obj=logging_obj,
        timeout=10,
        client=openai_client,
        use_openai_sdk=True,
    )
    await openai_client.close()

    assert response.status_code == 200
    assert response.json() == {"value": "ek_test"}
    assert "content-encoding" not in response.headers


@pytest.mark.asyncio
async def test_translation_client_secret_uses_openai_sdk_custom_post():
    async def send_response(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/realtime/translations/client_secrets"
        assert json.loads(request.content)["session"]["audio"]["output"]["language"] == "es"
        return httpx.Response(200, json={"value": "ek_translation"})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(send_response))
    openai_client = AsyncOpenAI(api_key="sk-test", base_url="https://example.com/v1", http_client=http_client)
    handler = BaseLLMHTTPHandler()

    response = await handler.async_realtime_translation_client_secret_handler(
        api_base="https://example.com",
        api_key="sk-test",
        request_data={
            "session": {
                "model": "gpt-realtime-translate",
                "audio": {"output": {"language": "es"}},
            }
        },
        logging_obj=MagicMock(),
        timeout=10,
        client=openai_client,
        use_openai_sdk=True,
    )
    await openai_client.close()

    assert response.status_code == 200
    assert response.json() == {"value": "ek_translation"}


@pytest.mark.asyncio
async def test_translation_calls_use_openai_sdk_custom_post():
    async def send_response(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/realtime/translations/calls"
        body = await request.aread()
        assert request.headers["content-type"] == "application/sdp"
        assert body == b"v=0\r\n"
        return httpx.Response(201, content=b"v=0\r\n")

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(send_response))
    openai_client = AsyncOpenAI(api_key="ek_test", base_url="https://example.com/v1", http_client=http_client)
    handler = BaseLLMHTTPHandler()

    response = await handler.async_realtime_calls_handler(
        api_base="https://example.com",
        openai_ephemeral_key="ek_test",
        sdp_body=b"v=0\r\n",
        logging_obj=MagicMock(),
        timeout=10,
        model="gpt-realtime-translate",
        client=openai_client,
        translation=True,
        use_openai_sdk=True,
    )
    await openai_client.close()

    assert response.status_code == 201
    assert response.text == "v=0\r\n"
