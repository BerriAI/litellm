from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.container_handler import generic_container_handler
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager

FILE_NOT_FOUND_BODY = {
    "error": {
        "message": "File not found.",
        "type": "invalid_request_error",
        "param": None,
        "code": None,
    }
}


def _sync_client(response: httpx.Response) -> HTTPHandler:
    handler = HTTPHandler()
    handler.client = httpx.Client(transport=httpx.MockTransport(lambda _request: response))
    return handler


def _async_client(response: httpx.Response) -> AsyncHTTPHandler:
    handler = AsyncHTTPHandler()
    handler.client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: response))
    return handler


def _handle(endpoint_name: str, client, **overrides):
    return generic_container_handler.handle(
        endpoint_name=endpoint_name,
        container_provider_config=ProviderConfigManager.get_provider_container_config(
            provider=litellm.LlmProviders.OPENAI
        ),
        litellm_params=GenericLiteLLMParams(api_key="sk-test"),
        logging_obj=MagicMock(),
        client=client,
        container_id="cntr_real",
        file_id="cfile_nonexistent",
        **overrides,
    )


def test_binary_endpoint_raises_on_error_status():
    with pytest.raises(BaseLLMException) as exc_info:
        _handle(
            "retrieve_container_file_content",
            _sync_client(httpx.Response(404, json=FILE_NOT_FOUND_BODY)),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.message == "File not found."


@pytest.mark.asyncio
async def test_async_binary_endpoint_raises_on_error_status():
    with pytest.raises(BaseLLMException) as exc_info:
        await _handle(
            "aretrieve_container_file_content",
            _async_client(httpx.Response(404, json=FILE_NOT_FOUND_BODY)),
            _is_async=True,
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.message == "File not found."


def test_binary_endpoint_returns_raw_content_on_success():
    content = _handle(
        "retrieve_container_file_content",
        _sync_client(httpx.Response(200, content=b"\x00binary-payload")),
    )

    assert content == b"\x00binary-payload"


def test_error_status_with_non_json_body_surfaces_response_text():
    with pytest.raises(BaseLLMException) as exc_info:
        _handle(
            "retrieve_container_file_content",
            _sync_client(httpx.Response(502, content=b"<html>bad gateway</html>")),
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.message == "<html>bad gateway</html>"


def test_json_endpoint_still_raises_provider_error_message():
    with pytest.raises(BaseLLMException) as exc_info:
        _handle(
            "retrieve_container_file",
            _sync_client(httpx.Response(404, json=FILE_NOT_FOUND_BODY)),
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.message == "File not found."
