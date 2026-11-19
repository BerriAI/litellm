from collections.abc import Callable
from typing import Final

import anthropic
import httpx
import openai
import pytest

from litellm.litellm_core_utils.clientside_credential_handler import carries_caller_credentials
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


@pytest.mark.parametrize("carrier", ["headers", "extra_headers", "default_headers"])
@pytest.mark.parametrize("header", ["Authorization", "x-api-key", "Api-Key"])
def test_a_credential_header_in_any_header_carrier_counts_as_a_caller_credential(carrier: str, header: str) -> None:
    assert carries_caller_credentials({carrier: {header: "caller-secret"}}) is True


@pytest.mark.parametrize("carrier", ["headers", "extra_headers", "default_headers"])
def test_a_plain_header_is_not_a_caller_credential(carrier: str) -> None:
    assert carries_caller_credentials({carrier: {"x-app": "my-app", "traceparent": "00-abc-def-01"}}) is False


def _handler_sending(headers: dict[str, str]) -> AsyncHTTPHandler:
    handler: Final = AsyncHTTPHandler()
    handler.client = httpx.AsyncClient(headers=headers)
    return handler


@pytest.mark.parametrize(
    "make_client",
    (
        lambda: openai.AsyncOpenAI(api_key="caller-secret"),
        lambda: openai.AsyncAzureOpenAI(
            api_key="caller-secret", api_version="2024-10-21", azure_endpoint="https://caller.openai.azure.com"
        ),
        lambda: anthropic.AsyncAnthropic(api_key="caller-secret"),
        lambda: _handler_sending({"x-api-key": "caller-secret"}),
    ),
    ids=("openai", "azure", "anthropic", "litellm_handler"),
)
def test_a_client_that_sends_a_key_counts_as_a_caller_credential(make_client: Callable[[], object]) -> None:
    assert carries_caller_credentials({"client": make_client()}) is True


@pytest.mark.parametrize(
    "make_client",
    (lambda: AsyncHTTPHandler(), lambda: _handler_sending({"x-app": "my-app"})),
    ids=("default_handler", "handler_with_plain_headers"),
)
def test_a_client_that_sends_no_key_is_not_a_caller_credential(make_client: Callable[[], object]) -> None:
    assert carries_caller_credentials({"client": make_client()}) is False
