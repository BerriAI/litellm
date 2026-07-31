import logging
import os
from collections.abc import Iterator

import pytest

import litellm
from litellm._logging import verbose_logger

# Captured at import, before conftest's isolate_host_aws_config strips it.
BEARER_TOKEN = os.getenv("AWS_BEARER_TOKEN_BEDROCK")

pytestmark = pytest.mark.skipif(
    BEARER_TOKEN is None,
    reason="requires AWS_BEARER_TOKEN_BEDROCK for a live Bedrock call",
)


@pytest.fixture(autouse=True)
def restore_bedrock_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", BEARER_TOKEN or "")
    monkeypatch.setenv("AWS_REGION_NAME", "us-west-2")


@pytest.fixture(autouse=True)
def rust_debug_only() -> Iterator[None]:
    level = verbose_logger.level
    handlers = tuple(verbose_logger.handlers)
    propagate = verbose_logger.propagate
    verbose_logger.setLevel(logging.DEBUG)
    verbose_logger.handlers.clear()
    verbose_logger.propagate = False
    yield
    verbose_logger.setLevel(level)
    verbose_logger.handlers.extend(handlers)
    verbose_logger.propagate = propagate


@pytest.mark.asyncio
async def test_bedrock_messages_with_rust() -> None:
    response = await litellm.anthropic.messages.acreate(
        model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=20,
        api_key=BEARER_TOKEN,
        aws_region_name="us-west-2",
        rust=True,
    )

    assert response["content"][0]["text"]
    assert response["_hidden_params"]["additional_headers"]["x-litellm-rust"] == "true"


@pytest.mark.asyncio
async def test_bedrock_messages_streaming_with_rust() -> None:
    response = await litellm.anthropic.messages.acreate(
        model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=20,
        stream=True,
        api_key=BEARER_TOKEN,
        aws_region_name="us-west-2",
        rust=True,
    )

    chunks = [chunk async for chunk in response]

    assert chunks
    assert response._hidden_params["additional_headers"]["x-litellm-rust"] == "true"
