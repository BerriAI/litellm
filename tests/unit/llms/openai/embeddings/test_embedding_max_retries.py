"""`max_retries=0` on an OpenAI embedding call must reach the client.

An embedding request that names zero retries is asking for a single attempt.
`litellm.embedding(..., max_retries=0)` and `litellm.aembedding(..., max_retries=0)`
used to fall back to the package default, so a failing request was silently
attempted again instead of failing on the first try.
"""

import asyncio
from typing import Final
from unittest.mock import patch

import httpx
import pytest

import litellm

EMBEDDING_MODEL: Final = "openai/text-embedding-3-small"
SYNC_CLIENT: Final = "litellm.llms.openai.openai.OpenAI"
ASYNC_CLIENT: Final = "litellm.llms.openai.openai.AsyncOpenAI"


class _ClientBuilt(Exception):
    """Ends the call once the OpenAI client is built, so no request is sent."""


def test_sync_embedding_keeps_explicit_zero_max_retries() -> None:
    with patch(SYNC_CLIENT, side_effect=_ClientBuilt) as mock_client:
        with pytest.raises(Exception):
            litellm.embedding(
                model=EMBEDDING_MODEL,
                input=["hello"],
                api_key="sk-test",
                api_base="http://127.0.0.1:9/v1/explicit-zero",
                max_retries=0,
            )

    assert mock_client.call_args.kwargs["max_retries"] == 0


def test_sync_embedding_default_max_retries_unchanged() -> None:
    with patch(SYNC_CLIENT, side_effect=_ClientBuilt) as mock_client:
        with pytest.raises(Exception):
            litellm.embedding(
                model=EMBEDDING_MODEL,
                input=["hello"],
                api_key="sk-test",
                api_base="http://127.0.0.1:9/v1/no-max-retries",
            )

    assert mock_client.call_args.kwargs["max_retries"] == litellm.DEFAULT_MAX_RETRIES


def test_async_embedding_keeps_explicit_zero_max_retries() -> None:
    with patch(ASYNC_CLIENT, side_effect=_ClientBuilt) as mock_client:
        with pytest.raises(Exception):
            asyncio.run(
                litellm.aembedding(
                    model=EMBEDDING_MODEL,
                    input=["hello"],
                    api_key="sk-test",
                    api_base="http://127.0.0.1:9/v1/async-explicit-zero",
                    max_retries=0,
                )
            )

    assert mock_client.call_args.kwargs["max_retries"] == 0


def test_embedding_with_zero_max_retries_makes_one_attempt() -> None:
    attempts: Final[list[httpx.Request]] = []

    def _fail_every_attempt(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(500, json={"error": {"message": "upstream failed", "type": "server_error"}})

    from openai import OpenAI

    def _client_with_failing_transport(*args: object, **kwargs: object) -> OpenAI:
        kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(_fail_every_attempt))
        return OpenAI(*args, **kwargs)

    with patch(SYNC_CLIENT, side_effect=_client_with_failing_transport):
        with pytest.raises(Exception):
            litellm.embedding(
                model=EMBEDDING_MODEL,
                input=["hello"],
                api_key="sk-test",
                api_base="http://127.0.0.1:9/v1/single-attempt",
                max_retries=0,
            )

    assert len(attempts) == 1
