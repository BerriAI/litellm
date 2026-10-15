import asyncio
from typing import Final, Literal

import pytest

import litellm
from litellm.caching.caching import Cache
from litellm.caching.caching_handler import (
    _PENDING_CACHE_WRITES,  # pyright: ignore[reportPrivateUsage]  # await the existing background cache writer before the next request
)
from litellm.types.utils import ModelResponse
from tests.test_litellm_rust.chat_completions.data import REQUEST, call
from tests.test_litellm_rust.support.caching import use_cache
from tests.test_litellm_rust.support.recording_server import RecordingServer
from tests.test_litellm_rust.support.results import cache_key
from tests.test_litellm_rust.support.routing import CALL_ID, assert_served_natively

pytestmark = pytest.mark.requires_rust_extension


def content(response: object) -> object:
    assert isinstance(response, ModelResponse)
    return response.model_dump()["choices"][0]["message"]["content"]


async def call_python(server: RecordingServer, monkeypatch: pytest.MonkeyPatch) -> object:
    with monkeypatch.context() as patched:
        patched.setenv("LITELLM_RUST", "0")
        response: Final = await litellm.acompletion(
            **REQUEST, api_key="test-key", api_base=server.base_url, litellm_call_id=CALL_ID
        )
    await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_a_repeated_native_call_is_answered_from_the_python_cache(
    server: RecordingServer, asynchronous: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    use_cache(monkeypatch)

    first: Final = await call(server, asynchronous=asynchronous)
    second: Final = await call(server, asynchronous=asynchronous)

    assert_served_natively(server, first)
    assert cache_key(first) is None
    assert isinstance(cache_key(second), str)
    assert content(second) == content(first)


@pytest.mark.asyncio
@pytest.mark.parametrize("writer", ("python", "native"))
async def test_native_and_python_calls_share_python_cache_entries(
    server: RecordingServer, monkeypatch: pytest.MonkeyPatch, writer: Literal["python", "native"]
) -> None:
    use_cache(monkeypatch)

    first: Final = await (call_python(server, monkeypatch) if writer == "python" else call(server))
    second: Final = await (call(server) if writer == "python" else call_python(server, monkeypatch))

    assert len(server.requests) == 1
    assert content(second) == content(first)
