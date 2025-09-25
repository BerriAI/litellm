import asyncio

import pytest

from litellm.llms.openai.openai import _DeferredOpenAITTSStream


class _FakeHTTPResponse:
    def __init__(self, chunks):
        self._chunks = chunks

    async def aiter_bytes(self, chunk_size: int = 1024):
        for c in self._chunks:
            await asyncio.sleep(0)
            yield c


class _FakeStreamed:
    def __init__(self, chunks):
        self.http_response = _FakeHTTPResponse(chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeContextFactory:
    def __init__(self, chunks):
        self._chunks = chunks

    def __call__(self, **kwargs):
        # Return an async context manager compatible object
        return _FakeStreamed(self._chunks)


class _FakeClientNS:
    pass


def _make_fake_client(chunks):
    client = _FakeClientNS()
    client.audio = _FakeClientNS()
    client.audio.speech = _FakeClientNS()
    client.audio.speech.with_streaming_response = _FakeClientNS()
    # create(**kwargs) should return an async context manager
    client.audio.speech.with_streaming_response.create = _FakeContextFactory(chunks)
    return client


@pytest.mark.asyncio
async def test_deferred_streaming_yields_bytes():
    chunks = [b"one", b"two", b"three"]
    client = _make_fake_client(chunks)
    stream = _DeferredOpenAITTSStream(client=client, request_kwargs={"model": "x", "voice": "y", "input": "z"})

    out = []
    async for b in stream.aiter_bytes(chunk_size=2):
        out.append(b)

    assert out == chunks

