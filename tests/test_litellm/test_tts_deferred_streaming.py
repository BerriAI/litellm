import asyncio

from types import SimpleNamespace

from litellm.llms.openai.openai import _DeferredOpenAITTSStream


class _FakeHTTPResponse:
    def __init__(self, chunks):
        self._chunks = chunks

    async def aiter_bytes(self, chunk_size: int = 1024):
        for c in self._chunks:
            await asyncio.sleep(0)
            yield c


class _FakeStreamed:
    def __init__(self, chunks, enter_counter):
        self.http_response = _FakeHTTPResponse(chunks)
        self._enter_counter = enter_counter

    async def __aenter__(self):
        self._enter_counter["count"] += 1
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeContextFactory:
    def __init__(self, chunks, enter_counter):
        self._chunks = chunks
        self._enter_counter = enter_counter

    def __call__(self, **kwargs):
        # Return an async context manager compatible object
        return _FakeStreamed(self._chunks, self._enter_counter)


def _make_fake_client(chunks, enter_counter):
    client = SimpleNamespace()
    client.audio = SimpleNamespace()
    client.audio.speech = SimpleNamespace()
    client.audio.speech.with_streaming_response = SimpleNamespace()
    # create(**kwargs) should return an async context manager
    client.audio.speech.with_streaming_response.create = _FakeContextFactory(chunks, enter_counter)
    return client


def test_deferred_streaming_yields_bytes():
    chunks = [b"one", b"two", b"three"]
    enter_counter = {"count": 0}
    client = _make_fake_client(chunks, enter_counter)
    stream = _DeferredOpenAITTSStream(
        client=client,
        request_kwargs={"model": "x", "voice": "y", "input": "z"},
    )

    # Ensure stream context not opened until iteration
    assert enter_counter["count"] == 0

    async def _collect():
        out_local = []
        async for b in stream.aiter_bytes(chunk_size=2):
            out_local.append(b)
        return out_local

    out = asyncio.run(_collect())
    assert out == chunks
    # Ensure context was opened exactly once during iteration
    assert enter_counter["count"] == 1

