import asyncio
import json
from io import BytesIO
from pathlib import Path
from typing import Final

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.foundry_local.audio_transcription.transformation import FoundryLocalAudioTranscriptionConfig
from litellm.llms.openai.common_utils import OpenAIError
from litellm.types.utils import ModelResponse


@pytest.mark.parametrize(
    ("model", "custom_llm_provider"),
    [("foundry_local/local-model:1", None), ("local-model:1", "foundry_local")],
)
@pytest.mark.parametrize(
    ("api_base", "api_key", "env_base", "env_key", "expected_base", "expected_key"),
    [
        (None, None, "http://127.0.0.1:61234/v1", None, "http://127.0.0.1:61234/v1", "fake-api-key"),
        (None, None, "http://127.0.0.1:61234/v1", "env-key", "http://127.0.0.1:61234/v1", "env-key"),
        ("http://localhost:5272/v1", None, None, "env-key", "http://localhost:5272/v1", "env-key"),
        (None, "explicit-key", "http://127.0.0.1:61234/v1", None, "http://127.0.0.1:61234/v1", "explicit-key"),
        ("http://localhost:5272/v1", "explicit-key", None, None, "http://localhost:5272/v1", "explicit-key"),
        (
            "http://localhost:5272/v1",
            "explicit-key",
            "http://127.0.0.1:61234/v1",
            "env-key",
            "http://localhost:5272/v1",
            "explicit-key",
        ),
    ],
)
def test_should_resolve_foundry_endpoint_and_credentials(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    custom_llm_provider: str | None,
    api_base: str | None,
    api_key: str | None,
    env_base: str | None,
    env_key: str | None,
    expected_base: str | None,
    expected_key: str,
) -> None:
    monkeypatch.delenv("FOUNDRY_LOCAL_API_BASE", raising=False)
    monkeypatch.delenv("FOUNDRY_LOCAL_API_KEY", raising=False)
    if env_base is not None:
        monkeypatch.setenv("FOUNDRY_LOCAL_API_BASE", env_base)
    if env_key is not None:
        monkeypatch.setenv("FOUNDRY_LOCAL_API_KEY", env_key)

    assert get_llm_provider(model, custom_llm_provider=custom_llm_provider, api_base=api_base, api_key=api_key) == (
        "local-model:1",
        "foundry_local",
        expected_key,
        expected_base,
    )


def _respond(request: httpx.Request, stream: bool) -> httpx.Response:
    assert str(request.url) == "http://127.0.0.1:61234/v1/chat/completions"
    payload: Final = json.loads(request.content)
    assert payload["model"] == "local-model:1"
    assert payload["messages"] == [{"role": "user", "content": "Hello"}]
    assert payload["max_tokens"] == 16
    assert bool(payload.get("stream")) is stream
    if stream:
        chunk: Final = {
            "id": "chatcmpl-local",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "local-model:1",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": None}],
        }
        final_chunk: Final = {
            **chunk,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=f"data: {json.dumps(chunk)}\n\ndata: {json.dumps(final_chunk)}\n\ndata: [DONE]\n\n",
        )
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "created": 0,
            "model": "local-model:1",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        },
    )


@pytest.mark.parametrize("provider", [None, "foundry_local"])
def test_should_require_a_foundry_endpoint(monkeypatch: pytest.MonkeyPatch, provider: str | None) -> None:
    monkeypatch.delenv("FOUNDRY_LOCAL_API_BASE", raising=False)
    with pytest.raises(litellm.BadRequestError, match="requires api_base"):
        get_llm_provider(
            "local-model:1" if provider else "foundry_local/local-model:1",
            custom_llm_provider=provider,
        )


def test_should_preserve_global_api_base_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOUNDRY_LOCAL_API_BASE", raising=False)
    monkeypatch.delenv("FOUNDRY_LOCAL_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "api_base", "http://127.0.0.1:61234/v1")
    assert get_llm_provider("foundry_local/local-model:1") == (
        "local-model:1",
        "foundry_local",
        "fake-api-key",
        "http://127.0.0.1:61234/v1",
    )


@pytest.mark.parametrize("stream", [False, True])
def test_should_complete_chat_over_foundry_http_service(stream: bool) -> None:
    with (
        httpx.Client(transport=httpx.MockTransport(lambda request: _respond(request, stream))) as http_client,
        OpenAI(api_key="fake-api-key", base_url="http://127.0.0.1:61234/v1", http_client=http_client) as client,
    ):
        response: Final = litellm.completion(
            model="foundry_local/local-model:1",
            api_base="http://127.0.0.1:61234/v1",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=16,
            stream=stream,
            client=client,
        )
        if stream:
            assert not isinstance(response, ModelResponse)
            chunks: Final = tuple(response)
            assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "Hello"
            assert chunks[-1].choices[0].finish_reason == "stop"
        else:
            assert isinstance(response, ModelResponse)
            assert response.choices[0].message.content == "Hello"
            assert response.choices[0].finish_reason == "stop"
            assert response.usage.total_tokens == 3


def _embedding_response(request: httpx.Request) -> httpx.Response:
    assert str(request.url) == "http://127.0.0.1:61234/v1/embeddings"
    assert request.headers["Authorization"] == "Bearer env-key"
    payload: Final = json.loads(request.content)
    assert payload["model"] == "embedding-model:1"
    assert payload["input"] == ["Hello", "World"]
    return httpx.Response(
        200,
        json={
            "object": "list",
            "model": "embedding-model:1",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.25, 0.75]},
                {"object": "embedding", "index": 1, "embedding": [0.5, 0.5]},
            ],
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        },
    )


def test_should_embed_through_foundry_http_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOUNDRY_LOCAL_API_BASE", "http://127.0.0.1:61234/v1")
    monkeypatch.setenv("FOUNDRY_LOCAL_API_KEY", "env-key")
    with (
        httpx.Client(transport=httpx.MockTransport(_embedding_response)) as http_client,
        OpenAI(api_key="env-key", base_url="http://127.0.0.1:61234/v1", http_client=http_client) as client,
    ):
        response: Final = litellm.embedding(
            model="foundry_local/embedding-model:1", input=["Hello", "World"], client=client
        )
        assert [item["embedding"] for item in response.data] == [[0.25, 0.75], [0.5, 0.5]]
        assert response.usage.prompt_tokens == 2


@pytest.mark.asyncio
async def test_should_embed_asynchronously_through_foundry_http_service(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOUNDRY_LOCAL_API_BASE", "http://127.0.0.1:61234/v1")
    monkeypatch.setenv("FOUNDRY_LOCAL_API_KEY", "env-key")
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(_embedding_response)) as http_client,
        AsyncOpenAI(api_key="env-key", base_url="http://127.0.0.1:61234/v1", http_client=http_client) as client,
    ):
        response: Final = await litellm.aembedding(
            model="foundry_local/embedding-model:1", input=["Hello", "World"], client=client
        )
        assert [item["embedding"] for item in response.data] == [[0.25, 0.75], [0.5, 0.5]]
        assert response.usage.prompt_tokens == 2


@pytest.mark.parametrize("status_code", [200, 400])
@pytest.mark.parametrize("input_type", ["bytes", "tuple", "stream", "path"])
def test_should_transcribe_and_clean_up_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status_code: int, input_type: str
) -> None:
    monkeypatch.setenv("FOUNDRY_LOCAL_API_BASE", "http://127.0.0.1:61234/v1")
    monkeypatch.setenv("FOUNDRY_LOCAL_API_KEY", "env-key")
    paths: Final = []
    original: Final = tmp_path / "input.wav"
    original.write_bytes(b"audio-content")
    inputs: Final = {
        "bytes": b"audio-content",
        "tuple": ("input.wav", b"audio-content", "audio/wav"),
        "stream": BytesIO(b"audio-content"),
        "path": original,
    }

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://127.0.0.1:61234/v1/audio/transcriptions"
        assert request.headers["Content-Type"] == "application/json"
        assert request.headers["Authorization"] == "Bearer env-key"
        payload: Final = json.loads(request.content)
        assert payload["model"] == "whisper-tiny"
        assert payload["language"] == "en"
        assert payload["prompt"] == "Greeting"
        assert payload["stream"] is False
        path: Final = Path(payload["filename"])
        paths.append(path)
        assert path.read_bytes() == b"audio-content"
        return httpx.Response(
            status_code, json={"text": "Hello"} if status_code == 200 else {"error": {"message": "Invalid audio"}}
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        client: Final = HTTPHandler(client=http_client)
        if status_code == 200:
            response: Final = litellm.transcription(
                model="foundry_local/whisper-tiny",
                file=inputs[input_type],
                language="en",
                prompt="Greeting",
                client=client,
                max_retries=0,
            )
            assert response.text == "Hello"
        else:
            with pytest.raises(OpenAIError, match="Invalid audio") as error:
                litellm.transcription(
                    model="foundry_local/whisper-tiny",
                    file=inputs[input_type],
                    language="en",
                    prompt="Greeting",
                    client=client,
                    max_retries=0,
                )
            assert error.value.status_code == 400
    assert len(paths) == 1
    assert not paths[0].exists()
    assert not paths[0].parent.exists()
    assert original.read_bytes() == b"audio-content"


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
async def test_should_keep_async_audio_until_completion_and_clean_up(
    monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    monkeypatch.setenv("FOUNDRY_LOCAL_API_BASE", "http://127.0.0.1:61234/v1")
    paths: Final = []

    async def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://127.0.0.1:61234/v1/audio/transcriptions"
        payload: Final = json.loads(request.content)
        path: Final = Path(payload["filename"])
        paths.append(path)
        await asyncio.sleep(0)
        assert path.read_bytes() == b"audio-content"
        if outcome == "cancel":
            raise asyncio.CancelledError
        return httpx.Response(
            400 if outcome == "error" else 200,
            json={"error": {"message": "Invalid audio"}} if outcome == "error" else {"text": "Hello"},
        )

    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(respond))
    async with client.client:
        if outcome == "success":
            response: Final = await litellm.atranscription(
                model="foundry_local/whisper-tiny",
                file=("input.wav", b"audio-content"),
                client=client,
                max_retries=0,
            )
            assert response.text == "Hello"
        else:
            with pytest.raises(asyncio.CancelledError if outcome == "cancel" else litellm.BadRequestError):
                await litellm.atranscription(
                    model="foundry_local/whisper-tiny",
                    file=("input.wav", b"audio-content"),
                    client=client,
                    max_retries=0,
                )
    assert len(paths) == 1
    assert not paths[0].exists()
    assert not paths[0].parent.exists()


def test_should_reject_bare_audio_paths_and_unsupported_transcription_options(tmp_path: Path) -> None:
    config: Final = FoundryLocalAudioTranscriptionConfig()
    path: Final = tmp_path / "input.wav"
    path.write_bytes(b"audio-content")
    with pytest.raises(ValueError, match="bare str"):
        with config.audio_file_context(str(path)):
            pytest.fail("A bare string must not be opened as a server-side path")
    with pytest.raises(litellm.UnsupportedParamsError):
        litellm.utils.get_optional_params_transcription(
            model="whisper-tiny", custom_llm_provider="foundry_local", timestamp_granularities=["word"]
        )
    with pytest.raises(ValueError, match="requires api_base"):
        config.get_complete_url(None, None, "whisper-tiny", {}, {})


def test_should_require_prepared_audio_for_transcription() -> None:
    config: Final = FoundryLocalAudioTranscriptionConfig()
    with pytest.raises(TypeError, match="must be prepared using audio_file_context"):
        config.transform_audio_transcription_request("whisper-tiny", b"audio-content", {}, {})


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_should_complete_chat_asynchronously_over_foundry_http_service(stream: bool) -> None:
    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(lambda request: _respond(request, stream))) as http_client,
        AsyncOpenAI(api_key="fake-api-key", base_url="http://127.0.0.1:61234/v1", http_client=http_client) as client,
    ):
        response: Final = await litellm.acompletion(
            model="foundry_local/local-model:1",
            api_base="http://127.0.0.1:61234/v1",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=16,
            stream=stream,
            client=client,
        )
        if stream:
            assert not isinstance(response, ModelResponse)
            chunks: Final = [chunk async for chunk in response]
            assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "Hello"
            assert chunks[-1].choices[0].finish_reason == "stop"
        else:
            assert isinstance(response, ModelResponse)
            assert response.choices[0].message.content == "Hello"
            assert response.choices[0].finish_reason == "stop"
            assert response.usage.total_tokens == 3
