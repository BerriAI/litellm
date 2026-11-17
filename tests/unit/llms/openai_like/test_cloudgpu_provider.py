import json
from collections.abc import AsyncIterator
from typing import Final

import httpx
import openai
import pytest
import respx

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

CLOUDGPU_MODEL: Final = "cloudgpu/deepseek-v4-flash"
CLOUDGPU_MODEL_ID: Final = "deepseek-v4-flash"
CLOUDGPU_BASE: Final = "https://cloudgpu.app/v1"


def test_cloudgpu_json_config_exists():
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    assert JSONProviderRegistry.exists("cloudgpu")
    cfg = JSONProviderRegistry.get("cloudgpu")
    assert cfg.base_url == CLOUDGPU_BASE
    assert cfg.api_key_env == "CLOUDGPU_API_KEY"
    assert cfg.api_base_env == "CLOUDGPU_API_BASE"


def test_cloudgpu_provider_resolution(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("CLOUDGPU_API_KEY", "cloudgpu-test-key")
    monkeypatch.delenv("CLOUDGPU_API_BASE", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model=CLOUDGPU_MODEL, custom_llm_provider=None, api_base=None, api_key=None
    )

    assert model == CLOUDGPU_MODEL_ID
    assert provider == "cloudgpu"
    assert api_key == "cloudgpu-test-key"
    assert api_base == CLOUDGPU_BASE


def test_cloudgpu_explicit_credentials_win(monkeypatch: pytest.MonkeyPatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("CLOUDGPU_API_KEY", "cloudgpu-env-key")

    _, provider, api_key, api_base = get_llm_provider(
        model=CLOUDGPU_MODEL,
        custom_llm_provider=None,
        api_base="https://cloudgpu.internal.example/v1",
        api_key="cloudgpu-explicit-key",
    )

    assert provider == "cloudgpu"
    assert api_key == "cloudgpu-explicit-key"
    assert api_base == "https://cloudgpu.internal.example/v1"


def test_cloudgpu_chat_completion_request():
    with respx.mock() as upstream:
        route = upstream.post(f"{CLOUDGPU_BASE}/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl_cloudgpu",
                "object": "chat.completion",
                "created": 1_793_000_000,
                "model": CLOUDGPU_MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello from CloudGPU"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8},
            },
        )
        response = litellm.completion(
            model=CLOUDGPU_MODEL,
            messages=[{"role": "user", "content": "Say hello"}],
            api_key="cloudgpu-test-key",
        )

    request = route.calls.last.request
    body = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == "Bearer cloudgpu-test-key"
    assert body["model"] == CLOUDGPU_MODEL_ID
    assert response.choices[0].message.content == "Hello from CloudGPU"


@pytest.fixture
def cloudgpu_upstream(respx_mock: respx.MockRouter) -> respx.Route:
    return respx_mock.post(f"{CLOUDGPU_BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl_cloudgpu",
                "object": "chat.completion",
                "created": 1_793_000_000,
                "model": CLOUDGPU_MODEL_ID,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )


@pytest.fixture
async def cloudgpu_http_client(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport()) as client:
        monkeypatch.setattr(litellm, "aclient_session", client)
        yield client


@pytest.mark.asyncio
@pytest.mark.respx(assert_all_called=False)
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("missing_key", [None, ""])
@pytest.mark.parametrize("key_source", ["environment", "openai_key", "api_key"])
@pytest.mark.parametrize("routing", ["prefix", "provider"])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("experimental", [False, True])
async def test_cloudgpu_missing_key_never_sends_openai_credentials(
    monkeypatch: pytest.MonkeyPatch,
    cloudgpu_http_client: httpx.AsyncClient,
    cloudgpu_upstream: respx.Route,
    is_async: bool,
    key_source: str,
    missing_key: str | None,
    routing: str,
    stream: bool,
    experimental: bool,
) -> None:
    monkeypatch.delenv("CLOUDGPU_API_KEY", raising=False)
    monkeypatch.delenv("CLOUDGPU_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "openai_key", None)
    monkeypatch.setenv("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", str(experimental))
    if key_source == "environment":
        monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-test-key")
    else:
        monkeypatch.setattr(litellm, key_source, "unrelated-openai-test-key")
    if is_async:
        with pytest.raises(litellm.AuthenticationError, match="CLOUDGPU_API_KEY"):
            await litellm.acompletion(
                model=CLOUDGPU_MODEL if routing == "prefix" else CLOUDGPU_MODEL_ID,
                custom_llm_provider="cloudgpu" if routing == "provider" else None,
                api_key=missing_key,
                messages=[{"role": "user", "content": "hello"}],
                stream=stream,
                client=AsyncHTTPHandler(transport=httpx.AsyncHTTPTransport()) if experimental else None,
            )
    else:
        with pytest.raises(litellm.AuthenticationError, match="CLOUDGPU_API_KEY"):
            litellm.completion(
                model=CLOUDGPU_MODEL if routing == "prefix" else CLOUDGPU_MODEL_ID,
                custom_llm_provider="cloudgpu" if routing == "provider" else None,
                api_key=missing_key,
                messages=[{"role": "user", "content": "hello"}],
                stream=stream,
            )
    assert cloudgpu_upstream.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    ("explicit_key", "environment_key"),
    [
        (None, "env-cloudgpu-test-key"),
        ("explicit-cloudgpu-test-key", None),
        ("explicit-cloudgpu-test-key", "env-cloudgpu-test-key"),
    ],
)
@pytest.mark.parametrize("routing", ["prefix", "provider"])
@pytest.mark.parametrize("experimental", [False, True])
async def test_cloudgpu_completion_uses_only_selected_credentials(
    monkeypatch: pytest.MonkeyPatch,
    cloudgpu_upstream: respx.Route,
    is_async: bool,
    explicit_key: str | None,
    environment_key: str | None,
    cloudgpu_http_client: httpx.AsyncClient,
    routing: str,
    experimental: bool,
) -> None:
    if environment_key is None:
        monkeypatch.delenv("CLOUDGPU_API_KEY", raising=False)
    else:
        monkeypatch.setenv("CLOUDGPU_API_KEY", environment_key)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-test-key")
    monkeypatch.delenv("CLOUDGPU_API_BASE", raising=False)
    monkeypatch.setenv("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", str(experimental))
    result: Final = (litellm.acompletion if is_async else litellm.completion)(
        model=CLOUDGPU_MODEL if routing == "prefix" else CLOUDGPU_MODEL_ID,
        custom_llm_provider="cloudgpu" if routing == "provider" else None,
        api_key=explicit_key,
        client=AsyncHTTPHandler(transport=httpx.AsyncHTTPTransport()) if is_async and experimental else None,
        messages=[{"role": "user", "content": "hello"}],
    )
    if not isinstance(result, (litellm.ModelResponse, litellm.CustomStreamWrapper)):
        await result
    assert cloudgpu_upstream.call_count == 1
    assert cloudgpu_upstream.calls.last.request.headers["authorization"] == (
        f"Bearer {explicit_key or 'env-cloudgpu-test-key'}"
    )


@pytest.mark.asyncio
async def test_cloudgpu_explicit_async_sdk_client_keeps_its_credentials(
    monkeypatch: pytest.MonkeyPatch, cloudgpu_upstream: respx.Route
) -> None:
    monkeypatch.delenv("CLOUDGPU_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-test-key")
    monkeypatch.setenv("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", "False")
    async with openai.AsyncOpenAI(
        api_key="client-cloudgpu-test-key",
        base_url=CLOUDGPU_BASE,
        http_client=httpx.AsyncClient(transport=httpx.AsyncHTTPTransport()),
    ) as client:
        await litellm.acompletion(model=CLOUDGPU_MODEL, messages=[{"role": "user", "content": "hello"}], client=client)
    assert cloudgpu_upstream.call_count == 1
    assert cloudgpu_upstream.calls.last.request.headers["authorization"] == "Bearer client-cloudgpu-test-key"


def test_cloudgpu_explicit_sync_sdk_client_keeps_its_credentials(
    monkeypatch: pytest.MonkeyPatch, cloudgpu_upstream: respx.Route
) -> None:
    monkeypatch.delenv("CLOUDGPU_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-test-key")
    monkeypatch.setenv("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", "False")
    with openai.OpenAI(api_key="client-cloudgpu-test-key", base_url=CLOUDGPU_BASE) as client:
        litellm.completion(model=CLOUDGPU_MODEL, messages=[{"role": "user", "content": "hello"}], client=client)
    assert cloudgpu_upstream.call_count == 1
    assert cloudgpu_upstream.calls.last.request.headers["authorization"] == "Bearer client-cloudgpu-test-key"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("experimental", [False, True])
async def test_cloudgpu_stream_preserves_provider_credentials(
    monkeypatch: pytest.MonkeyPatch,
    respx_mock: respx.MockRouter,
    cloudgpu_http_client: httpx.AsyncClient,
    is_async: bool,
    experimental: bool,
) -> None:
    monkeypatch.setenv("CLOUDGPU_API_KEY", "stream-cloudgpu-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-test-key")
    monkeypatch.setenv("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", str(experimental))
    chunk: Final = {
        "id": "chatcmpl_stream",
        "object": "chat.completion.chunk",
        "created": 1_793_000_000,
        "model": CLOUDGPU_MODEL_ID,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
    }
    route: Final = respx_mock.post(f"{CLOUDGPU_BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200, text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n", headers={"content-type": "text/event-stream"}
        )
    )
    result: Final = (litellm.acompletion if is_async else litellm.completion)(
        model=CLOUDGPU_MODEL,
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        client=AsyncHTTPHandler(transport=httpx.AsyncHTTPTransport()) if is_async and experimental else None,
    )
    stream: Final = result if isinstance(result, (litellm.ModelResponse, litellm.CustomStreamWrapper)) else await result
    assert isinstance(stream, litellm.CustomStreamWrapper)
    if is_async:
        async_content: Final = "".join([part.choices[0].delta.content or "" async for part in stream])
        assert async_content == "hello"
    else:
        sync_content: Final = "".join(part.choices[0].delta.content or "" for part in stream)
        assert sync_content == "hello"
    assert route.call_count == 1
    assert route.calls.last.request.headers["authorization"] == "Bearer stream-cloudgpu-test-key"
