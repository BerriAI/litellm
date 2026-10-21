import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, Literal

import pytest
import respx

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.dynamic_config import create_config_class
from litellm.llms.openai_like.json_loader import JSONProviderRegistry

BASE_URL: Final = "https://api-inference.bitdeer.ai/v1"
MODEL: Final = "deepseek-ai/DeepSeek-V4-Flash"
EMBEDDING_MODEL: Final = "BAAI/bge-m3"
RERANK_MODEL: Final = "BAAI/bge-reranker-v2-m3"
IMAGE_MODEL: Final = "seedream-5.0-lite"
MODELS: Final = (
    "deepseek-ai/DeepSeek-V4.1-Flash",
    "deepseek-ai/DeepSeek-V4.1-Flash-1017",
    "zai-org/GLM-5.3",
    "zai-org/GLM-5.3-Flash",
    "Qwen/Qwen3.8-27B",
    MODEL,
)


@pytest.mark.parametrize("api_key", (None, "explicit-key"))
@pytest.mark.parametrize("api_base", (BASE_URL, BASE_URL + "/"))
@pytest.mark.parametrize("routing", ("prefix", "provider", "url"))
def test_bitdeer_credentials(monkeypatch: pytest.MonkeyPatch, api_key: str | None, api_base: str, routing: str) -> None:
    monkeypatch.setenv("BITDEER_API_KEY", "environment-key")

    result: Final = get_llm_provider(
        model=f"bitdeer-ai/{MODEL}" if routing == "prefix" else MODEL,
        custom_llm_provider="bitdeer-ai" if routing == "provider" else None,
        api_base=api_base,
        api_key=api_key,
    )

    assert result == (MODEL, "bitdeer-ai", api_key or "environment-key", api_base)


def test_bitdeer_default_base_and_missing_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BITDEER_API_KEY", raising=False)
    assert get_llm_provider(model=f"bitdeer-ai/{MODEL}") == (MODEL, "bitdeer-ai", None, BASE_URL)
    monkeypatch.setenv("BITDEER_API_KEY", "environment-key")
    assert get_llm_provider(model=f"bitdeer-ai/{MODEL}") == (MODEL, "bitdeer-ai", "environment-key", BASE_URL)


@pytest.mark.parametrize("api_key", (None, ""))
@pytest.mark.parametrize(
    "api_base",
    (
        "https://custom.example/v1",
        "http://api-inference.bitdeer.ai/v1",
        "https://api-inference.bitdeer.ai:8443/v1",
        "https://api-inference.bitdeer.ai.evil.example/v1",
        "https://api-inference.bitdeer.ai@evil.example/v1",
        "https://evil.example/https://api-inference.bitdeer.ai/v1",
        BASE_URL + "/other",
        BASE_URL + "?redirect=https://evil.example",
    ),
)
def test_bitdeer_custom_base_requires_explicit_key(
    monkeypatch: pytest.MonkeyPatch, api_base: str, api_key: str | None
) -> None:
    monkeypatch.setenv("BITDEER_API_KEY", "server-secret")

    with pytest.raises(litellm.BadRequestError, match="explicit api_key is required") as error:
        get_llm_provider(model=f"bitdeer-ai/{MODEL}", api_base=api_base, api_key=api_key)

    assert "server-secret" not in str(error.value)


@pytest.mark.parametrize(
    "api_base", ("http://api-inference.bitdeer.ai/v1", "https://api-inference.bitdeer.ai:8443/v1", BASE_URL + "/other")
)
def test_bitdeer_url_autodetection_rejects_unsafe_key_fallback(monkeypatch: pytest.MonkeyPatch, api_base: str) -> None:
    monkeypatch.setenv("BITDEER_API_KEY", "server-secret")
    with pytest.raises(litellm.BadRequestError, match="explicit api_key is required"):
        get_llm_provider(model=MODEL, api_base=api_base)


def test_bitdeer_dynamic_config_checks_custom_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BITDEER_API_KEY", "server-secret")
    provider: Final = JSONProviderRegistry.get("bitdeer-ai")
    assert provider is not None
    config: Final = create_config_class(provider)()

    with pytest.raises(ValueError, match="explicit api_key is required"):
        config._get_openai_compatible_provider_info("https://custom.example/v1", None)

    assert config._get_openai_compatible_provider_info("https://custom.example/v1", "explicit-key") == (
        "https://custom.example/v1",
        "explicit-key",
    )


@pytest.mark.parametrize("model", (MODEL, f"bitdeer-ai/{MODEL}"))
@pytest.mark.parametrize("api_key", (None, "explicit-key"))
def test_bitdeer_completion_sends_selected_key(
    monkeypatch: pytest.MonkeyPatch, model: str, api_key: str | None
) -> None:
    monkeypatch.setenv("BITDEER_API_KEY", "environment-key")
    with respx.mock() as upstream:
        route: Final = upstream.post(BASE_URL + "/chat/completions").respond(
            200,
            json={
                "id": "chatcmpl-bitdeer",
                "object": "chat.completion",
                "created": 0,
                "model": MODEL,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )
        response: Final = litellm.completion(
            model=model,
            api_base=BASE_URL,
            api_key=api_key,
            messages=[{"role": "user", "content": "Hi"}],
            max_completion_tokens=32,
        )

    request: Final = route.calls.last.request
    payload: Final = json.loads(request.content)
    assert route.call_count == 1
    assert request.headers["authorization"] == f"Bearer {api_key or 'environment-key'}"
    assert payload["model"] == MODEL
    assert payload["max_tokens"] == 32
    assert "max_completion_tokens" not in payload
    assert response.choices[0].message.content == "Hello"
    assert response._hidden_params["response_cost"] == pytest.approx(
        10 * litellm.model_cost[f"bitdeer-ai/{MODEL}"]["input_cost_per_token"]
        + 2 * litellm.model_cost[f"bitdeer-ai/{MODEL}"]["output_cost_per_token"]
    )


def test_bitdeer_completion_rejects_custom_base_before_sending(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BITDEER_API_KEY", "server-secret")
    with respx.mock() as upstream:
        with pytest.raises(litellm.BadRequestError, match="explicit api_key is required"):
            litellm.completion(
                model=f"bitdeer-ai/{MODEL}",
                api_base="https://custom.example/v1",
                messages=[{"role": "user", "content": "Hi"}],
            )
        assert len(upstream.calls) == 0


def _request(
    endpoint: Literal["chat", "image"], model: str, custom_llm_provider: str | None = None, api_base: str | None = None
) -> object:
    if endpoint == "chat":
        return litellm.completion(
            model=model,
            custom_llm_provider=custom_llm_provider,
            api_base=api_base,
            messages=[{"role": "user", "content": "Hi"}],
        )
    return litellm.image_generation(
        model=model, custom_llm_provider=custom_llm_provider, api_base=api_base, prompt="A tree"
    )


async def _arequest(endpoint: Literal["chat", "image"], model: str) -> object:
    if endpoint == "chat":
        return await litellm.acompletion(model=model, messages=[{"role": "user", "content": "Hi"}])
    return await litellm.aimage_generation(model=model, prompt="A tree")


@pytest.mark.parametrize("routing", ("prefix", "provider", "url"))
@pytest.mark.parametrize("endpoint", ("chat", "image"))
def test_bitdeer_missing_credentials_never_send_openai_key(
    monkeypatch: pytest.MonkeyPatch,
    routing: Literal["prefix", "provider", "url"],
    endpoint: Literal["chat", "image"],
) -> None:
    monkeypatch.delenv("BITDEER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setattr(litellm, "api_key", "global-secret")
    monkeypatch.setattr(litellm, "openai_key", "global-openai-secret")
    model: Final = f"bitdeer-ai/{MODEL}" if routing == "prefix" else MODEL
    provider: Final = "bitdeer-ai" if routing == "provider" else None
    api_base: Final = BASE_URL if routing == "url" else None

    with respx.mock() as upstream:
        with pytest.raises(litellm.AuthenticationError, match="BITDEER_API_KEY") as error:
            _request(endpoint, model=model, custom_llm_provider=provider, api_base=api_base)
        assert len(upstream.calls) == 0
        assert "secret" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", ("chat", "image"))
async def test_bitdeer_async_missing_credentials_never_send_openai_key(
    monkeypatch: pytest.MonkeyPatch, endpoint: Literal["chat", "image"]
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.delenv("BITDEER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setattr(litellm, "api_key", "global-secret")
    monkeypatch.setattr(litellm, "openai_key", "global-openai-secret")

    with respx.mock() as upstream:
        with pytest.raises(litellm.AuthenticationError, match="BITDEER_API_KEY") as error:
            await _arequest(endpoint, model=f"bitdeer-ai/{MODEL}")
        assert len(upstream.calls) == 0
        assert "secret" not in str(error.value)


def _unique_entries(pairs: Sequence[tuple[str, object]]) -> Mapping[str, object]:
    keys: Final = tuple(key for key, _ in pairs)
    assert len(keys) == len(set(keys)), "Duplicate JSON key"
    return dict(pairs)


def test_bitdeer_pricing_and_packaged_registry() -> None:
    root: Final = Path(litellm.__file__).parent.parent
    prices: Final = json.loads(
        (root / "model_prices_and_context_window.json").read_text(), object_pairs_hook=_unique_entries
    )
    backup: Final = json.loads((root / "litellm/model_prices_and_context_window_backup.json").read_text())

    for model in (*MODELS, EMBEDDING_MODEL, RERANK_MODEL, IMAGE_MODEL):
        extra: Final = f"bitdeer-ai/{model}"
        assert prices[extra] == backup[extra] == litellm.model_cost[extra]
        assert prices[extra]["litellm_provider"] == "bitdeer-ai"

    for model in MODELS:
        name: Final = f"bitdeer-ai/{model}"
        entry: Final = prices[name]
        assert entry["mode"] == "chat"
        assert entry["max_input_tokens"] > 0
        assert entry["litellm_provider"] == "bitdeer-ai"
        prompt_cost, output_cost = litellm.cost_per_token(model=name, prompt_tokens=100, completion_tokens=10)
        assert prompt_cost == pytest.approx(100 * entry["input_cost_per_token"])
        assert output_cost == pytest.approx(10 * entry["output_cost_per_token"])


def test_bitdeer_embedding_sends_selected_key_and_charges_input_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BITDEER_API_KEY", "environment-key")
    with respx.mock() as upstream:
        route: Final = upstream.post(BASE_URL + "/embeddings").respond(
            200,
            json={
                "object": "list",
                "model": EMBEDDING_MODEL,
                "data": [{"object": "embedding", "index": 0, "embedding": [0.25, -0.5]}],
                "usage": {"prompt_tokens": 7, "total_tokens": 7},
            },
        )
        response: Final = litellm.embedding(model=f"bitdeer-ai/{EMBEDDING_MODEL}", input=["hello"], api_base=BASE_URL)

    request: Final = route.calls.last.request
    assert json.loads(request.content) == {"model": EMBEDDING_MODEL, "input": ["hello"]}
    assert request.headers["authorization"] == "Bearer environment-key"
    assert response.data[0]["embedding"] == [0.25, -0.5]
    assert response._hidden_params["response_cost"] == pytest.approx(
        7 * litellm.model_cost[f"bitdeer-ai/{EMBEDDING_MODEL}"]["input_cost_per_token"]
    )


def test_bitdeer_embedding_without_bitdeer_key_never_sends_the_openai_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BITDEER_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    with respx.mock() as upstream:
        with pytest.raises(litellm.AuthenticationError, match="api_key"):
            litellm.embedding(model=f"bitdeer-ai/{EMBEDDING_MODEL}", input=["hello"], api_base=BASE_URL)
        assert len(upstream.calls) == 0


def test_bitdeer_rerank_orders_results_and_sends_selected_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BITDEER_API_KEY", "environment-key")
    with respx.mock() as upstream:
        route: Final = upstream.post(BASE_URL + "/rerank").respond(
            200,
            json={
                "id": "rerank-bitdeer",
                "model": RERANK_MODEL,
                "usage": {"total_tokens": 9},
                "results": [
                    {"index": 1, "relevance_score": 0.91, "document": {"text": "paris"}},
                    {"index": 0, "relevance_score": 0.02, "document": {"text": "berlin"}},
                ],
            },
        )
        response: Final = litellm.rerank(
            model=f"bitdeer-ai/{RERANK_MODEL}",
            query="capital of france",
            documents=["berlin", "paris"],
            api_key="explicit-key",
        )

    request: Final = route.calls.last.request
    payload: Final = json.loads(request.content)
    assert payload["model"] == RERANK_MODEL
    assert payload["query"] == "capital of france"
    assert payload["documents"] == ["berlin", "paris"]
    assert request.headers["authorization"] == "Bearer explicit-key"
    assert [(result["index"], result["relevance_score"]) for result in response.results] == [(1, 0.91), (0, 0.02)]
    assert response._hidden_params["response_cost"] == pytest.approx(
        9 * litellm.model_cost[f"bitdeer-ai/{RERANK_MODEL}"]["input_cost_per_token"]
    )


def test_bitdeer_rerank_without_bitdeer_key_never_sends_the_hosted_vllm_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BITDEER_API_KEY", raising=False)
    monkeypatch.setenv("HOSTED_VLLM_API_KEY", "vllm-secret")
    with respx.mock() as upstream:
        with pytest.raises(litellm.AuthenticationError, match="BITDEER_API_KEY"):
            litellm.rerank(model=f"bitdeer-ai/{RERANK_MODEL}", query="q", documents=["d"])
        assert len(upstream.calls) == 0


def test_bitdeer_rerank_rejects_custom_base_before_sending(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BITDEER_API_KEY", "server-secret")
    with respx.mock() as upstream:
        with pytest.raises(litellm.BadRequestError, match="explicit api_key is required"):
            litellm.rerank(
                model=f"bitdeer-ai/{RERANK_MODEL}",
                query="q",
                documents=["d"],
                api_base="https://custom.example/v1",
            )
        assert len(upstream.calls) == 0


def test_bitdeer_image_generation_sends_selected_key_and_charges_per_image(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BITDEER_API_KEY", "environment-key")
    with respx.mock() as upstream:
        route: Final = upstream.post(BASE_URL + "/images/generations").respond(
            200,
            json={"created": 0, "data": [{"url": "https://images.example/out.jpeg", "size": "2048x2048"}]},
        )
        response: Final = litellm.image_generation(
            model=f"bitdeer-ai/{IMAGE_MODEL}", prompt="a red circle", api_base=BASE_URL, api_key="explicit-key"
        )

    request: Final = route.calls.last.request
    payload: Final = json.loads(request.content)
    assert payload["model"] == IMAGE_MODEL
    assert payload["prompt"] == "a red circle"
    assert request.headers["authorization"] == "Bearer explicit-key"
    assert response.data[0].url == "https://images.example/out.jpeg"
    assert response._hidden_params["response_cost"] == pytest.approx(
        litellm.model_cost[f"bitdeer-ai/{IMAGE_MODEL}"]["output_cost_per_image"]
    )
