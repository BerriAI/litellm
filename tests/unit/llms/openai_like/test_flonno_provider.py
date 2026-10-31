import json
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import httpx
import pytest
import respx
from openai import OpenAI

import litellm
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.proxy.auth.model_checks import get_provider_models
from litellm.types.utils import LlmProviders

FLONNO_API_BASE: Final = "https://www.flonno.com/v1"
FLONNO_MODEL: Final = "qwen3.8-27b"
FLONNO_PREFIXED: Final = f"flonno/{FLONNO_MODEL}"

_REPO_ROOT: Final = Path(__file__).resolve().parents[4]
_PRIMARY_PRICES: Final = json.loads((_REPO_ROOT / "model_prices_and_context_window.json").read_text())
_BACKUP_PRICES: Final = json.loads((_REPO_ROOT / "litellm" / "model_prices_and_context_window_backup.json").read_text())


class _RecordingFlonno:
    def __init__(self, usage: Mapping[str, object]) -> None:
        self.usage: Final = usage
        self.requests: tuple[httpx.Request, ...] = ()

    def respond(self, request: httpx.Request) -> httpx.Response:
        self.requests = (*self.requests, request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-flonno",
                "object": "chat.completion",
                "created": 1234567890,
                "model": json.loads(request.content)["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": dict(self.usage),
            },
        )

    def client(self) -> OpenAI:
        return OpenAI(
            api_key="flonno-test-key",
            base_url=FLONNO_API_BASE,
            http_client=httpx.Client(transport=httpx.MockTransport(self.respond)),
        )


def test_flonno_in_provider_list_and_json_registry():
    assert LlmProviders.FLONNO.value == "flonno"
    assert "flonno" in litellm.provider_list
    assert JSONProviderRegistry.exists("flonno")
    provider = JSONProviderRegistry.get("flonno")
    assert provider is not None
    assert provider.base_url == FLONNO_API_BASE
    assert provider.api_key_env == "FLONNO_API_KEY"
    assert "/v1/chat/completions" in provider.supported_endpoints
    assert "/v1/messages" in provider.supported_endpoints
    assert "/v1/responses" not in provider.supported_endpoints


def test_flonno_in_openai_compatible_providers():
    from litellm.constants import openai_compatible_providers

    assert "flonno" in openai_compatible_providers


def test_prefixed_model_reaches_flonno_chat_completions_without_the_prefix():
    flonno: Final = _RecordingFlonno({"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4})

    with flonno.client() as client:
        litellm.completion(
            model=FLONNO_PREFIXED,
            messages=[{"role": "user", "content": "hello"}],
            client=client,
        )

    assert len(flonno.requests) == 1
    assert str(flonno.requests[0].url) == f"{FLONNO_API_BASE}/chat/completions"
    assert json.loads(flonno.requests[0].content) == {
        "model": FLONNO_MODEL,
        "messages": [{"role": "user", "content": "hello"}],
    }


@pytest.mark.asyncio
async def test_anthropic_messages_request_goes_to_flonno_natively(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    with respx.mock() as upstream:
        route: Final = upstream.post(f"{FLONNO_API_BASE}/messages").respond(
            200,
            json={
                "id": "msg_flonno",
                "type": "message",
                "role": "assistant",
                "model": FLONNO_MODEL,
                "content": [{"type": "text", "text": "Hello from Flonno"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 4, "output_tokens": 3},
            },
        )
        response: Final = await litellm.anthropic.messages.acreate(
            model=FLONNO_PREFIXED,
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=32,
            api_key="sk-test",
        )

    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer sk-test"
    assert body["model"] == FLONNO_MODEL
    assert body["messages"] == [{"role": "user", "content": "Say hello"}]
    assert response["content"][0]["text"] == "Hello from Flonno"


def test_flonno_key_and_base_come_from_flonno_env_vars(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FLONNO_API_KEY", "flonno-env-key")
    monkeypatch.delenv("FLONNO_API_BASE", raising=False)

    assert get_llm_provider(model=FLONNO_PREFIXED) == (
        FLONNO_MODEL,
        "flonno",
        "flonno-env-key",
        FLONNO_API_BASE,
    )

    monkeypatch.setenv("FLONNO_API_BASE", "https://flonno.internal.example/v1")

    assert get_llm_provider(model=FLONNO_PREFIXED)[3] == "https://flonno.internal.example/v1"


def test_flonno_wildcard_lists_every_flonno_catalog_model():
    catalog_models: Final = sorted(
        key for key, row in litellm.model_cost.items() if row.get("litellm_provider") == "flonno"
    )

    assert catalog_models == [FLONNO_PREFIXED]
    assert sorted(get_provider_models("flonno") or []) == catalog_models


def test_catalog_row_matches_its_backup_and_fits_its_context():
    row: Final = _PRIMARY_PRICES[FLONNO_PREFIXED]

    assert row == _BACKUP_PRICES[FLONNO_PREFIXED]
    assert row["mode"] == "chat"
    assert row["max_output_tokens"] <= row["max_input_tokens"]
    assert row["input_cost_per_token"] == 3e-08
    assert row["output_cost_per_token"] == 2.5e-07
    assert litellm.get_model_info(FLONNO_PREFIXED)["litellm_provider"] == "flonno"
