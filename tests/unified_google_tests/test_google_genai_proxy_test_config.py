import time
from pathlib import Path
from typing import Final, ReadOnly, TypedDict

import httpx
import pytest
import respx
import yaml
from pydantic import TypeAdapter

import litellm
from litellm import Router

CONFIG_PATH: Final = Path(__file__).parent / "google_genai_proxy_test_config.yaml"
GEMINI_HOST: Final = "generativelanguage.googleapis.com"
GEMINI_GENERATE_CONTENT_PATH: Final = "/v1beta/models/gemini-2.5-flash-lite:generateContent"
RESOURCE_EXHAUSTED: Final = {
    "error": {"code": 429, "message": "Resource exhausted. Please try again later.", "status": "RESOURCE_EXHAUSTED"}
}
PONG: Final = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": "pong"}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 1, "totalTokenCount": 9},
}
CONSECUTIVE_RATE_LIMITS: Final = 3
MINIMUM_BACKOFF_SECONDS: Final = 0.5 + 1.0 + 2.0


class _Deployment(TypedDict):
    model_name: ReadOnly[str]
    litellm_params: ReadOnly[dict[str, str]]


class _ProxyConfig(TypedDict):
    model_list: ReadOnly[list[_Deployment]]
    router_settings: ReadOnly[dict[str, dict[str, int]]]


def _router_from_ci_proxy_config() -> Router:
    config: Final = TypeAdapter(_ProxyConfig).validate_python(yaml.safe_load(CONFIG_PATH.read_text()))
    gemini_deployments: Final = [
        {"model_name": deployment["model_name"], "litellm_params": {**deployment["litellm_params"], "api_key": "test"}}
        for deployment in config["model_list"]
        if deployment["model_name"] == "gemini-2.5-flash-lite"
    ]
    return Router(model_list=gemini_deployments, retry_policy=config["router_settings"]["retry_policy"])


@pytest.mark.asyncio
async def test_ci_proxy_config_rides_out_consecutive_429s_with_backoff(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    route: Final = respx_mock.post(host=GEMINI_HOST, path=GEMINI_GENERATE_CONTENT_PATH).mock(
        side_effect=[httpx.Response(429, json=RESOURCE_EXHAUSTED)] * CONSECUTIVE_RATE_LIMITS
        + [httpx.Response(200, json=PONG)]
    )
    started: Final = time.monotonic()
    response: Final = await _router_from_ci_proxy_config().agenerate_content(
        model="gemini-2.5-flash-lite",
        contents=[{"role": "user", "parts": [{"text": "Reply with only the single word: pong"}]}],
    )
    elapsed: Final = time.monotonic() - started

    assert response.model_dump()["candidates"][0]["content"]["parts"][0]["text"] == "pong"
    assert route.call_count == CONSECUTIVE_RATE_LIMITS + 1
    assert elapsed >= MINIMUM_BACKOFF_SECONDS
