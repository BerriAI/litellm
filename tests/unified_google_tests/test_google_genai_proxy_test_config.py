import time
from pathlib import Path
from typing import Final

import httpx
import pytest
import respx
import yaml
from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm import Router
from litellm.constants import INITIAL_RETRY_DELAY, MAX_RETRY_DELAY
from litellm.llms.vertex_ai.common_utils import get_vertex_base_url
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase

CONFIG_PATH: Final = Path(__file__).parent / "google_genai_proxy_test_config.yaml"
GEMINI_DEPLOYMENT: Final = "gemini-2.5-flash-lite"
VERTEX_DEPLOYMENT: Final = "vertex-gemini-2.5-flash-lite"
GEMINI_HOST: Final = "generativelanguage.googleapis.com"
GEMINI_GENERATE_CONTENT_PATH: Final = "/v1beta/models/gemini-2.5-flash-lite:generateContent"
VERTEX_GLOBAL_BASE_URL: Final = "https://aiplatform.googleapis.com"
RESOURCE_EXHAUSTED: Final = {
    "error": {"code": 429, "message": "Resource exhausted. Please try again later.", "status": "RESOURCE_EXHAUSTED"}
}
PONG: Final = {
    "candidates": [{"content": {"role": "model", "parts": [{"text": "pong"}]}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 1, "totalTokenCount": 9},
}
CONSECUTIVE_RATE_LIMITS: Final = 3
MINIMUM_BACKOFF_SECONDS: Final = sum(
    min(INITIAL_RETRY_DELAY * 2**attempt, MAX_RETRY_DELAY) for attempt in range(CONSECUTIVE_RATE_LIMITS)
)


class _Deployment(TypedDict):
    model_name: ReadOnly[str]
    litellm_params: ReadOnly[dict[str, str]]


class _ProxyConfig(TypedDict):
    model_list: ReadOnly[list[_Deployment]]
    router_settings: ReadOnly[dict[str, dict[str, int]]]


def _ci_proxy_config() -> _ProxyConfig:
    return TypeAdapter(_ProxyConfig).validate_python(yaml.safe_load(CONFIG_PATH.read_text()))


def _litellm_params(config: _ProxyConfig, model_name: str) -> dict[str, str]:
    return next(
        deployment["litellm_params"] for deployment in config["model_list"] if deployment["model_name"] == model_name
    )


def _router_from_ci_proxy_config() -> Router:
    config: Final = _ci_proxy_config()
    return Router(
        model_list=[
            {
                "model_name": GEMINI_DEPLOYMENT,
                "litellm_params": {**_litellm_params(config, GEMINI_DEPLOYMENT), "api_key": "test"},
            }
        ],
        retry_policy=config["router_settings"]["retry_policy"],
    )


def test_ci_proxy_config_sends_vertex_calls_to_the_global_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERTEXAI_LOCATION", "us-east5")
    location: Final = VertexBase.safe_get_vertex_ai_location(_litellm_params(_ci_proxy_config(), VERTEX_DEPLOYMENT))

    assert location == "global"
    assert get_vertex_base_url(location) == VERTEX_GLOBAL_BASE_URL


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
        model=GEMINI_DEPLOYMENT,
        contents=[{"role": "user", "parts": [{"text": "Reply with only the single word: pong"}]}],
    )
    elapsed: Final = time.monotonic() - started

    assert response.model_dump()["candidates"][0]["content"]["parts"][0]["text"] == "pong"
    assert route.call_count == CONSECUTIVE_RATE_LIMITS + 1
    assert elapsed >= MINIMUM_BACKOFF_SECONDS
