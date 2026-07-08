import os
import sys
from typing import Any

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm


class _FakeVertexBase:
    def __init__(self) -> None:
        self.region_calls: list[dict[str, Any]] = []
        self.ensure_calls: list[dict[str, Any]] = []

    def get_vertex_region(self, *, vertex_region: str | None, model: str) -> str:
        self.region_calls.append({"vertex_region": vertex_region, "model": model})
        return vertex_region or "global"

    async def _ensure_access_token_async(
        self,
        *,
        credentials: str | None,
        project_id: str | None,
        custom_llm_provider: str,
    ) -> tuple[str, str]:
        self.ensure_calls.append(
            {
                "credentials": credentials,
                "project_id": project_id,
                "custom_llm_provider": custom_llm_provider,
            }
        )
        return "access-token", project_id or "resolved-project"


class _FakeWebsocketConnect:
    def __init__(self, calls: list[dict[str, Any]], url: str, **kwargs: Any) -> None:
        calls.append({"url": url, **kwargs})

    async def __aenter__(self) -> "_FakeWebsocketConnect":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return False


@pytest.mark.asyncio
async def test_vertex_realtime_health_check_uses_model_vertex_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.litellm_core_utils.health_check_helpers import HealthCheckHelpers
    from litellm.realtime_api import main as realtime_main

    fake_vertex_base = _FakeVertexBase()
    connect_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(realtime_main, "vertex_llm_base", fake_vertex_base)
    monkeypatch.setattr(
        "websockets.connect",
        lambda url, **kwargs: _FakeWebsocketConnect(connect_calls, url, **kwargs),
    )
    monkeypatch.setattr(
        HealthCheckHelpers,
        "_update_model_params_with_health_check_tracking_information",
        staticmethod(lambda model_params: model_params),
    )

    result = await litellm.ahealth_check(
        model_params={
            "model": "vertex_ai/gemini-live-2.5-flash-native-audio",
            "vertex_credentials": '{"type":"service_account"}',
            "vertex_project": "test-project",
            "vertex_location": "us-central1",
        },
        mode="realtime",
    )

    assert result == {}
    assert fake_vertex_base.region_calls == [
        {"vertex_region": "us-central1", "model": "gemini-live-2.5-flash-native-audio"}
    ]
    assert fake_vertex_base.ensure_calls == [
        {
            "credentials": '{"type":"service_account"}',
            "project_id": "test-project",
            "custom_llm_provider": "vertex_ai",
        }
    ]
    assert connect_calls[0]["url"] == (
        "wss://us-central1-aiplatform.googleapis.com/ws/"
        "google.cloud.aiplatform.v1.LlmBidiService/BidiGenerateContent"
    )
    assert connect_calls[0]["additional_headers"] == {
        "Authorization": "Bearer access-token",
        "x-goog-user-project": "test-project",
    }
