from datetime import datetime, timezone
from typing import Final

import httpx
import pytest

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging

pytestmark: Final = pytest.mark.usefixtures("local_model_cost_map")

_LIVE_ROUTE: Final = "/vertex_ai/live"
_LIVE_MODEL: Final = "gemini-live-2.5-flash"
_START: Final = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
_END: Final = datetime(2025, 1, 2, 3, 4, 9, tzinfo=timezone.utc)
_FIRST_TURN_USAGE: Final = {"promptTokenCount": 10, "candidatesTokenCount": 4, "totalTokenCount": 14}
_SECOND_TURN_USAGE: Final = {"promptTokenCount": 30, "candidatesTokenCount": 6, "totalTokenCount": 36}


def _logging_obj() -> LiteLLMLoggingObj:
    return LiteLLMLoggingObj(
        model="unknown",
        messages=[{"role": "user", "content": "WebSocket connection"}],
        stream=True,
        call_type="pass_through_endpoint",
        start_time=_START,
        litellm_call_id="call-live",
        function_id="websocket_passthrough",
    )


def _normalize_live_session(response_body: dict | list[dict[str, object]] | None) -> dict:
    return PassThroughEndpointLogging().normalize_llm_passthrough_logging_payload(
        httpx_response=httpx.Response(200, request=httpx.Request("GET", f"https://proxy.example.test{_LIVE_ROUTE}")),
        response_body=response_body,
        request_body={},
        logging_obj=_logging_obj(),
        url_route=_LIVE_ROUTE,
        result="websocket_connection_successful",
        start_time=_START,
        end_time=_END,
        cache_hit=False,
        model=_LIVE_MODEL,
    )


def test_vertex_ai_live_route_sums_usage_of_every_turn_in_the_websocket_frames() -> None:
    normalized = _normalize_live_session(
        [
            {"setupComplete": {}},
            {"serverContent": {"modelTurn": {"parts": [{"text": "hello"}]}}},
            {"usageMetadata": _FIRST_TURN_USAGE},
            {"serverContent": {"modelTurn": {"parts": [{"text": "goodbye"}]}}},
            {"usageMetadata": _SECOND_TURN_USAGE},
        ]
    )

    response = normalized["standard_logging_response_object"]
    assert (response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens) == (40, 10, 50)
    assert response.model == _LIVE_MODEL
    assert normalized["kwargs"] == {"model": _LIVE_MODEL, "custom_llm_provider": "vertex_ai"}


@pytest.mark.parametrize(
    "response_body",
    [
        pytest.param({"usageMetadata": _FIRST_TURN_USAGE}, id="single-json-object"),
        pytest.param(None, id="no-body"),
        pytest.param([{"setupComplete": {}}], id="frames-without-usage"),
    ],
)
def test_vertex_ai_live_route_without_usage_frames_yields_no_logging_response(
    response_body: dict | list[dict[str, object]] | None,
) -> None:
    normalized = _normalize_live_session(response_body)

    assert normalized["standard_logging_response_object"] is None
    assert normalized["kwargs"] == {"model": _LIVE_MODEL}
