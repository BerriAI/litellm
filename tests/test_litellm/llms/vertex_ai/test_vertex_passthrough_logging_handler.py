from datetime import datetime
from unittest.mock import MagicMock

import httpx
import litellm
import pytest

from litellm.proxy.pass_through_endpoints.llm_provider_handlers.vertex_passthrough_logging_handler import (
    VertexPassthroughLoggingHandler,
)


def test_lyria_predict_response_preserves_audio_response_and_logs_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        litellm.model_cost,
        "vertex_ai/lyria-002",
        {"output_cost_per_second": 0.002},
    )
    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    response = httpx.Response(
        status_code=200,
        json={
            "predictions": [
                {
                    "audioContent": "clip-1",
                    "mimeType": "audio/wav",
                },
                {
                    "audioContent": "clip-2",
                    "mimeType": "audio/wav",
                },
            ]
        },
    )

    result = VertexPassthroughLoggingHandler.vertex_passthrough_handler(
        httpx_response=response,
        logging_obj=logging_obj,
        url_route="/v1/projects/test/locations/us-central1/publishers/google/models/lyria-002:predict",
        result=response.text,
        start_time=datetime.now(),
        end_time=datetime.now(),
        cache_hit=False,
        request_body={"instances": [{"prompt": "ambient piano"}]},
    )

    assert result["result"] == {
        "response": {
            "predictions": [
                {
                    "audioContent": "clip-1",
                    "mimeType": "audio/wav",
                },
                {
                    "audioContent": "clip-2",
                    "mimeType": "audio/wav",
                },
            ]
        }
    }
    assert result["kwargs"]["model"] == "lyria-002"
    assert result["kwargs"]["custom_llm_provider"] == "vertex_ai"
    assert result["kwargs"]["response_cost"] == pytest.approx(0.12)
    assert logging_obj.model == "lyria-002"
    assert logging_obj.model_call_details["response_cost"] == pytest.approx(0.12)
