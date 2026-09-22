"""Fal AI pass-through: upstream URL to model extraction and resolution-keyed spend tracking."""

from datetime import datetime
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.fal_ai_passthrough_logging_handler import (
    FalAIPassthroughLoggingHandler,
)
from litellm.types.utils import ImageResponse

pytestmark: Final = pytest.mark.usefixtures("local_model_cost_map")

UPSTREAM_URL: Final = "https://queue.fal.run/fal-ai/trellis-2"


def _logging_obj(call_id: str = "call-fal") -> LiteLLMLoggingObj:
    return LiteLLMLoggingObj(
        model="unknown",
        messages=[{"role": "user", "content": "passthrough"}],
        stream=False,
        call_type="pass_through_endpoint",
        start_time=datetime.now(),
        litellm_call_id=call_id,
        function_id="passthrough",
    )


def test_is_fal_ai_route_matches_only_the_fal_ai_provider():
    assert FalAIPassthroughLoggingHandler.is_fal_ai_route(UPSTREAM_URL, "fal_ai") is True
    assert FalAIPassthroughLoggingHandler.is_fal_ai_route(UPSTREAM_URL, "deepgram") is False
    assert FalAIPassthroughLoggingHandler.is_fal_ai_route(UPSTREAM_URL, None) is False


def test_handler_extracts_model_urls_and_resolution_keyed_cost():
    upstream_body: Final = {
        "model_glb": {"url": "https://fal.media/model.glb", "content_type": "model/gltf-binary"},
        "images": [{"url": "https://fal.media/preview.png"}],
        "timings": {"inference": 1.2},
    }
    logging_obj: Final = _logging_obj()
    expected_cost: Final = litellm.model_cost["fal_ai/fal-ai/trellis-2"]["output_cost_per_image_1536"]

    handler_result = FalAIPassthroughLoggingHandler().fal_ai_passthrough_handler(
        response_body=upstream_body,
        request_body={"image_url": "https://example.com/in.png", "resolution": 1536},
        logging_obj=logging_obj,
        url_route=UPSTREAM_URL,
        kwargs={"litellm_params": {"metadata": {}}},
    )

    result = handler_result["result"]
    assert isinstance(result, ImageResponse)
    assert [image.url for image in result.data or ()] == [
        "https://fal.media/model.glb",
        "https://fal.media/preview.png",
    ]
    assert result._hidden_params["response_cost"] == pytest.approx(expected_cost)
    assert handler_result["kwargs"]["model"] == "fal-ai/trellis-2"
    assert handler_result["kwargs"]["custom_llm_provider"] == "fal_ai"
    assert handler_result["kwargs"]["response_cost"] == pytest.approx(expected_cost)
    assert handler_result["kwargs"]["litellm_params"] == {"metadata": {}}
    assert logging_obj.model == "fal-ai/trellis-2"
    assert logging_obj.model_call_details["model"] == "fal-ai/trellis-2"
    assert logging_obj.model_call_details["custom_llm_provider"] == "fal_ai"
    assert logging_obj.model_call_details["response_cost"] == pytest.approx(expected_cost)


def test_handler_charges_nothing_and_names_the_model_for_queue_status_and_result_polls():
    for upstream_url in (
        "https://queue.fal.run/fal-ai/trellis-2/requests/req-1/status",
        "https://queue.fal.run/fal-ai/trellis-2/requests/req-1",
    ):
        logging_obj: Final = _logging_obj()
        handler_result = FalAIPassthroughLoggingHandler().fal_ai_passthrough_handler(
            response_body={"status": "COMPLETED"},
            request_body={},
            logging_obj=logging_obj,
            url_route=upstream_url,
            kwargs={},
        )
        assert handler_result["kwargs"]["model"] == "fal-ai/trellis-2"
        assert handler_result["kwargs"]["response_cost"] is None
        assert logging_obj.model_call_details["response_cost"] is None


def test_handler_charges_for_queue_submit():
    handler_result = FalAIPassthroughLoggingHandler().fal_ai_passthrough_handler(
        response_body={"request_id": "req-1", "status": "IN_QUEUE"},
        request_body={"image_url": "https://example.com/in.png", "resolution": 1536},
        logging_obj=_logging_obj(),
        url_route="https://queue.fal.run/fal-ai/trellis-2",
        kwargs={},
    )
    assert handler_result["kwargs"]["model"] == "fal-ai/trellis-2"
    assert handler_result["kwargs"]["response_cost"] == pytest.approx(
        litellm.model_cost["fal_ai/fal-ai/trellis-2"]["output_cost_per_image_1536"]
    )


def test_handler_strips_queue_base_path_prefix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FAL_AI_QUEUE_API_BASE", "https://gw.example/fal/queue")
    handler_result = FalAIPassthroughLoggingHandler().fal_ai_passthrough_handler(
        response_body={"request_id": "req-1", "status": "IN_QUEUE"},
        request_body={"image_url": "https://example.com/in.png", "resolution": 1536},
        logging_obj=_logging_obj(),
        url_route="https://gw.example/fal/queue/fal-ai/trellis-2",
        kwargs={},
    )

    assert handler_result["kwargs"]["model"] == "fal-ai/trellis-2"
    assert handler_result["kwargs"]["response_cost"] == pytest.approx(
        litellm.model_cost["fal_ai/fal-ai/trellis-2"]["output_cost_per_image_1536"]
    )


def test_handler_without_url_values_returns_empty_image_response_and_no_cost():
    handler_result = FalAIPassthroughLoggingHandler().fal_ai_passthrough_handler(
        response_body={"status": "COMPLETED"},
        request_body={},
        logging_obj=_logging_obj(),
        url_route="https://queue.fal.run/fal-ai/no-such-model",
        kwargs={},
    )

    assert isinstance(handler_result["result"], ImageResponse)
    assert not handler_result["result"].data
    assert handler_result["kwargs"]["response_cost"] is None
    assert handler_result["kwargs"]["model"] == "fal-ai/no-such-model"
