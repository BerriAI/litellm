"""
Tests proving the Rust OCR path executes Python callbacks from the callback
manager.
"""

import os
from datetime import datetime
from typing import Any, Optional

import pytest

import litellm
from base_ocr_unit_tests import TEST_PDF_URL
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.rust_bridge import ocr as rust_ocr_bridge
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import StandardLoggingPayload

MODEL = "mistral/mistral-ocr-latest"
DOCUMENT: dict[str, object] = {
    "type": "document_url",
    "document_url": TEST_PDF_URL,
}


class OCRCustomLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.standard_logging_payload: Optional[StandardLoggingPayload] = None
        self.response_obj: Any = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.kwargs: dict[str, Any] = {}

    async def async_log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        self.kwargs = kwargs
        self.standard_logging_payload = kwargs.get("standard_logging_object")
        self.response_obj = response_obj
        self.start_time = start_time
        self.end_time = end_time


class OCRCustomGuardrail(CustomGuardrail):
    def __init__(self) -> None:
        super().__init__(
            guardrail_name="ocr-test-guardrail",
            event_hook=GuardrailEventHooks.pre_call,
        )
        self.calls: list[dict[str, object]] = []
        self.success_log_calls = 0

    async def async_pre_call_hook(
        self, user_api_key_dict, cache, data: dict[str, Any], call_type: str
    ) -> None:
        self.calls.append({"data": data, "call_type": call_type})
        data["document"] = {
            **data["document"],
            "guardrail_executed": True,
        }
        data["optional_params"] = {
            **data["optional_params"],
            "include_image_base64": True,
        }

    async def async_log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        self.success_log_calls += 1


@pytest.fixture(autouse=True)
def reset_litellm_rust_callbacks():
    litellm.logging_callback_manager._reset_all_callbacks()
    litellm.use_litellm_rust(False, ocr=None, aocr=None)
    yield
    litellm.logging_callback_manager._reset_all_callbacks()
    litellm.use_litellm_rust(False, ocr=None, aocr=None)


def _mistral_api_key() -> str:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        pytest.skip("MISTRAL_API_KEY is required for live Mistral OCR callback tests")
    return api_key


def _require_native_rust_aocr() -> None:
    if rust_ocr_bridge.load_rust_aocr() is None:
        pytest.skip("native Rust OCR bridge is not available")


@pytest.mark.asyncio
async def test_rust_ocr_executes_custom_logger_from_callback_manager():
    _require_native_rust_aocr()
    custom_logger = OCRCustomLogger()
    litellm.logging_callback_manager.add_litellm_callback(custom_logger)
    litellm.use_litellm_rust(True)

    response = await litellm.aocr(
        model=MODEL,
        document=DOCUMENT,
        api_key=_mistral_api_key(),
    )

    assert len(response.pages) > 0
    assert custom_logger.response_obj["object"] == "ocr"
    assert len(custom_logger.response_obj["value"]["pages"]) > 0
    assert isinstance(custom_logger.start_time, datetime)
    assert isinstance(custom_logger.end_time, datetime)
    assert custom_logger.kwargs["api_base"] == "https://api.mistral.ai/v1"
    assert custom_logger.kwargs["document"] == DOCUMENT
    assert custom_logger.kwargs["optional_params"] == {}

    logged_payload = custom_logger.standard_logging_payload
    assert logged_payload is not None
    assert logged_payload["model"] == "mistral-ocr-latest"
    assert logged_payload["custom_llm_provider"] == "mistral"
    assert logged_payload["call_type"] == "ocr"
    assert logged_payload["response_cost"] == 0.0


@pytest.mark.asyncio
async def test_rust_ocr_executes_custom_guardrail_from_callback_manager():
    _require_native_rust_aocr()
    custom_guardrail = OCRCustomGuardrail()
    litellm.logging_callback_manager.add_litellm_callback(custom_guardrail)
    litellm.use_litellm_rust(True)

    response = await litellm.aocr(
        model=MODEL,
        document=DOCUMENT,
        api_key=_mistral_api_key(),
    )

    assert len(response.pages) > 0
    assert custom_guardrail.calls[0]["call_type"] == "ocr"
    assert custom_guardrail.calls[0]["data"]["document"]["guardrail_executed"] is True
    assert (
        custom_guardrail.calls[0]["data"]["optional_params"]["include_image_base64"]
        is True
    )
    assert custom_guardrail.success_log_calls == 1
