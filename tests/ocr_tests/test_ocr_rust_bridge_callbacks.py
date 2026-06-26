"""Network-free tests proving Rust OCR receives and executes Python callbacks."""

from datetime import datetime
from typing import Any, Optional

import pytest

import litellm
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import StandardLoggingPayload

MODEL = "mistral/mistral-ocr-latest"
DOCUMENT: dict[str, object] = {
    "type": "document_url",
    "document_url": "data:application/pdf;base64,JVBERi0xLjQK",
}
FAKE_RUST_OCR_RESPONSE: dict[str, object] = {
    "pages": [{"index": 0, "markdown": "hello from rust ocr"}],
    "model": "mistral-ocr-latest",
    "document_annotation": None,
    "usage_info": {"pages_processed": 1},
    "object": "ocr",
    "_hidden_params": {"litellm_rust": True},
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
        data["optional_params"] = {
            **data["optional_params"],
            "include_image_base64": True,
        }
        data["guardrail_executed"] = True

    async def async_log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: datetime,
        end_time: datetime,
    ) -> None:
        self.success_log_calls += 1


class MockExecutingRustAocrBridge:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def __call__(
        self,
        model: str,
        document: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout_seconds: float | None,
        callbacks: list[object] | None = None,
        guardrails: list[object] | None = None,
    ) -> dict[str, object]:
        data: dict[str, Any] = {
            "model": model,
            "document": document,
            "api_key": api_key,
            "api_base": api_base,
            "custom_llm_provider": custom_llm_provider,
            "extra_headers": extra_headers,
            "optional_params": optional_params,
            "timeout_seconds": timeout_seconds,
        }
        for guardrail in guardrails or []:
            await guardrail.async_pre_call_hook(
                user_api_key_dict=None,
                cache=None,
                data=data,
                call_type="ocr",
            )

        response_obj = {
            "object": "ocr",
            "value": FAKE_RUST_OCR_RESPONSE,
        }
        standard_logging_payload: StandardLoggingPayload = {  # type: ignore[assignment]
            "model": model,
            "custom_llm_provider": custom_llm_provider,
            "call_type": "ocr",
            "response_cost": 0.0,
            "hidden_params": {"litellm_rust": True},
        }
        kwargs = {
            "api_base": api_base,
            "document": document,
            "optional_params": data["optional_params"],
            "standard_logging_object": standard_logging_payload,
        }
        start_time = datetime.fromtimestamp(0)
        end_time = datetime.fromtimestamp(1)
        for callback in callbacks or []:
            await callback.async_log_success_event(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
            )

        self.calls.append(data)
        return FAKE_RUST_OCR_RESPONSE


@pytest.fixture(autouse=True)
def reset_litellm_rust_callbacks():
    litellm.logging_callback_manager._reset_all_callbacks()
    litellm.use_litellm_rust(False, ocr=None, aocr=None)
    yield
    litellm.logging_callback_manager._reset_all_callbacks()
    litellm.use_litellm_rust(False, ocr=None, aocr=None)


@pytest.mark.asyncio
async def test_rust_ocr_executes_custom_logger_from_callback_manager():
    bridge = MockExecutingRustAocrBridge()
    custom_logger = OCRCustomLogger()
    litellm.logging_callback_manager.add_litellm_callback(custom_logger)
    litellm.use_litellm_rust(True, aocr=bridge)

    response = await litellm.aocr(
        model=MODEL,
        document=DOCUMENT,
        api_key="sk-test",
    )

    assert len(response.pages) > 0
    assert response._hidden_params["litellm_rust"] is True
    assert custom_logger.response_obj["object"] == "ocr"
    assert len(custom_logger.response_obj["value"]["pages"]) > 0
    assert isinstance(custom_logger.start_time, datetime)
    assert isinstance(custom_logger.end_time, datetime)
    assert custom_logger.kwargs["api_base"] == "https://api.mistral.ai/v1"
    assert custom_logger.kwargs["document"] == DOCUMENT
    assert custom_logger.kwargs["optional_params"] == {}
    assert bridge.calls[0]["custom_llm_provider"] == "mistral"

    logged_payload = custom_logger.standard_logging_payload
    assert logged_payload is not None
    assert logged_payload["model"] == "mistral-ocr-latest"
    assert logged_payload["custom_llm_provider"] == "mistral"
    assert logged_payload["call_type"] == "ocr"
    assert logged_payload["response_cost"] == 0.0
    assert logged_payload["hidden_params"]["litellm_rust"] is True


@pytest.mark.asyncio
async def test_rust_ocr_executes_custom_guardrail_from_callback_manager():
    bridge = MockExecutingRustAocrBridge()
    custom_guardrail = OCRCustomGuardrail()
    litellm.logging_callback_manager.add_litellm_callback(custom_guardrail)
    litellm.use_litellm_rust(True, aocr=bridge)

    response = await litellm.aocr(
        model=MODEL,
        document=DOCUMENT,
        api_key="sk-test",
    )

    assert len(response.pages) > 0
    assert response._hidden_params["litellm_rust"] is True
    assert custom_guardrail.calls[0]["call_type"] == "ocr"
    assert custom_guardrail.calls[0]["data"]["guardrail_executed"] is True
    assert (
        custom_guardrail.calls[0]["data"]["optional_params"]["include_image_base64"]
        is True
    )
    assert custom_guardrail.success_log_calls == 1
    assert bridge.calls[0]["optional_params"]["include_image_base64"] is True
