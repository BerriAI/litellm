from __future__ import annotations

from typing import Final

import pytest

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRPage, OCRResponse, OCRUsageInfo
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.router import Router
from litellm.utils import CustomLogger


class _RecordingLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.calls: Final[list[tuple[dict, object]]] = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        self.calls.append((kwargs.get("standard_logging_object") or {}, response_obj))


@pytest.mark.asyncio
async def test_router_aocr_resolves_deployment_and_returns_ocr_response(monkeypatch):
    recorder: Final = _RecordingLogger()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    monkeypatch.setattr(litellm, "success_callback", [])
    monkeypatch.setattr(litellm, "failure_callback", [])
    captured: Final[list[dict]] = []

    async def _fake_native_call(*args: object, **kwargs: object) -> OCRResponse:
        captured.append(dict(kwargs))
        return OCRResponse(
            model="mistral-ocr-latest",
            pages=[OCRPage(index=0, markdown="Test PDF File")],
            usage_info=OCRUsageInfo(pages_processed=1, doc_size_bytes=1024),
        )

    import sys
    monkeypatch.setattr(sys.modules["litellm.ocr"], "aocr", _fake_native_call)
    router: Final = Router(
        model_list=[
            {
                "model_name": "ocr-alias",
                "litellm_params": {
                    "model": "mistral/mistral-ocr-latest",
                    "api_key": "fake-mistral-key",
                },
            }
        ]
    )
    response: Final = await router.aocr(
        model="ocr-alias",
        document={"type": "document_url", "document_url": "https://example.com/doc.pdf"},
    )
    assert captured and captured[0].get("model") == "mistral/mistral-ocr-latest"
    assert captured[0].get("api_key") == "fake-mistral-key"
    assert isinstance(response, OCRResponse)
    assert response.object == "ocr"
    assert [page.index for page in response.pages] == [0]
    assert "test pdf file" in " ".join(page.markdown for page in response.pages).lower()
    await GLOBAL_LOGGING_WORKER.clear_queue()
    if recorder.calls:
        payload, _ = recorder.calls[0]
        assert payload.get("status") == "success"
        assert payload.get("call_type") == "aocr"
        assert payload.get("custom_llm_provider") == "mistral"
