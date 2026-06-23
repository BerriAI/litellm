import importlib

import httpx
import pytest

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.llms.mistral.ocr.rust_provider import (
    MistralRustOcrProvider,
    get_mistral_rust_ocr_provider,
)
from litellm.rust_bridge import loader
from litellm.rust_bridge.ocr import providers

ocr_main = importlib.import_module("litellm.ocr.main")

MODEL = "mistral-ocr-latest"
SUPPORTED_PARAMS = {
    "pages",
    "include_image_base64",
    "image_limit",
    "image_min_size",
    "bbox_annotation_format",
    "document_annotation_format",
    "document_annotation_prompt",
    "extract_header",
    "extract_footer",
    "table_format",
    "confidence_scores_granularity",
    "id",
}
DOCUMENT = {
    "type": "document_url",
    "document_url": "https://example.com/doc.pdf",
}


@pytest.fixture(autouse=True)
def reset_rust_bridge_state():
    loader.set_rust_core_enabled(False)
    loader.set_rust_core_strict(False)
    yield
    loader.set_rust_core_enabled(False)
    loader.set_rust_core_strict(False)


class _FakeRustModule:
    @staticmethod
    def ocr(payload):
        provider = payload["provider"]
        operation = payload["operation"]

        assert provider == "mistral"
        if operation == "map_params":
            return {
                key: value
                for key, value in payload["non_default_params"].items()
                if key in SUPPORTED_PARAMS
            }
        if operation == "transform_request":
            return {
                "data": {
                    "model": payload["model"],
                    "document": payload["document"],
                    **payload["optional_params"],
                },
                "files": None,
            }
        if operation == "transform_response":
            response_json = payload["response_json"]
            return {
                "pages": response_json.get("pages", []),
                "model": response_json.get("model", payload["model"]),
                "document_annotation": response_json.get("document_annotation"),
                "usage_info": response_json.get("usage_info"),
                "object": "ocr",
            }
        raise AssertionError(f"Unexpected operation: {operation}")


class _FakeHTTPClient:
    def __init__(self):
        self.requests = []

    def post(self, url, headers, json, timeout):
        self.requests.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return httpx.Response(
            200,
            json={
                "pages": [{"index": 0, "markdown": "hello"}],
                "model": "mistral-ocr-2505-completion",
                "document_annotation": None,
                "usage_info": {"pages_processed": 1},
            },
        )


def test_mistral_rust_ocr_provider_enum_is_owned_by_mistral():
    assert MistralRustOcrProvider.MISTRAL.value == "mistral"
    assert get_mistral_rust_ocr_provider("mistral") == "mistral"
    assert get_mistral_rust_ocr_provider("azure_ai") is None


def test_rust_ocr_provider_returns_none_when_scope_disabled(monkeypatch):
    monkeypatch.setattr(loader, "_load_rust_module", lambda: _FakeRustModule)

    assert (
        providers.call_ocr(
            {
                "provider": MistralRustOcrProvider.MISTRAL.value,
                "operation": "map_params",
                "non_default_params": {"extract_header": True},
            }
        )
        is None
    )


def test_mistral_ocr_map_params_uses_provider_gated_rust(monkeypatch):
    loader.set_rust_core_enabled("ocr:mistral")
    monkeypatch.setattr(loader, "_load_rust_module", lambda: _FakeRustModule)

    result = providers.call_ocr(
        {
            "provider": MistralRustOcrProvider.MISTRAL.value,
            "operation": "map_params",
            "non_default_params": {
                "extract_header": True,
                "unsupported_param": "value",
            },
        },
    )

    assert result == {"extract_header": True}


def test_litellm_rust_ocr_calls_rust_bridge(monkeypatch):
    fake_client = _FakeHTTPClient()
    monkeypatch.setattr(loader, "_load_rust_module", lambda: _FakeRustModule)
    monkeypatch.setattr(ocr_main, "_get_httpx_client", lambda: fake_client)

    response = litellm.rust_ocr(
        model="mistral/mistral-ocr-latest",
        document=DOCUMENT,
        api_key="test-key",
        pages=[0],
        include_image_base64=False,
        unsupported_param="drop",
    )

    assert response.pages[0].index == 0
    assert response.model == "mistral-ocr-2505-completion"
    assert response.usage_info.pages_processed == 1
    assert len(fake_client.requests) == 1
    request = fake_client.requests[0]
    assert request["url"] == "https://api.mistral.ai/v1/ocr"
    assert request["headers"] == {"Authorization": "Bearer test-key"}
    assert request["json"] == {
        "model": MODEL,
        "document": DOCUMENT,
        "pages": [0],
        "include_image_base64": False,
    }


def test_litellm_ocr_routes_to_rust_when_mistral_scope_enabled(monkeypatch):
    fake_client = _FakeHTTPClient()
    loader.set_rust_core_enabled("ocr:mistral")
    monkeypatch.setattr(loader, "_load_rust_module", lambda: _FakeRustModule)
    monkeypatch.setattr(ocr_main, "_get_httpx_client", lambda: fake_client)

    response = litellm.ocr(
        model="mistral/mistral-ocr-latest",
        document=DOCUMENT,
        api_key="test-key",
        pages=[0],
        include_image_base64=False,
    )

    assert response.pages[0].markdown == "hello"
    assert len(fake_client.requests) == 1


def test_litellm_ocr_uses_python_path_when_rust_scope_disabled(monkeypatch):
    fake_client = _FakeHTTPClient()
    monkeypatch.setattr(ocr_main, "_get_httpx_client", lambda: fake_client)
    monkeypatch.setattr(
        ocr_main.base_llm_http_handler,
        "ocr",
        lambda **kwargs: OCRResponse(
            pages=[{"index": 0, "markdown": "python"}],
            model="mistral-ocr-2505-completion",
            document_annotation=None,
            usage_info={"pages_processed": 1},
            object="ocr",
        ),
    )

    response = litellm.ocr(
        model="mistral/mistral-ocr-latest",
        document=DOCUMENT,
        api_key="test-key",
        pages=[0],
        include_image_base64=False,
    )

    assert response.pages[0].markdown == "python"
    assert fake_client.requests == []
