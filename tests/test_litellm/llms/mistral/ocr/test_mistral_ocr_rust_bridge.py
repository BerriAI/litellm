import httpx

from litellm.llms.mistral.ocr.transformation import MistralOCRConfig
from litellm.rust_bridge import loader
from litellm.rust_bridge import ocr as rust_ocr
from litellm.rust_bridge.ocr import providers
from litellm.rust_bridge.ocr import RustOcrProvider

MODEL = "mistral-ocr-latest"
DOCUMENT = {
    "type": "document_url",
    "document_url": "https://example.com/doc.pdf",
}


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
                if key != "unsupported_param"
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


class _FakeLoggingObj:
    pass


def test_rust_ocr_provider_enum_is_explicit():
    assert providers.RUST_OCR_PROVIDERS == {RustOcrProvider.MISTRAL.value}


def test_unknown_ocr_provider_uses_python_fallback():
    fallback_config = MistralOCRConfig()

    config = rust_ocr.get_rust_ocr_provider_config("azure_ai", fallback_config)

    assert config is fallback_config


def test_rust_ocr_provider_returns_none_when_scope_disabled(monkeypatch):
    monkeypatch.delenv("LITELLM_USE_RUST_CORE", raising=False)
    monkeypatch.setattr(loader, "_load_rust_module", lambda: _FakeRustModule)

    assert (
        providers.call_ocr(
            {
                "provider": RustOcrProvider.MISTRAL.value,
                "operation": "map_params",
                "non_default_params": {"extract_header": True},
            }
        )
        is None
    )


def test_mistral_ocr_map_params_uses_provider_gated_rust(monkeypatch):
    monkeypatch.setenv("LITELLM_USE_RUST_CORE", "ocr:mistral")
    monkeypatch.setattr(loader, "_load_rust_module", lambda: _FakeRustModule)

    result = providers.call_ocr(
        {
            "provider": RustOcrProvider.MISTRAL.value,
            "operation": "map_params",
            "non_default_params": {
                "extract_header": True,
                "unsupported_param": "value",
            },
        },
    )

    assert result == {"extract_header": True}


def test_mistral_ocr_provider_wrapper_uses_rust_when_enabled(monkeypatch):
    monkeypatch.setenv("LITELLM_USE_RUST_CORE", "ocr:mistral")
    monkeypatch.setattr(loader, "_load_rust_module", lambda: _FakeRustModule)

    config = rust_ocr.get_rust_ocr_provider_config("mistral", MistralOCRConfig())

    request = config.transform_ocr_request(
        model=MODEL,
        document=DOCUMENT,
        optional_params={"include_image_base64": True},
        headers={},
    )
    assert request.data == {
        "model": MODEL,
        "document": DOCUMENT,
        "include_image_base64": True,
    }
    assert request.files is None

    response = config.transform_ocr_response(
        model=MODEL,
        raw_response=httpx.Response(
            200,
            json={
                "pages": [{"index": 0, "markdown": "hello"}],
                "model": "mistral-ocr-2505-completion",
                "document_annotation": None,
                "usage_info": {"pages_processed": 1},
            },
        ),
        logging_obj=_FakeLoggingObj(),
    )

    assert response.pages[0].index == 0
    assert response.model == "mistral-ocr-2505-completion"
    assert response.usage_info.pages_processed == 1


def test_mistral_ocr_provider_wrapper_falls_back_when_rust_module_missing(
    monkeypatch,
):
    monkeypatch.setenv("LITELLM_USE_RUST_CORE", "ocr:mistral")
    monkeypatch.setattr(loader, "_load_rust_module", lambda: None)

    config = rust_ocr.get_rust_ocr_provider_config("mistral", MistralOCRConfig())

    result = config.transform_ocr_request(
        model=MODEL,
        document=DOCUMENT,
        optional_params={"include_image_base64": True},
        headers={},
    )

    assert result.data == {
        "model": MODEL,
        "document": DOCUMENT,
        "include_image_base64": True,
    }


def test_mistral_ocr_config_stays_python_fallback(monkeypatch):
    monkeypatch.setenv("LITELLM_USE_RUST_CORE", "ocr:mistral")
    monkeypatch.setattr(loader, "_load_rust_module", lambda: _FakeRustModule)

    config = MistralOCRConfig()
    result = config.map_ocr_params(
        non_default_params={
            "extract_header": True,
            "unsupported_param": "value",
        },
        optional_params={},
        model=MODEL,
    )

    assert result == {"extract_header": True}
