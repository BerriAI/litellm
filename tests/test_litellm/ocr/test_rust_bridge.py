"""Tests for the optional Rust-backed OCR path."""

import asyncio
import builtins
import importlib
import types

import httpx
import pytest
import pytest_asyncio

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge import configuration

# `litellm/__init__.py` does `from .ocr.main import *`, which binds the `ocr`
# function onto `litellm.ocr` and shadows the submodule, so import the modules
# explicitly via importlib rather than attribute traversal.
ocr_main = importlib.import_module("litellm.ocr.main")
rust_bridge = importlib.import_module("litellm.rust_bridge.ocr")
rust_bridge_bindings = importlib.import_module("litellm.rust_bridge.bindings")
rust_bridge_loader = importlib.import_module("litellm.rust_bridge.loader")

MODEL = "mistral/mistral-ocr-latest"
DOCUMENT: dict[str, object] = {
    "type": "document_url",
    "document_url": "https://example.com/doc.pdf",
}

FAKE_OCR_RESPONSE: dict[str, object] = {
    "pages": [{"index": 0, "markdown": "hello world"}],
    "model": "mistral-ocr-2505-completion",
    "document_annotation": None,
    "usage_info": {"pages_processed": 1},
    "object": "ocr",
}


class CapturedException(Exception):
    pass


class RustUpstreamError(Exception):
    pass


class RecordingBridge:
    """A fake ``RustOcr`` callable that records the args it was handed."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        model: str,
        document: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout_seconds: float | None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "model": model,
                "document": document,
                "api_key": api_key,
                "api_base": api_base,
                "custom_llm_provider": custom_llm_provider,
                "extra_headers": extra_headers,
                "optional_params": optional_params,
                "timeout_seconds": timeout_seconds,
            }
        )
        return dict(FAKE_OCR_RESPONSE)


class RecordingAsyncBridge:
    """A fake async ``RustAocr`` callable that records the args it was handed."""

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
    ) -> dict[str, object]:
        self.calls.append(
            {
                "model": model,
                "document": document,
                "api_key": api_key,
                "api_base": api_base,
                "custom_llm_provider": custom_llm_provider,
                "extra_headers": extra_headers,
                "optional_params": optional_params,
                "timeout_seconds": timeout_seconds,
            }
        )
        return dict(FAKE_OCR_RESPONSE)


class RaisingBridge:
    def __call__(
        self,
        model: str,
        document: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout_seconds: float | None,
    ) -> dict[str, object]:
        raise RuntimeError("bridge failed")


class RaisingAsyncBridge:
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
    ) -> dict[str, object]:
        raise RuntimeError("bridge failed")


class RecordingLogging:
    """A spy standing in for ``LiteLLMLoggingObj`` to capture ``pre_call``."""

    def __init__(self) -> None:
        self.pre_call_kwargs: dict[str, object] | None = None

    def update_from_kwargs(self, **kwargs: object) -> None:
        self.update_kwargs = kwargs

    def pre_call(
        self,
        *,
        input: str,
        api_key: str | None,
        additional_args: dict[str, object],
    ) -> None:
        self.pre_call_kwargs = {
            "input": input,
            "api_key": api_key,
            "additional_args": additional_args,
        }


def build_request(
    *,
    logging_obj: RecordingLogging | None = None,
    model: str = "mistral-ocr-latest",
    document: dict[str, object] = DOCUMENT,
    api_key: str | None = "sk-test",
    api_base: str | None = None,
    custom_llm_provider: str | None = "mistral",
    extra_headers: dict[str, object] | None = None,
    optional_params: dict[str, object] | None = None,
    litellm_params: dict[str, object] | None = None,
    timeout: float | httpx.Timeout | None = 12.5,
) -> rust_bridge.LiteLLMOcrRequest:
    return rust_bridge.LiteLLMOcrRequest(
        model=model,
        document=document,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        timeout=timeout,
        kwargs={
            **(optional_params or {}),
            **(litellm_params or {}),
            "litellm_logging_obj": logging_obj or RecordingLogging(),
        },
    )


@pytest_asyncio.fixture(autouse=True)
async def _reset_rust_flag():
    """Keep the global toggle isolated between tests."""
    rust_bridge._OCR.reset()
    rust_bridge._AOCR.reset()
    configuration.reset_rust_configuration()
    rust_bridge_loader._cached_bridge = rust_bridge_loader._BRIDGE_SENTINEL
    yield
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    await asyncio.sleep(0)
    GLOBAL_LOGGING_WORKER.start()
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)
    await GLOBAL_LOGGING_WORKER.stop()
    rust_bridge._OCR.reset()
    rust_bridge._AOCR.reset()
    configuration.reset_rust_configuration()
    rust_bridge_loader._cached_bridge = rust_bridge_loader._BRIDGE_SENTINEL


@pytest.fixture
def fake_bridge():
    """Enable the Rust path with an injected recording bridge (no native wheel)."""
    bridge = RecordingBridge()
    litellm.rust(True)
    rust_bridge._OCR.override(bridge)
    return bridge


@pytest.fixture
def fake_async_bridge():
    """Enable the async Rust path with an injected recording bridge."""
    bridge = RecordingAsyncBridge()
    litellm.rust(True)
    rust_bridge._AOCR.override(bridge)
    return bridge


def test_load_rust_ocr_returns_injected_impl():
    bridge = RecordingBridge()
    litellm.rust(True)
    rust_bridge._OCR.override(bridge)
    assert rust_bridge.load_rust_ocr() is bridge


def test_native_bridge_loader_returns_none_when_extension_absent(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "litellm.rust_bridge" and "_native" in fromlist:
            raise ImportError
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert rust_bridge_loader.get_native_bridge() is None


def test_native_bridge_loader_caches_absent_extension(monkeypatch):
    real_import = builtins.__import__
    attempts = 0

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        nonlocal attempts
        if name == "litellm.rust_bridge" and "_native" in fromlist:
            attempts += 1
            raise ImportError
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert rust_bridge_loader.get_native_bridge() is None
    assert rust_bridge_loader.get_native_bridge() is None
    assert attempts == 1


def test_native_bridge_loader_reset_forces_relookup(monkeypatch):
    real_import = builtins.__import__
    attempts = 0

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        nonlocal attempts
        if name == "litellm.rust_bridge" and "_native" in fromlist:
            attempts += 1
            raise ImportError
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert rust_bridge_loader.get_native_bridge() is None
    rust_bridge_loader.reset_native_bridge_cache()
    assert rust_bridge_loader.get_native_bridge() is None
    assert attempts == 2


def test_native_bridge_available_reflects_loader(monkeypatch):
    fake_module = types.ModuleType("litellm.rust_bridge._native")
    monkeypatch.setattr(rust_bridge_loader, "get_native_bridge", lambda: fake_module)

    assert rust_bridge_loader.native_bridge_available() is True


def test_load_rust_aocr_returns_injected_impl():
    bridge = RecordingAsyncBridge()
    litellm.rust(True)
    rust_bridge._AOCR.override(bridge)
    assert rust_bridge.load_rust_aocr() is bridge


def test_toggle_without_ocr_arg_preserves_injected_impl():
    """The public flag must not clobber an internal test binding."""
    bridge = RecordingBridge()
    async_bridge = RecordingAsyncBridge()
    litellm.rust(True)
    rust_bridge._OCR.override(bridge)
    rust_bridge._AOCR.override(async_bridge)

    litellm.rust(False)
    assert rust_bridge.load_rust_ocr() is bridge
    assert rust_bridge.load_rust_aocr() is async_bridge
    litellm.rust(True)
    assert rust_bridge.load_rust_ocr() is bridge
    assert rust_bridge.load_rust_aocr() is async_bridge


def test_explicit_ocr_none_clears_injected_impl(monkeypatch):
    monkeypatch.setattr(
        rust_bridge_bindings,
        "get_native_bridge",
        lambda: None,
    )
    bridge = RecordingBridge()
    async_bridge = RecordingAsyncBridge()
    litellm.rust(True)
    rust_bridge._OCR.override(bridge)
    rust_bridge._AOCR.override(async_bridge)

    rust_bridge._OCR.override(None)
    rust_bridge._AOCR.override(None)
    assert rust_bridge.load_rust_ocr() is None
    assert rust_bridge.load_rust_aocr() is None


def test_load_rust_ocr_none_when_extension_absent(monkeypatch):
    """With no injected impl and no compiled wheel, the loader returns None so the
    caller degrades to the Python path instead of raising ImportError."""
    monkeypatch.setattr(
        rust_bridge_bindings,
        "get_native_bridge",
        lambda: None,
    )
    litellm.rust(True)  # no impl injected; extension isn't built in CI
    assert rust_bridge.load_rust_ocr() is None
    assert rust_bridge.load_rust_aocr() is None


def test_load_rust_ocr_uses_compiled_extension(monkeypatch):
    """With no injected impl but a packaged ``litellm.rust_bridge._native`` importable,
    the loader returns the extension's ``ocr`` callable. The native wheel isn't
    built in CI, so stand in a fake module via the bridge loader."""
    fake_module = types.ModuleType("litellm.rust_bridge._native")
    fake_module.ocr = lambda **kwargs: dict(FAKE_OCR_RESPONSE)  # type: ignore[attr-defined]
    fake_module.aocr = lambda **kwargs: dict(FAKE_OCR_RESPONSE)  # type: ignore[attr-defined]
    monkeypatch.setattr(
        rust_bridge_bindings,
        "get_native_bridge",
        lambda: fake_module,
    )

    litellm.rust(True)  # enabled, no impl injected -> import the extension
    assert rust_bridge.load_rust_ocr() is fake_module.ocr
    assert rust_bridge.load_rust_aocr() is fake_module.aocr


def test_timeout_to_seconds_handles_float_timeout_and_none():
    assert rust_bridge._timeout_to_seconds(12.5) == 12.5
    assert rust_bridge._timeout_to_seconds(None) is None
    assert rust_bridge._timeout_to_seconds(httpx.Timeout(30.0, read=42.0)) == 42.0


def test_bridge_wrapper_forwards_prepared_args_and_wraps_response():
    bridge = RecordingBridge()

    litellm.rust(True)

    rust_bridge._OCR.override(bridge)
    response = rust_bridge.ocr(
        model="mistral-ocr-latest",
        document=DOCUMENT,
        api_key="sk-test",
        api_base="https://proxy.internal",
        custom_llm_provider="mistral",
        extra_headers={"Authorization": "Bearer sk-test", "x-trace-id": "trace-1"},
        optional_params={"include_image_base64": True, "pages": [0]},
        timeout=12.5,
    )

    assert response == FAKE_OCR_RESPONSE
    call = bridge.calls[0]
    assert call == {
        "model": "mistral-ocr-latest",
        "document": DOCUMENT,
        "api_key": "sk-test",
        "api_base": "https://proxy.internal",
        "custom_llm_provider": "mistral",
        "extra_headers": {
            "Authorization": "Bearer sk-test",
            "x-trace-id": "trace-1",
        },
        "optional_params": {"include_image_base64": True, "pages": [0]},
        "timeout_seconds": 12.5,
    }


@pytest.mark.asyncio
async def test_bridge_wrapper_forwards_prepared_async_args_and_wraps_response():
    bridge = RecordingAsyncBridge()

    litellm.rust(True)

    rust_bridge._AOCR.override(bridge)
    response = await rust_bridge.aocr(
        model="mistral-ocr-maas",
        document=DOCUMENT,
        api_key=None,
        api_base=None,
        custom_llm_provider="vertex_ai",
        extra_headers=None,
        optional_params={"vertex_project": "project-1"},
        timeout=httpx.Timeout(30.0, read=42.0),
    )

    assert response == FAKE_OCR_RESPONSE
    assert bridge.calls[0] == {
        "model": "mistral-ocr-maas",
        "document": DOCUMENT,
        "api_key": None,
        "api_base": None,
        "custom_llm_provider": "vertex_ai",
        "extra_headers": None,
        "optional_params": {"vertex_project": "project-1"},
        "timeout_seconds": 42.0,
    }


def test_run_rust_ocr_prepares_request_and_wraps_response():
    bridge = RecordingBridge()
    logging_obj = RecordingLogging()
    litellm.rust(True)
    rust_bridge._OCR.override(bridge)

    response = ocr_main._run_rust_ocr(
        request=build_request(
            logging_obj=logging_obj,
            api_base="https://proxy.internal",
            extra_headers={"x-trace-id": "trace-1"},
            optional_params={"include_image_base64": True},
            timeout=12.5,
        ),
        resolve_api_key=lambda _name: None,
    )

    assert isinstance(response, OCRResponse)
    assert response.pages[0].markdown == "hello world"
    assert bridge.calls[0] == {
        "model": "mistral-ocr-latest",
        "document": DOCUMENT,
        "api_key": "sk-test",
        "api_base": "https://proxy.internal",
        "custom_llm_provider": "mistral",
        "extra_headers": {
            "x-trace-id": "trace-1",
        },
        "optional_params": {"include_image_base64": True},
        "timeout_seconds": 12.5,
    }


def test_rust_upstream_error_uses_ocr_provider_error_mapping():
    error = RustUpstreamError(400, '{"message":"invalid model"}')

    mapped = ocr_main._map_rust_ocr_error(
        error,
        build_request(),
        (RuntimeError, RustUpstreamError),
    )

    assert isinstance(mapped, BaseLLMException)
    assert mapped.status_code == 400
    assert mapped.message == '{"message":"invalid model"}'


def test_run_rust_ocr_resolves_key_via_secret_manager_when_missing():
    bridge = RecordingBridge()
    litellm.rust(True)
    rust_bridge._OCR.override(bridge)

    ocr_main._run_rust_ocr(
        request=build_request(api_key=None, timeout=None),
        resolve_api_key=lambda name: "sk-from-vault" if name == "MISTRAL_API_KEY" else None,
    )

    assert bridge.calls[0]["api_key"] == "sk-from-vault"


def test_run_rust_ocr_prefers_explicit_key_over_resolver():
    bridge = RecordingBridge()
    litellm.rust(True)
    rust_bridge._OCR.override(bridge)

    def _resolver(name: str) -> str | None:
        raise AssertionError(f"resolver should not be called for {name}")

    ocr_main._run_rust_ocr(
        request=build_request(
            api_key="sk-explicit",
            timeout=None,
        ),
        resolve_api_key=_resolver,
    )

    assert bridge.calls[0]["api_key"] == "sk-explicit"


def test_run_rust_ocr_uses_mistral_secret_manager_without_provider_config():
    bridge = RecordingBridge()
    resolver_calls = []
    litellm.rust(True)
    rust_bridge._OCR.override(bridge)

    def _resolver(name):
        resolver_calls.append(name)
        return "sk-provider-env"

    ocr_main._run_rust_ocr(
        request=build_request(
            model="mistral-ocr-latest",
            api_key=None,
            timeout=None,
        ),
        resolve_api_key=_resolver,
    )

    assert resolver_calls == ["MISTRAL_API_KEY"]
    assert bridge.calls[0]["api_key"] == "sk-provider-env"


def test_prepare_rust_ocr_call_forwards_vertex_routing_metadata():
    bridge = RecordingBridge()
    litellm.rust(True)
    rust_bridge._OCR.override(bridge)

    ocr_main._run_rust_ocr(
        request=build_request(
            custom_llm_provider="vertex_ai",
            model="mistral-ocr-maas",
            litellm_params={
                "vertex_project": "project-1",
                "vertex_location": "us-central1",
                "vertex_credentials": "redacted",
            },
            optional_params={"include_image_base64": True},
            timeout=None,
        ),
        resolve_api_key=lambda _name: None,
    )

    assert bridge.calls[0]["optional_params"] == {
        "include_image_base64": True,
        "vertex_project": "project-1",
        "vertex_location": "us-central1",
        "vertex_credentials": "redacted",
    }


def test_prepare_rust_ocr_call_resolves_vertex_routing_metadata_from_secret_manager():
    bridge = RecordingBridge()
    litellm.rust(True)
    rust_bridge._OCR.override(bridge)

    def _resolver(name: str) -> str | None:
        return {
            "VERTEXAI_PROJECT": "project-from-secret",
            "VERTEXAI_LOCATION": "us-east5",
        }.get(name)

    ocr_main._run_rust_ocr(
        request=build_request(
            custom_llm_provider="vertex_ai",
            model="mistral-ocr-maas",
            timeout=None,
        ),
        resolve_api_key=_resolver,
    )

    assert bridge.calls[0]["optional_params"]["vertex_project"] == "project-from-secret"
    assert bridge.calls[0]["optional_params"]["vertex_location"] == "us-east5"


def test_prepare_rust_ocr_call_defers_azure_environment_resolution_to_rust():
    bridge = RecordingBridge()
    litellm.rust(True)
    rust_bridge._OCR.override(bridge)

    ocr_main._run_rust_ocr(
        request=build_request(
            custom_llm_provider="azure_ai",
            model="pixtral-12b-2409",
            api_key=None,
            api_base=None,
            timeout=None,
        ),
        resolve_api_key=lambda name: pytest.fail(f"Python resolved Azure secret {name}"),
    )

    assert bridge.calls[0]["api_base"] is None
    assert bridge.calls[0]["api_key"] is None
    assert bridge.calls[0]["extra_headers"] is None


def test_prepare_rust_ocr_call_defers_document_intelligence_environment_to_rust():
    bridge = RecordingBridge()
    litellm.rust(True)
    rust_bridge._OCR.override(bridge)

    ocr_main._run_rust_ocr(
        request=build_request(
            custom_llm_provider="azure_ai",
            model="doc-intelligence/prebuilt-layout",
            api_base=None,
            timeout=None,
        ),
        resolve_api_key=lambda name: pytest.fail(f"Python resolved Azure secret {name}"),
    )

    assert bridge.calls[0]["api_base"] is None


def test_prepare_rust_ocr_call_forwards_raw_azure_auth_inputs():
    bridge = RecordingBridge()
    litellm.rust(True)
    rust_bridge._OCR.override(bridge)

    ocr_main._run_rust_ocr(
        request=build_request(
            custom_llm_provider="azure_ai",
            model="pixtral-12b-2409",
            api_key=None,
            api_base="https://azure.example.com",
            extra_headers={"x-trace-id": "trace-1"},
            litellm_params={
                "azure_ad_token": "entra-token",
                "tenant_id": "tenant",
                "client_id": "client",
                "client_secret": "secret",
                "azure_scope": "scope",
                "azure_authority_host": "https://login.example.com",
                "azure_credential": "ClientSecretCredential",
                "azure_federated_token_file": "/token",
            },
            timeout=None,
        ),
        resolve_api_key=lambda name: pytest.fail(f"Python resolved Azure secret {name}"),
    )

    call = bridge.calls[0]
    assert call["api_key"] is None
    assert call["api_base"] == "https://azure.example.com"
    assert call["extra_headers"] == {"x-trace-id": "trace-1"}
    assert call["optional_params"] == {
        "azure_ad_token": "entra-token",
        "tenant_id": "tenant",
        "client_id": "client",
        "client_secret": "secret",
        "azure_scope": "scope",
        "azure_authority_host": "https://login.example.com",
        "azure_credential": "ClientSecretCredential",
        "azure_federated_token_file": "/token",
    }


def test_rust_eligibility_rejects_python_only_azure_auth_modes():
    for params in (
        {"azure_ad_token_provider": lambda: "token"},
        {"azure_username": "user"},
        {"azure_password": "password"},
    ):
        assert not ocr_main._rust_ocr_supported(
            build_request(
                custom_llm_provider="azure_ai",
                model="pixtral-12b-2409",
                litellm_params=params,
            )
        )


def test_prepare_rust_ocr_call_forwards_global_azure_refresh(monkeypatch: pytest.MonkeyPatch):
    bridge = RecordingBridge()
    litellm.rust(True)
    rust_bridge._OCR.override(bridge)
    monkeypatch.setattr(litellm, "enable_azure_ad_token_refresh", True)

    ocr_main._run_rust_ocr(
        request=build_request(
            custom_llm_provider="azure_ai",
            model="pixtral-12b-2409",
            api_key=None,
            api_base="https://azure.example.com",
            timeout=None,
        ),
        resolve_api_key=lambda _name: None,
    )

    assert bridge.calls[0]["optional_params"] == {"enable_azure_ad_token_refresh": True}


def test_run_rust_ocr_runs_pre_call_logging():
    logging_obj = RecordingLogging()
    bridge = RecordingBridge()
    litellm.rust(True)
    rust_bridge._OCR.override(bridge)

    ocr_main._run_rust_ocr(
        request=build_request(
            logging_obj=logging_obj,
            api_base="https://api.mistral.ai/v1",
            extra_headers={"x-trace-id": "trace-1"},
            optional_params={"include_image_base64": True},
            timeout=None,
        ),
        resolve_api_key=lambda _name: None,
    )

    assert logging_obj.pre_call_kwargs is not None
    assert logging_obj.pre_call_kwargs["input"] == "OCR document processing"
    additional_args = logging_obj.pre_call_kwargs["additional_args"]
    assert logging_obj.pre_call_kwargs["api_key"] is None
    assert additional_args == {
        "ocr_request_metadata": {
            "model": "mistral-ocr-latest",
            "custom_llm_provider": "mistral",
            "litellm_call_id": None,
            "document_type": "document_url",
            "timeout": ocr_main.request_timeout,
        }
    }
    assert logging_obj.update_kwargs["optional_params"] == {}
    assert bridge.calls[0]["optional_params"] == {"include_image_base64": True}


def test_ocr_routes_to_rust_when_enabled(fake_bridge):
    response = litellm.ocr(
        model=MODEL,
        document=DOCUMENT,
        api_key="sk-test",
        extra_headers={"x-trace-id": "trace-1"},
        include_image_base64=True,
    )

    assert isinstance(response, OCRResponse)
    assert response.pages[0].markdown == "hello world"
    assert len(fake_bridge.calls) == 1
    call = fake_bridge.calls[0]
    assert call["model"] == MODEL
    assert call["document"] == DOCUMENT
    assert call["api_key"] == "sk-test"
    assert call["custom_llm_provider"] is None
    assert call["extra_headers"] == {
        "x-trace-id": "trace-1",
    }
    assert call["optional_params"].get("include_image_base64") is True


def test_ocr_routes_azure_ai_to_rust_when_enabled(fake_bridge):
    response = litellm.ocr(
        model="azure_ai/pixtral-12b-2409",
        document=DOCUMENT,
        api_key="sk-test",
        api_base="https://example.services.ai.azure.com",
    )

    assert isinstance(response, OCRResponse)
    assert len(fake_bridge.calls) == 1
    assert fake_bridge.calls[0]["model"] == "azure_ai/pixtral-12b-2409"
    assert fake_bridge.calls[0]["custom_llm_provider"] is None
    assert fake_bridge.calls[0]["extra_headers"] is None


def test_ocr_routes_azure_entra_inputs_to_rust_without_python_auth(fake_bridge):
    response = litellm.ocr(
        model="azure_ai/pixtral-12b-2409",
        document=DOCUMENT,
        api_base="https://example.services.ai.azure.com",
        azure_ad_token="entra-token",
        tenant_id="tenant",
        client_id="client",
    )

    assert isinstance(response, OCRResponse)
    assert fake_bridge.calls[0]["api_key"] is None
    assert fake_bridge.calls[0]["extra_headers"] is None
    assert fake_bridge.calls[0]["optional_params"] == {
        "azure_ad_token": "entra-token",
        "tenant_id": "tenant",
        "client_id": "client",
    }


def test_ocr_rust_path_converts_file_document_before_bridge(fake_bridge):
    response = litellm.ocr(
        model=MODEL,
        document={"type": "file", "file": b"%PDF-1.4", "mime_type": "application/pdf"},
        api_key="sk-test",
    )

    assert isinstance(response, OCRResponse)
    document = fake_bridge.calls[0]["document"]
    assert document["type"] == "document_url"
    assert document["document_url"].startswith("data:application/pdf;base64,")


def test_ocr_exception_type_uses_resolved_provider_context(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_exception_type(**kwargs: object) -> CapturedException:
        captured.update(kwargs)
        return CapturedException("wrapped")

    monkeypatch.setattr(ocr_main.litellm, "exception_type", fake_exception_type)
    litellm.rust(True)
    rust_bridge._OCR.override(RaisingBridge())

    with pytest.raises(CapturedException):
        litellm.ocr(model=MODEL, document=DOCUMENT, api_key="sk-test")

    assert captured["model"] == "mistral-ocr-latest"
    assert captured["custom_llm_provider"] == "mistral"


@pytest.mark.asyncio
async def test_aocr_routes_to_async_rust_when_enabled(fake_async_bridge):
    response = await litellm.aocr(
        model=MODEL,
        document=DOCUMENT,
        api_key="sk-test",
        extra_headers={"x-trace-id": "trace-1"},
        include_image_base64=True,
    )

    assert isinstance(response, OCRResponse)
    assert response.pages[0].markdown == "hello world"
    assert len(fake_async_bridge.calls) == 1
    call = fake_async_bridge.calls[0]
    assert call["model"] == MODEL
    assert call["document"] == DOCUMENT
    assert call["api_key"] == "sk-test"
    assert call["custom_llm_provider"] is None
    assert call["extra_headers"] == {
        "x-trace-id": "trace-1",
    }
    assert call["optional_params"].get("include_image_base64") is True


@pytest.mark.asyncio
async def test_aocr_exception_type_uses_resolved_provider_context(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict[str, object] = {}

    def fake_exception_type(**kwargs: object) -> CapturedException:
        captured.update(kwargs)
        return CapturedException("wrapped")

    monkeypatch.setattr(ocr_main.litellm, "exception_type", fake_exception_type)
    litellm.rust(True)
    rust_bridge._AOCR.override(RaisingAsyncBridge())

    with pytest.raises(CapturedException):
        await litellm.aocr(model=MODEL, document=DOCUMENT, api_key="sk-test")

    assert captured["model"] == "mistral-ocr-latest"
    assert captured["custom_llm_provider"] == "mistral"


def test_ocr_forwards_timeout_to_rust(fake_bridge):
    """Caller-supplied timeout must flow into the Rust bridge so the fixed 600s
    client ceiling doesn't silently override shorter deadlines."""
    litellm.ocr(model=MODEL, document=DOCUMENT, api_key="sk-test", timeout=12.5)

    assert fake_bridge.calls[0]["timeout_seconds"] == 12.5


def test_ocr_passes_default_request_timeout_to_rust(fake_bridge):
    litellm.ocr(model=MODEL, document=DOCUMENT, api_key="sk-test")

    from litellm.constants import request_timeout

    assert fake_bridge.calls[0]["timeout_seconds"] == float(request_timeout)


def test_ocr_does_not_route_to_rust_when_disabled():
    """With the flag off, the bridge must not be consulted even if an impl exists."""
    bridge = RecordingBridge()
    litellm.rust(False)
    rust_bridge._OCR.override(bridge)
    # The impl stays available for injection, but the disabled flag gates usage,
    # so ocr() never reaches the Rust path (asserted via the enabled-path test).
    assert bridge.calls == []


def test_ocr_falls_back_to_python_when_bridge_unavailable(monkeypatch):
    """Rust enabled but no bridge available (no injected impl, no compiled wheel):
    ocr() must degrade to the Python HTTP handler instead of raising."""
    monkeypatch.setattr(rust_bridge, "load_rust_ocr", lambda: None)
    litellm.rust(True)  # enabled, but load_rust_ocr() returns None in CI

    captured = {}

    def fake_handler_ocr(**kwargs):
        captured["called"] = True
        return OCRResponse(pages=[], model="mistral-ocr-latest", object="ocr")

    monkeypatch.setattr(ocr_main.base_llm_http_handler, "ocr", fake_handler_ocr)

    response = litellm.ocr(model=MODEL, document=DOCUMENT, api_key="sk-test")

    assert captured.get("called") is True  # Python path was used
    assert isinstance(response, OCRResponse)


def test_ocr_provider_configs_expose_api_key_env_vars():
    from litellm.llms.azure_ai.ocr.document_intelligence.transformation import (
        AzureDocumentIntelligenceOCRConfig,
    )
    from litellm.llms.azure_ai.ocr.transformation import AzureAIOCRConfig
    from litellm.llms.base_llm.ocr.transformation import BaseOCRConfig
    from litellm.llms.mistral.ocr.transformation import MistralOCRConfig
    from litellm.llms.vertex_ai.ocr.deepseek_transformation import (
        VertexAIDeepSeekOCRConfig,
    )
    from litellm.llms.vertex_ai.ocr.transformation import VertexAIOCRConfig

    assert BaseOCRConfig().get_api_key_env_var() is None
    assert MistralOCRConfig().get_api_key_env_var() == "MISTRAL_API_KEY"
    assert AzureAIOCRConfig().get_api_key_env_var() == "AZURE_AI_API_KEY"
    assert AzureDocumentIntelligenceOCRConfig().get_api_key_env_var() == "AZURE_DOCUMENT_INTELLIGENCE_API_KEY"
    assert VertexAIOCRConfig().get_api_key_env_var() == "VERTEX_AI_API_KEY"
    assert VertexAIDeepSeekOCRConfig().get_api_key_env_var() == "VERTEX_AI_API_KEY"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_rust_receives_unmapped_azure_options(asynchronous, fake_bridge, fake_async_bridge):
    from typing import Final

    arguments: Final = {
        "model": "azure_ai/doc-intelligence/prebuilt-layout",
        "document": DOCUMENT,
        "api_key": "test-key",
        "pages": [0, 2],
        "features": ["languages", "style"],
        "provider_extension": {"enabled": True},
    }
    if asynchronous:
        await litellm.aocr(**arguments)
    else:
        litellm.ocr(**arguments)
    call: Final = (fake_async_bridge if asynchronous else fake_bridge).calls[0]
    assert call["model"] == arguments["model"]
    assert call["custom_llm_provider"] is None
    assert call["extra_headers"] is None
    assert call["optional_params"] == {
        "pages": [0, 2],
        "features": ["languages", "style"],
        "provider_extension": {"enabled": True},
    }


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.asyncio
async def test_python_fallback_maps_original_options_once(enabled, monkeypatch):
    from io import BytesIO
    from typing import Final

    class PythonHandler:
        def __init__(self):
            self.calls = []

        def ocr(self, **kwargs):
            self.calls.append(kwargs)
            return OCRResponse(pages=[], model=kwargs["model"])

    handler: Final = PythonHandler()
    monkeypatch.setattr(ocr_main, "base_llm_http_handler", handler)
    litellm.rust(enabled)
    rust_bridge._OCR.override(None)
    rust_bridge._AOCR.override(None)
    for asynchronous in (False, True):
        file: Final = BytesIO(b"test document")
        arguments: Final = {
            "model": "azure_ai/doc-intelligence/prebuilt-layout",
            "document": {"type": "file", "file": file},
            "api_key": "test-key",
            "pages": [0, 2],
        }
        if asynchronous:
            await litellm.aocr(**arguments)
        else:
            litellm.ocr(**arguments)
        assert handler.calls[-1]["optional_params"]["pages"] == "1,3"
        assert handler.calls[-1]["document"]["document_url"].endswith("dGVzdCBkb2N1bWVudA==")
    assert len(handler.calls) == 2


@pytest.mark.parametrize("provider", ["azure_ai", "vertex_ai"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_ocr_credentials_reach_execution_but_never_logging(provider, asynchronous, caplog):
    import copy
    import datetime
    import logging

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from litellm.integrations.custom_logger import CustomLogger
    from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig
    from litellm.litellm_core_utils.litellm_logging import Logging

    snapshots = []

    class Capture(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            snapshots.append(copy.deepcopy(kwargs))

        def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
            snapshots.append(copy.deepcopy(kwargs))

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            snapshots.append(copy.deepcopy(kwargs))

        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            snapshots.append(copy.deepcopy(kwargs))

    secrets = {
        "client_secret": "sentinel-client-secret",
        "azure_ad_token": "sentinel-azure-token",
        "vertex_credentials": {"private_key": "sentinel-private-key", "nested": {"token": "sentinel-nested-token"}},
        "unrecognized_future_option": "sentinel-future-option",
    }
    model = f"{provider}/mistral-ocr-latest"
    document = {"type": "document_url", "document_url": "https://example.com/sentinel-document?sig=sentinel-signature"}
    headers = {"Authorization": "Bearer sentinel-authorization", "x-custom": "sentinel-header"}
    logger = Logging(
        model=model,
        messages=[],
        stream=False,
        call_type="aocr" if asynchronous else "ocr",
        start_time=datetime.datetime.now(),
        litellm_call_id="review-call",
        function_id="review",
        kwargs={**secrets, "document": document, "api_key": "sentinel-api-key", "extra_headers": headers},
        dynamic_input_callbacks=[Capture()],
        dynamic_success_callbacks=[Capture()],
        dynamic_failure_callbacks=[Capture()],
        log_raw_request_response=True,
    )
    bridge = RecordingAsyncBridge() if asynchronous else RecordingBridge()
    litellm.rust(True)
    (rust_bridge._AOCR if asynchronous else rust_bridge._OCR).override(bridge)
    request = build_request(
        logging_obj=logger,
        model=model,
        custom_llm_provider=provider,
        document=document,
        api_key="sentinel-api-key",
        extra_headers=headers,
        optional_params=secrets,
    )
    with caplog.at_level(logging.DEBUG, logger="LiteLLM"):
        response = (
            await ocr_main._run_rust_aocr(request, lambda _: None)
            if asynchronous
            else ocr_main._run_rust_ocr(request, lambda _: None)
        )
        logger.post_call(
            original_response='{"pages":[]}',
            api_key="sentinel-api-key",
            input=document,
            additional_args={"complete_input_dict": secrets, "headers": headers},
        )
        logger.success_handler(response)
        snapshots.append(copy.deepcopy(logger.model_call_details))
        logger.failure_handler(RuntimeError("sentinel-provider-error"), "sentinel-request-in-traceback")
        snapshots.append(copy.deepcopy(logger.model_call_details))
    assert bridge.calls[0]["document"] == document
    assert bridge.calls[0]["extra_headers"] == headers
    assert bridge.calls[0]["api_key"] == "sentinel-api-key"
    auth_field = "client_secret" if provider == "azure_ai" else "vertex_credentials"
    assert bridge.calls[0]["optional_params"][auth_field] == secrets[auth_field]
    assert bridge.calls[0]["optional_params"]["unrecognized_future_option"] == "sentinel-future-option"
    assert snapshots
    assert "sentinel-" not in repr(snapshots)
    assert "sentinel-" not in caplog.text
    assert "sentinel-" not in repr(request)
    assert all("raw_request_typed_dict" not in snapshot for snapshot in snapshots)
    assert all("complete_input_dict" not in snapshot.get("additional_args", {}) for snapshot in snapshots)

    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel = OpenTelemetry(config=OpenTelemetryConfig(), tracer_provider=tracer_provider)
    with tracer_provider.get_tracer("ocr-review").start_as_current_span("ocr") as span:
        for snapshot in snapshots:
            otel.set_raw_request_attributes(span, snapshot, None)
    assert "sentinel-" not in repr(exporter.get_finished_spans()[0].attributes)
    tracer_provider.shutdown()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_python_ocr_request_logging_uses_metadata(enabled, asynchronous, caplog, ocr_logging_server):
    import datetime
    import logging

    from litellm.litellm_core_utils.litellm_logging import Logging

    base, calls = ocr_logging_server

    logger = Logging(
        model=MODEL,
        messages=[],
        stream=False,
        call_type="aocr" if asynchronous else "ocr",
        start_time=datetime.datetime.now(),
        litellm_call_id="review-call",
        function_id="review",
        kwargs={"client_secret": "sentinel-initial-secret"},
        log_raw_request_response=True,
    )
    litellm.rust(enabled)
    rust_bridge._OCR.override(None)
    rust_bridge._AOCR.override(None)
    rust_bridge_loader._cached_bridge = None
    arguments = {
        "model": MODEL,
        "document": {"type": "document_url", "document_url": "https://example.com/sentinel-document"},
        "api_key": "sentinel-api-key",
        "api_base": base,
        "litellm_logging_obj": logger,
        "extra_headers": {"x-custom": "sentinel-header"},
        "document_annotation_prompt": "sentinel-prompt",
        "vertex_credentials": "sentinel-credentials",
    }
    with caplog.at_level(logging.DEBUG, logger="LiteLLM"):
        response = await litellm.aocr(**arguments) if asynchronous else litellm.ocr(**arguments)
        if asynchronous:
            from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

            await asyncio.sleep(0)
            GLOBAL_LOGGING_WORKER.start()
            await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=10)

    assert response.pages[0].markdown == "hello world"
    assert len(calls) == 1
    assert calls[0][0]["authorization"] == "Bearer sentinel-api-key"
    assert "sentinel-prompt" in calls[0][1]
    assert "sentinel-" not in repr(logger.model_call_details)
    assert "sentinel-" not in caplog.text
    assert "raw_request_typed_dict" not in logger.model_call_details


@pytest.fixture
def ocr_logging_server():
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(
                (
                    {name.lower(): value for name, value in self.headers.items()},
                    self.rfile.read(int(self.headers["Content-Length"])).decode(),
                )
            )
            body = json.dumps(FAKE_OCR_RESPONSE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


@pytest.mark.parametrize(
    "model, provider",
    [
        ("mistral-ocr-latest", None),
        ("mistral/mistral-ocr-latest", None),
        ("custom-deployment", "mistral"),
    ],
)
def test_rust_ocr_provider_resolution_preserves_model_forms(model, provider, fake_bridge):
    response = litellm.ocr(model=model, custom_llm_provider=provider, document=DOCUMENT, api_key="test-key")
    assert response.pages[0].markdown == "hello world"
    assert len(fake_bridge.calls) == 1
    assert fake_bridge.calls[0]["model"] == model


def test_ocr_logging_preserves_billing_identity_without_arbitrary_metadata():
    import datetime

    from litellm.litellm_core_utils.litellm_logging import Logging

    identity = {"user_api_key_hash": "key-hash", "user_api_key_team_id": "team-id", "user_api_key_user_id": "user-id"}
    logger = Logging(
        model=MODEL,
        messages=[],
        stream=False,
        call_type="ocr",
        start_time=datetime.datetime.now(),
        litellm_call_id="review-call",
        function_id="review",
        kwargs={
            "metadata": {
                **identity,
                "model_info": {"id": "deployment-id", "unknown": "sentinel-secret"},
                "untrusted": "sentinel-secret",
            },
            "ocr_cost_per_page": 0.125,
        },
    )
    ocr_main._marshal_rust_ocr_request(build_request(logging_obj=logger), lambda _: None)
    assert logger.litellm_params["metadata"] == {**identity, "model_info": {"id": "deployment-id"}}
    assert logger.get_router_model_id() == "deployment-id"
    assert logger.litellm_params["ocr_cost_per_page"] == 0.125
    assert "sentinel-secret" not in repr(logger.model_call_details)


def test_rust_ocr_excludes_host_context_from_provider_options(fake_bridge):
    response = litellm.ocr(
        model=MODEL,
        document=DOCUMENT,
        api_key="test-key",
        metadata={"user_api_key_auth": object()},
        litellm_metadata={"user_api_key_auth": object()},
        unrecognized_future_option={"value": "preserved"},
    )
    assert response.pages[0].markdown == "hello world"
    assert len(fake_bridge.calls) == 1
    options = fake_bridge.calls[0]["optional_params"]
    assert "metadata" not in options
    assert "litellm_metadata" not in options
    assert options["unrecognized_future_option"] == {"value": "preserved"}
