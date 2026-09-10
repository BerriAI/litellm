"""Tests for the optional Rust-backed OCR path."""

import builtins
import importlib
import types

import httpx
import pytest

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
        input_sources: dict[str, str],
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
                "input_sources": input_sources,
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
        input_sources: dict[str, str],
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
                "input_sources": input_sources,
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
        input_sources: dict[str, str],
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
        input_sources: dict[str, str],
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


@pytest.fixture(autouse=True)
def _reset_rust_flag():
    """Keep the global toggle isolated between tests."""
    rust_bridge._OCR.reset()
    rust_bridge._AOCR.reset()
    configuration.reset_rust_configuration()
    rust_bridge_loader._cached_bridge = rust_bridge_loader._BRIDGE_SENTINEL
    yield
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
        "input_sources": {},
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
        "input_sources": {},
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
        "input_sources": {},
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
    assert call["input_sources"] == {}


def test_prepare_rust_ocr_call_preserves_proxy_input_sources():
    bridge = RecordingBridge()
    litellm.rust(True)
    rust_bridge._OCR.override(bridge)
    request_values = {
        "tenant_id": "tenant",
        "client_id": "client",
        "client_secret": "secret",
        "azure_authority_host": "https://login.example.com",
        "api_base": "https://azure.example.com",
    }

    ocr_main._run_rust_ocr(
        request=build_request(
            custom_llm_provider="azure_ai",
            model="pixtral-12b-2409",
            api_key=None,
            api_base="https://azure.example.com",
            litellm_params={
                "tenant_id": "tenant",
                "client_id": "client",
                "client_secret": "secret",
                "azure_authority_host": "https://login.example.com",
                "proxy_server_request": {"body": request_values},
            },
        ),
        resolve_api_key=lambda _name: None,
    )

    assert bridge.calls[0]["input_sources"] == {name: "request" for name in request_values}


def test_rust_ocr_logging_redacts_azure_credentials():
    bridge = RecordingBridge()
    logging_obj = RecordingLogging()
    litellm.rust(True)
    rust_bridge._OCR.override(bridge)

    ocr_main._run_rust_ocr(
        request=build_request(
            logging_obj=logging_obj,
            custom_llm_provider="azure_ai",
            model="pixtral-12b-2409",
            api_key=None,
            litellm_params={"azure_ad_token": "token", "client_secret": "secret"},
        ),
        resolve_api_key=lambda _name: None,
    )

    assert logging_obj.update_kwargs["optional_params"] == {
        "azure_ad_token": "****",
        "client_secret": "****",
    }
    assert logging_obj.pre_call_kwargs is not None
    additional_args = logging_obj.pre_call_kwargs["additional_args"]
    assert isinstance(additional_args, dict)
    complete_input = additional_args["complete_input_dict"]
    assert isinstance(complete_input, dict)
    assert complete_input["azure_ad_token"] == "****"
    assert complete_input["client_secret"] == "****"


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
            litellm_params={"proxy_server_request": {"body": {"enable_azure_ad_token_refresh": True}}},
            timeout=None,
        ),
        resolve_api_key=lambda _name: None,
    )

    assert bridge.calls[0]["optional_params"] == {"enable_azure_ad_token_refresh": True}
    assert bridge.calls[0]["input_sources"] == {"enable_azure_ad_token_refresh": "deployment"}


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
    complete_input = additional_args["complete_input_dict"]
    assert complete_input["document"] == DOCUMENT
    assert complete_input["include_image_base64"] is True
    assert additional_args["api_base"] == "https://api.mistral.ai/v1"
    assert additional_args["headers"] == {
        "x-trace-id": "trace-1",
    }


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


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("model", ["mistral/mistral-ocr-latest", "azure_ai/doc-intelligence/prebuilt-read"])
@pytest.mark.asyncio
async def test_native_public_ocr_matches_python(model, asynchronous):
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread
    from typing import Final
    from urllib.parse import parse_qsl, urlsplit

    native: Final = rust_bridge_loader.get_native_bridge()
    if native is None:
        pytest.skip("requires the compiled Rust extension")
    calls: Final = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body: Final = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            target: Final = urlsplit(self.path)
            calls.append(
                (
                    target.path,
                    parse_qsl(target.query),
                    self.headers.get("Authorization"),
                    self.headers.get("Ocp-Apim-Subscription-Key"),
                    body,
                )
            )
            payload: Final = (
                {"status": "succeeded", "analyzeResult": {"pages": []}}
                if "doc-intelligence" in model
                else {"pages": [{"index": 0, "markdown": "hello"}]}
            )
            encoded: Final = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *_args):
            pass

    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread: Final = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    responses: Final = []
    try:
        for enabled in (False, True):
            litellm.rust(enabled)
            arguments: Final = {
                "model": model,
                "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
                "api_key": "test-key",
                "api_base": f"http://127.0.0.1:{server.server_port}",
                "pages": [0, 2],
                "timeout": 3.0,
            }
            response: Final = await litellm.aocr(**arguments) if asynchronous else litellm.ocr(**arguments)
            responses.append(response.model_dump())
        assert len(calls) == 2
        assert calls[0] == calls[1]
        for key in ("model", "pages", "object"):
            assert responses[0][key] == responses[1][key]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
