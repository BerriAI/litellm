"""Thin Python wrapper for the native Rust OCR bridge."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol, cast  # noqa: TID251  # native extension exposes dynamically typed callables

import httpx

import litellm
from litellm.constants import request_timeout
from litellm.llms.azure_ai.ocr.common_utils import is_azure_cohere_parse_model
from litellm.llms.base_llm.ocr.transformation import PROVIDER_NATIVE_RESPONSE_KEY, OCRResponse
from litellm.rust_bridge.bindings import NativeBinding, native_exception_types
from litellm.rust_bridge.timeouts import timeout_to_seconds as _timeout_to_seconds
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager

_RUST_OCR_PROVIDERS: Final = frozenset({"mistral", "azure_ai", "vertex_ai"})
_RUST_OCR_CONFIG_FIELDS: Final = frozenset(
    {
        "azure_ad_token",
        "tenant_id",
        "client_id",
        "client_secret",
        "azure_scope",
        "azure_authority_host",
        "azure_credential",
        "azure_federated_token_file",
        "vertex_credentials",
        "vertex_ai_credentials",
        "vertex_project",
        "vertex_ai_project",
        "vertex_location",
        "vertex_ai_location",
    }
)
_RUST_OCR_SECRET_FIELDS: Final = frozenset(
    {"azure_ad_token", "client_secret", "azure_federated_token_file", "vertex_credentials", "vertex_ai_credentials"}
)


@dataclass(frozen=True, slots=True)
class LiteLLMOcrRequest:
    model: str
    document: Mapping[str, object]
    api_key: str | None
    api_base: str | None
    timeout: float | httpx.Timeout | None
    custom_llm_provider: str | None
    extra_headers: dict[str, object] | None
    kwargs: Mapping[str, object]
    input_sources: Mapping[str, str] | None = None


class RustOcr(Protocol):
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
        raise NotImplementedError


class RustAocr(Protocol):
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
    ) -> Awaitable[dict[str, object]]:
        raise NotImplementedError


class _OCRLogging(Protocol):
    def update_from_kwargs(
        self,
        *,
        kwargs: dict[str, object],
        model: str,
        optional_params: dict[str, object],
        litellm_params: dict[str, object],
        custom_llm_provider: str | None,
    ) -> None: ...

    def pre_call(
        self,
        *,
        input: str,
        api_key: str | None,
        additional_args: dict[str, object],
    ) -> None: ...


def _as_ocr(value: object) -> RustOcr | None:
    return cast(RustOcr, value) if callable(value) else None


def _as_aocr(value: object) -> RustAocr | None:
    return cast(RustAocr, value) if callable(value) else None


_OCR: Final = NativeBinding("ocr", validate=_as_ocr)
_AOCR: Final = NativeBinding("aocr", validate=_as_aocr)


def load_rust_ocr() -> RustOcr | None:
    return _OCR.load()


def load_rust_aocr() -> RustAocr | None:
    return _AOCR.load()


def provider(request: LiteLLMOcrRequest) -> str | None:
    if request.custom_llm_provider is not None:
        return request.custom_llm_provider
    prefix: Final = request.model.partition("/")[0]
    if prefix in _RUST_OCR_PROVIDERS:
        return prefix
    if request.model.startswith("mistral-ocr"):
        return "mistral"
    return None


def supported(request: LiteLLMOcrRequest) -> bool:
    request_provider: Final = provider(request)
    if request_provider not in _RUST_OCR_PROVIDERS:
        return False
    if request_provider == "azure_ai":
        return (
            not is_azure_cohere_parse_model(request.model)
            and not callable(request.kwargs.get("azure_ad_token_provider"))
            and request.kwargs.get("azure_username") is None
            and request.kwargs.get("azure_password") is None
        )
    return True


def _optional_params(request: LiteLLMOcrRequest, resolve_secret: Callable[[str], str | None]) -> Mapping[str, object]:
    optional_params: Final = MappingProxyType(
        {
            name: value
            for name, value in request.kwargs.items()
            if (name not in GenericLiteLLMParams.model_fields or name in _RUST_OCR_CONFIG_FIELDS)
            and name not in ("litellm_logging_obj", "aocr", "litellm_call_id", "proxy_server_request")
        }
    )
    request_provider: Final = provider(request)
    if request_provider == "azure_ai" and litellm.enable_azure_ad_token_refresh is True:
        return MappingProxyType({**optional_params, "enable_azure_ad_token_refresh": True})
    if request_provider != "vertex_ai":
        return optional_params
    project: Final = (
        request.kwargs.get("vertex_project")
        or request.kwargs.get("vertex_ai_project")
        or litellm.vertex_project
        or resolve_secret("VERTEXAI_PROJECT")
    )
    location: Final = (
        request.kwargs.get("vertex_location")
        or request.kwargs.get("vertex_ai_location")
        or litellm.vertex_location
        or resolve_secret("VERTEXAI_LOCATION")
        or resolve_secret("VERTEX_LOCATION")
    )
    credentials: Final = (
        request.kwargs.get("vertex_credentials")
        or request.kwargs.get("vertex_ai_credentials")
        or resolve_secret("VERTEXAI_CREDENTIALS")
    )
    vertex_params: Final = MappingProxyType(
        {
            name: value
            for name, value in (
                ("vertex_project", project),
                ("vertex_location", location),
                ("vertex_credentials", credentials),
            )
            if value is not None
        }
    )
    return MappingProxyType({**optional_params, **vertex_params})


def _input_sources(request: LiteLLMOcrRequest, optional_params: Mapping[str, object]) -> Mapping[str, str]:
    proxy_request_value: Final = request.kwargs.get("proxy_server_request")
    if not isinstance(proxy_request_value, Mapping):
        return MappingProxyType({})
    proxy_request: Final = cast(  # cast-ok: runtime Mapping check narrows metadata with unknown key and value types
        Mapping[object, object], proxy_request_value
    )
    credential_fields_value: Final = proxy_request.get("credential_fields", ())
    credential_fields: Final = (
        frozenset(name for name in credential_fields_value if isinstance(name, str))
        if isinstance(credential_fields_value, (list, tuple, set, frozenset))
        else frozenset()
    )
    request_fields_value: Final = proxy_request.get("body_fields")
    request_fields: Sequence[object]
    if isinstance(request_fields_value, Sequence) and not isinstance(request_fields_value, (str, bytes)):
        request_fields = cast(  # cast-ok: runtime Sequence check excludes scalar strings and bytes
            Sequence[object], request_fields_value
        )
    else:
        body_value: Final = proxy_request.get("body")
        request_fields = (
            tuple(cast(Mapping[object, object], body_value))  # cast-ok: runtime Mapping check establishes iterable keys
            if isinstance(body_value, Mapping)
            else ()
        )
    names: Final = frozenset(optional_params) | frozenset({"api_key", "api_base", "extra_headers"})
    request_sources: Final = MappingProxyType(
        {name: "request" for name in names if name in request_fields or name in credential_fields}
    )
    if litellm.enable_azure_ad_token_refresh is True and "enable_azure_ad_token_refresh" in optional_params:
        return MappingProxyType({**request_sources, "enable_azure_ad_token_refresh": "deployment"})
    return request_sources


def _marshal(
    request: LiteLLMOcrRequest,
    resolve_secret: Callable[[str], str | None],
    convert_file_document: Callable[[dict[str, object]], dict[str, str]],
) -> LiteLLMOcrRequest:
    if not isinstance(request.document, dict):
        raise TypeError(f"document must be a dict with 'type' and URL/file field, got {type(request.document)}")
    document: Final = (
        convert_file_document(request.document) if request.document.get("type") == "file" else request.document
    )
    request_provider: Final = provider(request)
    api_key: Final = (
        request.api_key or resolve_secret("MISTRAL_API_KEY") if request_provider == "mistral" else request.api_key
    )
    optional_params: Final = _optional_params(request, resolve_secret)
    input_sources: Final = _input_sources(request, optional_params)
    logged_optional_params: Final = MappingProxyType(
        {name: "****" if name in _RUST_OCR_SECRET_FIELDS else value for name, value in optional_params.items()}
    )
    logged_kwargs: Final = MappingProxyType(
        {
            name: "****" if name in _RUST_OCR_SECRET_FIELDS else value
            for name, value in request.kwargs.items()
            if name != "proxy_server_request"
        }
    )
    logging_obj: Final = cast(  # cast-ok: client decorator injects the logging object through untyped kwargs
        _OCRLogging, request.kwargs["litellm_logging_obj"]
    )
    logging_obj.update_from_kwargs(
        kwargs=dict(logged_kwargs),  # mutable-ok: legacy logging mutates its kwargs copy
        model=request.model,
        optional_params=dict(logged_optional_params),  # mutable-ok: legacy logging requires concrete dict params
        litellm_params={  # mutable-ok: legacy logging requires a concrete params dict
            "litellm_call_id": request.kwargs.get("litellm_call_id"),
            "api_base": request.api_base,
        },
        custom_llm_provider=request_provider,
    )
    logging_obj.pre_call(
        input="OCR document processing",
        api_key=api_key,
        additional_args={  # mutable-ok: pre_call mutates the additional_args dict
            "complete_input_dict": {  # mutable-ok: callbacks consume a JSON-serializable request dict
                "model": request.model,
                "document": document,
                **logged_optional_params,
            },
            "api_base": request.api_base or "",
            "headers": request.extra_headers or {},  # mutable-ok: logging callbacks consume a concrete headers dict
        },
    )
    return LiteLLMOcrRequest(
        model=request.model,
        document=document,
        api_key=api_key,
        api_base=request.api_base,
        timeout=request.timeout if request.timeout is not None else request_timeout,
        custom_llm_provider=request.custom_llm_provider,
        extra_headers=request.extra_headers,
        kwargs=optional_params,
        input_sources=input_sources,
    )


def _map_error(error: Exception, request: LiteLLMOcrRequest) -> Exception:
    exception_types: Final = native_exception_types()
    if exception_types is None or not isinstance(error, exception_types[1]):
        return error
    request_provider: Final = provider(request)
    if request_provider is None:
        return error
    provider_config: Final = ProviderConfigManager.get_provider_ocr_config(
        model=request.model.removeprefix(f"{request_provider}/"), provider=litellm.LlmProviders(request_provider)
    )
    if provider_config is None:
        return error
    error_args: Final = cast(  # cast-ok: BaseException.args exposes Any while native errors carry scalar args
        tuple[object, ...], error.args
    )
    status: Final = error_args[0] if error_args and isinstance(error_args[0], int) else 500
    message: Final = str(error_args[1]) if len(error_args) > 1 else str(error)
    error_factory: Final = cast(  # cast-ok: legacy provider error factories have untyped callable parameters
        Callable[..., Exception], provider_config.get_error_class
    )
    return error_factory(
        error_message=message,
        status_code=status or 500,
        headers={},  # mutable-ok: provider error factories require a concrete headers dict
    )


def _response(response: Mapping[str, object]) -> OCRResponse:
    provider_native_response: Final = response.get(PROVIDER_NATIVE_RESPONSE_KEY)
    normalized: Final = OCRResponse.model_validate(
        MappingProxyType({key: value for key, value in response.items() if key != PROVIDER_NATIVE_RESPONSE_KEY})
    )
    if isinstance(provider_native_response, Mapping):
        normalized.set_provider_native_response(provider_native_response)
    return normalized


def run(
    request: LiteLLMOcrRequest,
    resolve_secret: Callable[[str], str | None],
    convert_file_document: Callable[[dict[str, object]], dict[str, str]],
) -> OCRResponse | None:
    if load_rust_ocr() is None:
        return None
    marshalled: Final = _marshal(request, resolve_secret, convert_file_document)
    try:
        response: Final = ocr(
            model=marshalled.model,
            document=dict(marshalled.document),  # mutable-ok: PyO3 OCR binding requires a concrete dict
            api_key=marshalled.api_key,
            api_base=marshalled.api_base,
            custom_llm_provider=marshalled.custom_llm_provider,
            extra_headers=marshalled.extra_headers,
            optional_params=dict(marshalled.kwargs),  # mutable-ok: PyO3 OCR binding requires a concrete dict
            input_sources=marshalled.input_sources,
            timeout=marshalled.timeout,
        )
    except Exception as error:
        raise _map_error(error, request) from error
    return _response(response) if response is not None else None


async def arun(
    request: LiteLLMOcrRequest,
    resolve_secret: Callable[[str], str | None],
    convert_file_document: Callable[[dict[str, object]], dict[str, str]],
) -> OCRResponse | None:
    if load_rust_aocr() is None:
        return None
    marshalled: Final = _marshal(request, resolve_secret, convert_file_document)
    try:
        response: Final = await aocr(
            model=marshalled.model,
            document=dict(marshalled.document),  # mutable-ok: PyO3 OCR binding requires a concrete dict
            api_key=marshalled.api_key,
            api_base=marshalled.api_base,
            custom_llm_provider=marshalled.custom_llm_provider,
            extra_headers=marshalled.extra_headers,
            optional_params=dict(marshalled.kwargs),  # mutable-ok: PyO3 OCR binding requires a concrete dict
            input_sources=marshalled.input_sources,
            timeout=marshalled.timeout,
        )
    except Exception as error:
        raise _map_error(error, request) from error
    return _response(response) if response is not None else None


def ocr(
    *,
    model: str,
    document: dict[str, object],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,
    optional_params: dict[str, object],
    timeout: float | httpx.Timeout | None,
    input_sources: Mapping[str, str] | None = None,
) -> dict[str, object] | None:
    rust_ocr: Final = load_rust_ocr()
    if rust_ocr is None:
        return None
    return rust_ocr(
        model=model,
        document=document,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        optional_params=optional_params,
        input_sources=dict(input_sources or {}),  # mutable-ok: native boundary requires a concrete dict
        timeout_seconds=_timeout_to_seconds(timeout),
    )


async def aocr(
    *,
    model: str,
    document: dict[str, object],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,
    optional_params: dict[str, object],
    timeout: float | httpx.Timeout | None,
    input_sources: Mapping[str, str] | None = None,
) -> dict[str, object] | None:
    rust_aocr: Final = load_rust_aocr()
    if rust_aocr is None:
        return None
    return await rust_aocr(
        model=model,
        document=document,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        optional_params=optional_params,
        input_sources=dict(input_sources or {}),  # mutable-ok: native boundary requires a concrete dict
        timeout_seconds=_timeout_to_seconds(timeout),
    )
