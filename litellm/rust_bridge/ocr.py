from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import httpx
from pydantic import TypeAdapter

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.azure_ai.ocr.common_utils import is_azure_document_intelligence_model
from litellm.llms.base_llm.ocr.transformation import OCR_REQUEST_FORMAT_PARAM, BaseOCRConfig, OCRResponse
from litellm.rust_bridge import configuration as _configuration
from litellm.rust_bridge.bindings import UNCHANGED, NativeBinding, Unchanged
from litellm.rust_bridge.protocols import RustAocr, RustOcr
from litellm.rust_bridge.runtime import DispatchResult, aattempt, attempt
from litellm.rust_bridge.timeouts import timeout_to_seconds

rust: Final = _configuration.rust
rust_ocr_enabled: Final = _configuration.rust_ocr_enabled

_OCR: Final[NativeBinding[RustOcr]] = NativeBinding(lambda native: native.ocr)
_AOCR: Final[NativeBinding[RustAocr]] = NativeBinding(lambda native: native.aocr)
_HEADERS: Final = TypeAdapter(dict[str, object])


@dataclass(frozen=True, slots=True)
class PreparedOCRRequest:
    model: str
    document: dict[str, object]
    api_key: str | None
    api_base: str | None
    custom_llm_provider: str
    extra_headers: dict[str, object] | None
    provider_config: BaseOCRConfig
    optional_params: dict[str, object]
    litellm_params: dict[str, object]
    effective_timeout: float | httpx.Timeout
    litellm_logging_obj: LiteLLMLoggingObj


@dataclass(frozen=True, slots=True)
class _PreparedRustOCRCall:
    api_key: str | None
    api_base: str | None
    headers: dict[str, object]
    optional_params: dict[str, object]


_RUST_OCR_PROVIDERS: Final = frozenset(
    {
        "mistral",
        "azure_ai",
        "vertex_ai",
    }
)


def set_rust_ocr(
    *,
    ocr: RustOcr | None | Unchanged = UNCHANGED,
    aocr: RustAocr | None | Unchanged = UNCHANGED,
) -> None:
    if not isinstance(ocr, Unchanged):
        if ocr is None:
            _OCR.reset()
        else:
            _OCR.override(ocr)
    if not isinstance(aocr, Unchanged):
        if aocr is None:
            _AOCR.reset()
        else:
            _AOCR.override(aocr)


def load_rust_ocr() -> RustOcr | None:
    return _OCR.load()


def load_rust_aocr() -> RustAocr | None:
    return _AOCR.load()


def _rust_ocr_supported(prepared_request: PreparedOCRRequest) -> bool:
    if prepared_request.optional_params.get(OCR_REQUEST_FORMAT_PARAM) == "native":
        return False
    if not prepared_request.provider_config.supports_rust_bridge():
        return False
    return prepared_request.custom_llm_provider in _RUST_OCR_PROVIDERS


def _rust_bridge_optional_params(
    prepared_request: PreparedOCRRequest,
    resolve_secret: Callable[[str], str | None],
) -> dict[str, object]:
    if prepared_request.custom_llm_provider != "vertex_ai":
        return prepared_request.optional_params
    vertex_project: Final = (
        prepared_request.litellm_params.get("vertex_project")
        or prepared_request.litellm_params.get("vertex_ai_project")
        or litellm.vertex_project
        or resolve_secret("VERTEXAI_PROJECT")
    )
    vertex_location: Final = (
        prepared_request.litellm_params.get("vertex_location")
        or prepared_request.litellm_params.get("vertex_ai_location")
        or litellm.vertex_location
        or resolve_secret("VERTEXAI_LOCATION")
        or resolve_secret("VERTEX_LOCATION")
    )
    return {
        **prepared_request.optional_params,
        **{
            name: value
            for name, value in (("vertex_project", vertex_project), ("vertex_location", vertex_location))
            if value is not None
        },
    }


def _rust_bridge_api_base(
    prepared_request: PreparedOCRRequest,
    resolve_secret: Callable[[str], str | None],
) -> str | None:
    if prepared_request.api_base is not None:
        return prepared_request.api_base
    if prepared_request.custom_llm_provider == "azure_ai":
        if is_azure_document_intelligence_model(prepared_request.model):
            return resolve_secret("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        return resolve_secret("AZURE_AI_API_BASE")
    return None


def _prepare_rust_ocr_call(
    prepared_request: PreparedOCRRequest,
    resolve_api_key: Callable[[str], str | None],
) -> _PreparedRustOCRCall:
    provider_config: Final = prepared_request.provider_config
    api_key_env_var: Final = provider_config.get_api_key_env_var()
    resolved_api_key: Final = prepared_request.api_key or (
        resolve_api_key(api_key_env_var) if api_key_env_var is not None else None
    )
    resolved_headers: Final = _HEADERS.validate_python(
        provider_config.validate_environment(
            headers=prepared_request.extra_headers or {},
            model=prepared_request.model,
            api_key=resolved_api_key,
            api_base=prepared_request.api_base,
            litellm_params=prepared_request.litellm_params,
        )
    )
    resolved_complete_url: Final = provider_config.get_complete_url(
        api_base=prepared_request.api_base,
        model=prepared_request.model,
        optional_params=prepared_request.optional_params,
        litellm_params=prepared_request.litellm_params,
    )
    rust_api_base: Final = _rust_bridge_api_base(prepared_request, resolve_api_key)
    rust_optional_params: Final = _rust_bridge_optional_params(prepared_request, resolve_api_key)
    prepared_request.litellm_logging_obj.pre_call(
        input="OCR document processing",
        api_key=resolved_api_key,
        additional_args={
            "complete_input_dict": {
                "model": prepared_request.model,
                "document": prepared_request.document,
                **rust_optional_params,
            },
            "api_base": resolved_complete_url,
            "headers": resolved_headers,
        },
    )
    return _PreparedRustOCRCall(
        api_key=resolved_api_key,
        api_base=rust_api_base,
        headers=resolved_headers,
        optional_params=rust_optional_params,
    )


def attempt_ocr(
    prepared_request: PreparedOCRRequest,
    resolve_api_key: Callable[[str], str | None],
) -> DispatchResult[OCRResponse]:
    return attempt(
        load=_OCR.load,
        enabled=rust_ocr_enabled(),
        prepare=lambda: _prepare_rust_ocr_call(
            prepared_request=prepared_request,
            resolve_api_key=resolve_api_key,
        ),
        call=lambda native, prepared: native(
            model=prepared_request.model,
            document=prepared_request.document,
            api_key=prepared.api_key,
            api_base=prepared.api_base,
            custom_llm_provider=prepared_request.custom_llm_provider,
            extra_headers=prepared.headers,
            optional_params=prepared.optional_params,
            timeout_seconds=timeout_to_seconds(prepared_request.effective_timeout),
        ),
        adapt=OCRResponse.model_validate,
        eligible=_rust_ocr_supported(prepared_request),
    )


async def aattempt_ocr(
    prepared_request: PreparedOCRRequest,
    resolve_api_key: Callable[[str], str | None],
) -> DispatchResult[OCRResponse]:
    return await aattempt(
        load=_AOCR.load,
        enabled=rust_ocr_enabled(),
        prepare=lambda: _prepare_rust_ocr_call(
            prepared_request=prepared_request,
            resolve_api_key=resolve_api_key,
        ),
        call=lambda native, prepared: native(
            model=prepared_request.model,
            document=prepared_request.document,
            api_key=prepared.api_key,
            api_base=prepared.api_base,
            custom_llm_provider=prepared_request.custom_llm_provider,
            extra_headers=prepared.headers,
            optional_params=prepared.optional_params,
            timeout_seconds=timeout_to_seconds(prepared_request.effective_timeout),
        ),
        adapt=OCRResponse.model_validate,
        eligible=_rust_ocr_supported(prepared_request),
    )
