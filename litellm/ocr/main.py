"""
Main OCR function for LiteLLM.
"""

import asyncio
import base64
import mimetypes
import os
import re
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from io import IOBase
from typing import Any, Final, cast

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.constants import request_timeout
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.azure_ai.ocr.common_utils import (
    is_azure_document_intelligence_model,
)
from litellm.llms.base_llm.ocr.transformation import (
    OCR_REQUEST_FORMAT_PARAM,
    BaseOCRConfig,
    OCRResponse,
    parse_ocr_request_format,
)
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.rust_bridge import ocr as rust_ocr_bridge
from litellm.rust_bridge.callback_adapters import ProviderLoggingAdapter
from litellm.rust_bridge.request import (
    NativeRequestCapabilities,
    NativeRequestOptions,
    PreparedNativeCall,
    request_context,
    vertex_options,
)
from litellm.rust_bridge.timeouts import timeout_to_seconds
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager, client

####### ENVIRONMENT VARIABLES ###################
base_llm_http_handler = BaseLLMHTTPHandler()
#################################################


@dataclass
class _PreparedOCRRequest:
    model: str
    document: dict[str, Any]  # mutable-ok: public OCR document shape is a mutable SDK dictionary
    api_key: str | None
    api_base: str | None
    custom_llm_provider: str
    extra_headers: dict[str, object] | None  # mutable-ok: preserves the caller-owned SDK header contract
    provider_config: BaseOCRConfig
    optional_params: dict[str, object]  # mutable-ok: provider mapping consumes an owned parameter copy
    litellm_params: dict[str, object]  # mutable-ok: preserves the existing SDK parameter contract
    effective_timeout: float | httpx.Timeout
    litellm_logging_obj: LiteLLMLoggingObj
    execution_mode: str = "sync"


@dataclass(frozen=True, slots=True)
class _NativeOCRRequest:
    model: str
    document: Mapping[str, object]
    api_key: str | None
    api_base: str | None
    custom_llm_provider: str | None
    extra_headers: dict[str, object] | None  # mutable-ok: preserves the public SDK header contract
    kwargs: Mapping[str, object]
    effective_timeout: float | httpx.Timeout
    litellm_logging_obj: LiteLLMLoggingObj
    execution_mode: str


def _native_provider(model: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    provider, separator, _ = model.partition("/")
    return provider if separator else "mistral"


def _native_model(model: str, provider: str) -> str:
    prefix: Final = f"{provider}/"
    return model.removeprefix(prefix)


def _python_fallback_document(document: Mapping[str, object]) -> Mapping[str, object]:
    """Materialize local files only after native dispatch selected Python."""
    if document.get("type") != "file":
        return document
    owned_document: Final = dict(  # mutable-ok: legacy file conversion owns and rewrites this copy
        document
    )
    return convert_file_document_to_url_document(owned_document)


def _native_ocr_request(
    *,
    model: str,
    document: Mapping[str, object],
    api_key: str | None,
    api_base: str | None,
    timeout: float | httpx.Timeout | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,  # mutable-ok: preserves the public SDK header contract
    kwargs: Mapping[str, object],
    execution_mode: str,
) -> _NativeOCRRequest:
    logging_value: Final = kwargs["litellm_logging_obj"]
    if not isinstance(logging_value, LiteLLMLoggingObj):
        raise TypeError("litellm_logging_obj must be a LiteLLM Logging instance")
    logging_obj: Final = logging_value
    return _NativeOCRRequest(
        model=model,
        document=document,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        kwargs=dict(kwargs),  # mutable-ok: fallback receives an isolated request snapshot
        effective_timeout=timeout or request_timeout,
        litellm_logging_obj=logging_obj,
        execution_mode=execution_mode,
    )


def _prepare_ocr_request(
    model: str,
    document: Mapping[str, object],
    api_key: str | None,
    api_base: str | None,
    timeout: float | httpx.Timeout | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,
    kwargs: dict[str, object],
    execution_mode: str = "sync",
) -> _PreparedOCRRequest:
    litellm_logging_obj: Final = cast(LiteLLMLoggingObj, kwargs.pop("litellm_logging_obj"))
    litellm_call_id: Final = cast(str | None, kwargs.get("litellm_call_id", None))

    if not isinstance(document, dict):
        raise ValueError(f"document must be a dict with 'type' and URL/file field, got {type(document)}")

    doc_type = document.get("type")

    if doc_type not in ("document_url", "image_url", "file"):
        raise ValueError(f"Invalid document type: {doc_type}. Must be 'document_url', 'image_url', or 'file'")

    caller_supplied_api_base: Final = api_base is not None

    (
        model,
        custom_llm_provider,
        dynamic_api_key,
        dynamic_api_base,
    ) = litellm.get_llm_provider(
        model=model,
        custom_llm_provider=custom_llm_provider,
        api_base=api_base,
        api_key=api_key,
    )

    suppress_dynamic_api_base: Final = (
        not caller_supplied_api_base
        and custom_llm_provider == "azure_ai"
        and is_azure_document_intelligence_model(model)
    )
    if dynamic_api_key:
        api_key = dynamic_api_key
    if dynamic_api_base and not suppress_dynamic_api_base:
        api_base = dynamic_api_base

    ocr_provider_config: Final = ProviderConfigManager.get_provider_ocr_config(
        model=model,
        provider=litellm.LlmProviders(custom_llm_provider),
    )

    if ocr_provider_config is None:
        raise ValueError(f"OCR is not supported for provider: {custom_llm_provider}")

    verbose_logger.debug("OCR call - model: %s, provider: %s", model, custom_llm_provider)

    litellm_params: Final = GenericLiteLLMParams.model_validate(kwargs)

    supported_params: Final = ocr_provider_config.get_supported_ocr_params(model=model)
    requested_format: Final = kwargs.get(OCR_REQUEST_FORMAT_PARAM)
    if requested_format is not None:
        try:
            parsed_format: Final = parse_ocr_request_format(requested_format)
        except ValueError as e:
            raise litellm.exceptions.UnsupportedParamsError(
                message=f"{e}", model=model, llm_provider=custom_llm_provider
            ) from e
        if OCR_REQUEST_FORMAT_PARAM not in supported_params and parsed_format == "native":
            raise litellm.exceptions.UnsupportedParamsError(
                message=(
                    f"`{OCR_REQUEST_FORMAT_PARAM}='native'` is not supported for provider: {custom_llm_provider}, "
                    f"model: {model}"
                ),
                model=model,
                llm_provider=custom_llm_provider,
            )

    non_default_params: Final = {}
    for param in supported_params:
        if param in kwargs:
            non_default_params[param] = kwargs.pop(param)

    optional_params: Final = ocr_provider_config.map_ocr_params(
        non_default_params=non_default_params,
        optional_params={},
        model=model,
    )

    verbose_logger.debug("OCR optional_params after mapping: %s", optional_params)

    effective_timeout: Final = timeout or request_timeout

    litellm_logging_obj.update_from_kwargs(
        kwargs=kwargs,
        model=model,
        optional_params=optional_params,
        litellm_params={
            "litellm_call_id": litellm_call_id,
            "api_base": api_base,
        },
        custom_llm_provider=custom_llm_provider,
    )

    return _PreparedOCRRequest(
        model=model,
        document=document,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        provider_config=ocr_provider_config,
        optional_params=cast(dict[str, object], optional_params),
        litellm_params=dict(litellm_params),
        effective_timeout=effective_timeout,
        litellm_logging_obj=litellm_logging_obj,
        execution_mode=execution_mode,
    )


def _rust_bridge_optional_params(
    prepared_request: _PreparedOCRRequest,
) -> dict[str, object]:  # mutable-ok: returns an owned provider-parameter copy
    optional_params: Final = dict(prepared_request.optional_params)
    if prepared_request.custom_llm_provider == "vertex_ai":
        vertex_project: Final = (
            prepared_request.litellm_params.get("vertex_project")
            or prepared_request.litellm_params.get("vertex_ai_project")
            or litellm.vertex_project
        )
        vertex_location: Final = (
            prepared_request.litellm_params.get("vertex_location")
            or prepared_request.litellm_params.get("vertex_ai_location")
            or litellm.vertex_location
        )
        if vertex_project is not None:
            optional_params["vertex_project"] = vertex_project
        if vertex_location is not None:
            optional_params["vertex_location"] = vertex_location
    return optional_params


def _prepare_rust_ocr_call(
    prepared_request: _PreparedOCRRequest,
) -> PreparedNativeCall[rust_ocr_bridge.NativeOCRRequest]:
    rust_optional_params: Final = _rust_bridge_optional_params(prepared_request)
    return PreparedNativeCall(
        request=rust_ocr_bridge.NativeOCRRequest(
            model=prepared_request.model,
            document=prepared_request.document,
            optional_params=prepared_request.optional_params,
        ),
        options=NativeRequestOptions(
            vertex=vertex_options(rust_optional_params),
            api_key=prepared_request.api_key,
            api_base=prepared_request.api_base,
            custom_llm_provider=prepared_request.custom_llm_provider,
            extra_headers=prepared_request.extra_headers,
            timeout_seconds=timeout_to_seconds(prepared_request.effective_timeout),
        ),
        context=request_context(
            logging_obj=prepared_request.litellm_logging_obj,
            request_model=getattr(prepared_request.litellm_logging_obj, "model", prepared_request.model),
            litellm_params=prepared_request.litellm_params,
            capabilities=NativeRequestCapabilities(
                execution_mode=prepared_request.execution_mode,
                input_source_kind=str(prepared_request.document.get("type") or "unknown"),
                request_format=(
                    value
                    if isinstance((value := prepared_request.optional_params.get(OCR_REQUEST_FORMAT_PARAM)), str)
                    else None
                ),
                native_response_format=(prepared_request.optional_params.get(OCR_REQUEST_FORMAT_PARAM) == "native"),
            ),
        ),
        callback_adapter=ProviderLoggingAdapter(
            prepared_request.litellm_logging_obj,
            "OCR document processing",
            prepared_request.api_key,
        ),
    )


def _prepare_native_ocr_call(
    request: _NativeOCRRequest,
) -> PreparedNativeCall[rust_ocr_bridge.NativeOCRRequest]:
    provider: Final = _native_provider(request.model, request.custom_llm_provider)
    model: Final = _native_model(request.model, provider)
    litellm_params: Final = dict(  # mutable-ok: request context helper consumes a mapping snapshot
        GenericLiteLLMParams.model_validate(request.kwargs)
    )
    internal_params: Final = set(GenericLiteLLMParams.model_fields) | {  # mutable-ok: local classification set
        "litellm_logging_obj",
        "litellm_call_id",
    }
    optional_params: Final = {  # mutable-ok: native request owns provider extension fields
        name: value for name, value in request.kwargs.items() if name not in internal_params
    }
    vertex_params: Final = {  # mutable-ok: combines typed routing fields without mutating either source
        **litellm_params,
        **optional_params,
    }
    request_format: Final = optional_params.get(OCR_REQUEST_FORMAT_PARAM)
    auth_provider = request.kwargs.get("azure_ad_token_provider")
    if not callable(auth_provider):
        auth_provider = None
    request.litellm_logging_obj.update_from_kwargs(
        kwargs=dict(request.kwargs),  # mutable-ok: logger owns its request-state snapshot
        model=model,
        optional_params=optional_params,
        litellm_params={  # mutable-ok: logger contract requires its call context as a dict
            "litellm_call_id": litellm_params.get("litellm_call_id"),
            "api_base": request.api_base,
        },
        custom_llm_provider=provider,
    )
    return PreparedNativeCall(
        request=rust_ocr_bridge.NativeOCRRequest(
            model=model,
            document=request.document,
            optional_params=optional_params,
        ),
        options=NativeRequestOptions(
            vertex=vertex_options(vertex_params),
            api_key=request.api_key,
            api_base=request.api_base,
            custom_llm_provider=provider,
            extra_headers=request.extra_headers,
            timeout_seconds=timeout_to_seconds(request.effective_timeout),
            auth_provider=auth_provider,
        ),
        context=request_context(
            logging_obj=request.litellm_logging_obj,
            request_model=request.model,
            litellm_params=litellm_params,
            capabilities=NativeRequestCapabilities(
                execution_mode=request.execution_mode,
                input_source_kind=str(request.document.get("type") or "unknown"),
                request_format=request_format if isinstance(request_format, str) else None,
                native_response_format=request_format == "native",
            ),
        ),
        callback_adapter=ProviderLoggingAdapter(
            request.litellm_logging_obj,
            "OCR document processing",
            request.api_key,
        ),
    )


def _run_native_ocr(
    request: _NativeOCRRequest,
    fallback: Callable[[], OCRResponse | Coroutine[object, object, OCRResponse]],
) -> OCRResponse | Coroutine[object, object, OCRResponse]:
    provider: Final = _native_provider(request.model, request.custom_llm_provider)
    return rust_ocr_bridge.dispatch_ocr(
        prepare=lambda: _prepare_native_ocr_call(request),
        fallback=fallback,
        adapt=OCRResponse.model_validate,
        model=_native_model(request.model, provider),
        provider=provider,
    )


async def _run_native_aocr(
    request: _NativeOCRRequest,
    fallback: Callable[[], Coroutine[object, object, OCRResponse]],
) -> OCRResponse:
    provider: Final = _native_provider(request.model, request.custom_llm_provider)
    return await rust_ocr_bridge.adispatch_ocr(
        prepare=lambda: _prepare_native_ocr_call(request),
        fallback=fallback,
        adapt=OCRResponse.model_validate,
        model=_native_model(request.model, provider),
        provider=provider,
    )


@dataclass
class _OCROperation:
    request: _PreparedOCRRequest
    python: Callable[[], OCRResponse | Coroutine[object, object, OCRResponse]]

    def prepare(self) -> PreparedNativeCall[rust_ocr_bridge.NativeOCRRequest]:
        return _prepare_rust_ocr_call(self.request)

    def fallback(self) -> OCRResponse | Coroutine[object, object, OCRResponse]:
        return self.python()

    async def afallback(self) -> OCRResponse:
        result: Final = self.python()
        return await result if isinstance(result, Coroutine) else result


def _run_rust_ocr(
    prepared_request: _PreparedOCRRequest,
    resolve_api_key: Callable[[str], str | None],
    fallback: Callable[[], OCRResponse | Coroutine[object, object, OCRResponse]],
) -> OCRResponse | Coroutine[object, object, OCRResponse]:
    del resolve_api_key
    operation: Final = _OCROperation(prepared_request, fallback)
    return rust_ocr_bridge.dispatch_ocr(
        prepare=operation.prepare,
        fallback=operation.fallback,
        adapt=OCRResponse.model_validate,
        model=prepared_request.model,
        provider=prepared_request.custom_llm_provider,
    )


async def _run_rust_aocr(
    prepared_request: _PreparedOCRRequest,
    resolve_api_key: Callable[[str], str | None],
    fallback: Callable[[], Coroutine[object, object, OCRResponse]],
) -> OCRResponse:
    del resolve_api_key
    operation: Final = _OCROperation(prepared_request, fallback)
    return await rust_ocr_bridge.adispatch_ocr(
        prepare=operation.prepare,
        fallback=operation.afallback,
        adapt=OCRResponse.model_validate,
        model=prepared_request.model,
        provider=prepared_request.custom_llm_provider,
    )


def _python_fallback_document(document: Mapping[str, object]) -> Mapping[str, object]:
    """Materialize local files only after native dispatch selected Python."""
    if document.get("type") != "file":
        return document
    owned_document: Final = dict(  # mutable-ok: legacy file conversion owns and rewrites this copy
        document
    )
    return convert_file_document_to_url_document(owned_document)


@client
async def aocr(
    model: str,
    document: Mapping[str, object],
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: dict[str, object] | None = None,
    **kwargs: object,
) -> OCRResponse:
    """
    Async OCR function.

    Args:
        model: Model name (e.g., "mistral/mistral-ocr-latest")
        document: Document to process in Mistral format:
            {"type": "document_url", "document_url": "https://..."} for PDFs/docs,
            {"type": "image_url", "image_url": "https://..."} for images, or
            {"type": "file", "file": <path/bytes/file-obj>} for local files
        api_key: Optional API key
        api_base: Optional API base URL
        timeout: Optional timeout
        custom_llm_provider: Optional custom LLM provider
        extra_headers: Optional extra headers
        **kwargs: Additional parameters (e.g., include_image_base64, pages, image_limit)

    Returns:
        OCRResponse in Mistral OCR format with pages, model, usage_info, etc.

    Example:
        ```python
        import litellm

        # OCR with PDF
        response = await litellm.aocr(
            model="mistral/mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": "https://arxiv.org/pdf/2201.04234"
            },
            include_image_base64=True
        )

        # OCR with image
        response = await litellm.aocr(
            model="mistral/mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": "https://example.com/image.png"
            }
        )

        # OCR with base64 encoded PDF
        response = await litellm.aocr(
            model="mistral/mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_pdf}"
            }
        )

        # OCR with local file
        response = await litellm.aocr(
            model="mistral/mistral-ocr-latest",
            document={"type": "file", "file": "/path/to/document.pdf"}
        )
        ```
    """
    completion_kwargs: Final[dict[str, object]] = {
        "model": model,
        "document": document,
        "api_key": api_key,
        "api_base": api_base,
        "timeout": timeout,
        "custom_llm_provider": custom_llm_provider,
        "extra_headers": extra_headers,
        "kwargs": kwargs,
    }
    try:
        native_request: Final = _native_ocr_request(
            model=model,
            document=document,
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            kwargs=kwargs,
            execution_mode="async",
        )
        custom_llm_provider = _native_provider(model, custom_llm_provider)
        model = _native_model(model, custom_llm_provider)
        completion_kwargs.update({"model": model, "custom_llm_provider": custom_llm_provider})
        prepared: _PreparedOCRRequest | None = None

        async def python_fallback() -> OCRResponse:
            nonlocal prepared
            prepared = prepared or _prepare_ocr_request(
                model=native_request.model,
                document=native_request.document,
                api_key=native_request.api_key,
                api_base=native_request.api_base,
                timeout=native_request.effective_timeout,
                custom_llm_provider=native_request.custom_llm_provider,
                extra_headers=native_request.extra_headers,
                kwargs=dict(  # mutable-ok: Python fallback owns its single mutable preparation copy
                    native_request.kwargs
                ),
                execution_mode="async",
            )
            fallback_document: Final = _python_fallback_document(prepared.document)
            pending: Final = base_llm_http_handler.ocr(
                model=prepared.model,
                document=fallback_document,
                optional_params=prepared.optional_params,
                timeout=prepared.effective_timeout,
                logging_obj=prepared.litellm_logging_obj,
                api_key=prepared.api_key,
                api_base=prepared.api_base,
                custom_llm_provider=prepared.custom_llm_provider,
                aocr=True,
                headers=prepared.extra_headers,
                provider_config=prepared.provider_config,
                litellm_params=prepared.litellm_params,
            )
            response: Final = await pending if asyncio.iscoroutine(pending) else pending
            if response is None:  # pyright: ignore[reportUnnecessaryComparison]  # provider adapters can violate their declared return type
                raise ValueError(f"Got an unexpected None response from the OCR API: {response}")
            return response

        return await _run_native_aocr(
            request=native_request,
            fallback=python_fallback,
        )
    except Exception as e:
        raise litellm.exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=completion_kwargs,
            extra_kwargs=kwargs,
        )


#################################################
# Public utilities — used by the SDK and the proxy
#################################################

_MIME_PATTERN: Final = re.compile(r"^[\w.+-]+/[\w.+-]+$")

_MIME_TYPE_MAP: Final = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".bmp": "image/bmp",
}


def get_mime_type(file_path: str) -> str:
    """
    Determine MIME type from file path extension.

    Falls back to mimetypes.guess_type, then to 'application/octet-stream'.
    """
    ext: Final = os.path.splitext(file_path)[1].lower()
    mime: Final = _MIME_TYPE_MAP.get(ext)
    if mime:
        return mime
    guessed, _ = mimetypes.guess_type(file_path)
    return guessed or "application/octet-stream"


def convert_file_document_to_url_document(document: dict[str, Any]) -> dict[str, str]:
    """
    Convert a file-type document dict to a document_url-type document dict
    with an inline base64 data URI.

    Accepts document dicts like:
        {"type": "file", "file": Path("/path/to/doc.pdf")}       # pathlib.Path
        {"type": "file", "file": <binary file-like object>}      # file-like object (BinaryIO)
        {"type": "file", "file": b"raw bytes"}                   # raw bytes

    Bare ``str`` paths are not accepted — pass a ``pathlib.Path`` or
    ``open(path, "rb")`` instead. See the str check below for the rationale.

    Returns:
        {"type": "document_url", "document_url": "data:<mime>;base64,<data>"}
        or {"type": "image_url", "image_url": "data:<mime>;base64,<data>"}
    """
    file_input: Final = document.get("file")
    if file_input is None:
        raise ValueError(
            "document with type='file' must include a 'file' field containing "
            "a pathlib.Path, file-like object, or bytes"
        )

    file_bytes: bytes
    mime_type: str = "application/octet-stream"
    file_name: str | None = None

    if isinstance(file_input, str):
        # Bare strings are rejected here. The OCR ``document`` accepts a
        # ``{"type": "file", "file": <value>}`` shape, and when this helper
        # runs in a proxy request handler ``<value>`` is attacker-controlled.
        # Opening it as a path is an arbitrary local file read on the proxy
        # host, which is then base64-encoded and forwarded to the OCR
        # provider — an exfiltration primitive.
        raise ValueError(
            "OCR file input does not accept bare str values. Pass bytes, "
            "a pathlib.Path, or a file-like object. To OCR a local file "
            "from a path, call open(path, 'rb') yourself."
        )
    if isinstance(file_input, os.PathLike):
        # os.PathLike (pathlib.Path and custom __fspath__ classes) is a
        # Python-level type that HTTP form values can't fabricate.
        file_path: Final = str(file_input)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        mime_type = get_mime_type(file_path)
        file_name = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            file_bytes = f.read()
    elif isinstance(file_input, bytes):
        file_bytes = file_input
    elif isinstance(file_input, IOBase) or hasattr(file_input, "read"):
        if hasattr(file_input, "name"):
            file_name = getattr(file_input, "name", None)
            if file_name:
                mime_type = get_mime_type(file_name)
        file_bytes = file_input.read()
        if isinstance(file_bytes, str):
            file_bytes = file_bytes.encode("utf-8")
    else:
        raise ValueError(
            f"Unsupported file input type: {type(file_input)}. Expected pathlib.Path, bytes, or a file-like object."
        )

    if not file_bytes:
        raise ValueError("File is empty or could not be read")

    if "mime_type" in document:
        mime_type = document["mime_type"]

    if not _MIME_PATTERN.match(mime_type):
        raise ValueError(f"Invalid MIME type: {mime_type}")

    base64_data: Final = base64.b64encode(file_bytes).decode("utf-8")
    data_uri: Final = f"data:{mime_type};base64,{base64_data}"

    if mime_type.startswith("image/"):
        verbose_logger.debug(
            "OCR file input: Converted file to image_url data URI (mime=%s, size=%s bytes, name=%s)",
            mime_type,
            len(file_bytes),
            file_name,
        )
        return {"type": "image_url", "image_url": data_uri}

    verbose_logger.debug(
        "OCR file input: Converted file to document_url data URI (mime=%s, size=%s bytes, name=%s)",
        mime_type,
        len(file_bytes),
        file_name,
    )
    return {"type": "document_url", "document_url": data_uri}


@client
def ocr(
    model: str,
    document: Mapping[str, object],
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: dict[str, object] | None = None,
    **kwargs: object,
) -> OCRResponse | Coroutine[object, object, OCRResponse]:
    """
    Synchronous OCR function.

    Args:
        model: Model name (e.g., "mistral/mistral-ocr-latest")
        document: Document to process in Mistral format:
            {"type": "document_url", "document_url": "https://..."} for PDFs/docs,
            {"type": "image_url", "image_url": "https://..."} for images, or
            {"type": "file", "file": <path/bytes/file-obj>} for local files
        api_key: Optional API key
        api_base: Optional API base URL
        timeout: Optional timeout
        custom_llm_provider: Optional custom LLM provider
        extra_headers: Optional extra headers
        **kwargs: Additional parameters (e.g., include_image_base64, pages, image_limit)

    Returns:
        OCRResponse in Mistral OCR format with pages, model, usage_info, etc.

    Example:
        ```python
        import litellm

        # OCR with PDF
        response = litellm.ocr(
            model="mistral/mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": "https://arxiv.org/pdf/2201.04234"
            },
            include_image_base64=True
        )

        # OCR with image
        response = litellm.ocr(
            model="mistral/mistral-ocr-latest",
            document={
                "type": "image_url",
                "image_url": "https://example.com/image.png"
            }
        )

        # OCR with base64 encoded PDF
        response = litellm.ocr(
            model="mistral/mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_pdf}"
            }
        )

        # OCR with local file
        response = litellm.ocr(
            model="mistral/mistral-ocr-latest",
            document={"type": "file", "file": "/path/to/document.pdf"}
        )

        # Access pages
        for page in response.pages:
            print(f"Page {page.index}: {page.markdown}")
        ```
    """
    completion_kwargs: Final[dict[str, object]] = {
        "model": model,
        "document": document,
        "api_key": api_key,
        "api_base": api_base,
        "timeout": timeout,
        "custom_llm_provider": custom_llm_provider,
        "extra_headers": extra_headers,
        "kwargs": kwargs,
    }
    try:
        _is_async: Final = kwargs.pop("aocr", False) is True
        completion_kwargs["aocr"] = _is_async
        native_request: Final = _native_ocr_request(
            model=model,
            document=document,
            api_key=api_key,
            api_base=api_base,
            kwargs=kwargs,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            timeout=timeout,
            execution_mode="async" if _is_async else "sync",
        )
        custom_llm_provider = _native_provider(model, custom_llm_provider)
        model = _native_model(model, custom_llm_provider)
        completion_kwargs.update({"model": model, "custom_llm_provider": custom_llm_provider})
        prepared: _PreparedOCRRequest | None = None

        def python_fallback() -> OCRResponse | Coroutine[object, object, OCRResponse]:
            nonlocal prepared
            prepared = prepared or _prepare_ocr_request(
                model=native_request.model,
                document=native_request.document,
                api_key=native_request.api_key,
                api_base=native_request.api_base,
                timeout=native_request.effective_timeout,
                custom_llm_provider=native_request.custom_llm_provider,
                extra_headers=native_request.extra_headers,
                kwargs=dict(  # mutable-ok: Python fallback owns its single mutable preparation copy
                    native_request.kwargs
                ),
                execution_mode=native_request.execution_mode,
            )
            fallback_document: Final = _python_fallback_document(prepared.document)
            return base_llm_http_handler.ocr(
                model=prepared.model,
                document=fallback_document,
                optional_params=prepared.optional_params,
                timeout=prepared.effective_timeout,
                logging_obj=prepared.litellm_logging_obj,
                api_key=prepared.api_key,
                api_base=prepared.api_base,
                custom_llm_provider=prepared.custom_llm_provider,
                aocr=_is_async,
                headers=prepared.extra_headers,
                provider_config=prepared.provider_config,
                litellm_params=prepared.litellm_params,
            )

        return _run_native_ocr(
            request=native_request,
            fallback=python_fallback,
        )
    except Exception as e:
        raise litellm.exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=completion_kwargs,
            extra_kwargs=kwargs,
        )
