"""
Main OCR function for LiteLLM.
"""

import asyncio
import base64
import mimetypes
import os
import re
from dataclasses import dataclass
from io import IOBase
from typing import Any, Coroutine, Union, cast

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.constants import request_timeout
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.ocr.transformation import BaseOCRConfig, OCRResponse
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.ocr.rust_bridge import (
    RustAocr,
    RustOcr,
    load_rust_aocr,
    load_rust_ocr,
    rust_ocr_enabled,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager, client

####### ENVIRONMENT VARIABLES ###################
base_llm_http_handler = BaseLLMHTTPHandler()
#################################################


@dataclass
class _PreparedOCRRequest:
    model: str
    document: dict[str, Any]
    api_key: str | None
    api_base: str | None
    custom_llm_provider: str
    extra_headers: dict[str, object] | None
    provider_config: BaseOCRConfig
    optional_params: dict[str, object]
    litellm_params: dict[str, object]
    effective_timeout: Union[float, httpx.Timeout]
    litellm_logging_obj: LiteLLMLoggingObj


@dataclass
class _PreparedRustOCRArgs:
    optional_params: dict[str, object]


_RUST_BRIDGE_INTERNAL_KWARGS = {
    "litellm_call_id",
    "litellm_logging_obj",
}


def _timeout_to_seconds(
    timeout: Union[float, httpx.Timeout] | None,
) -> float | None:
    """Convert the Python OCR timeout to a single seconds value for the Rust bridge.

    The Rust HTTP client takes one duration; ``httpx.Timeout`` carries separate
    connect/read/write/pool values, so pick the read deadline as the closest
    analog to a total-request timeout.
    """
    if timeout is None:
        return None
    if isinstance(timeout, httpx.Timeout):
        return timeout.read
    return float(timeout)


def _prepare_ocr_request(
    model: str,
    document: dict[str, Any],
    api_key: str | None,
    api_base: str | None,
    timeout: Union[float, httpx.Timeout] | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, Any] | None,
    kwargs: dict[str, Any],
) -> _PreparedOCRRequest:
    litellm_logging_obj = cast(LiteLLMLoggingObj, kwargs.pop("litellm_logging_obj"))
    litellm_call_id = cast(str | None, kwargs.get("litellm_call_id", None))

    if not isinstance(document, dict):
        raise ValueError(
            f"document must be a dict with 'type' and URL/file field, got {type(document)}"
        )

    doc_type = document.get("type")

    if doc_type == "file":
        document = convert_file_document_to_url_document(document)
        doc_type = document.get("type")

    if doc_type not in ["document_url", "image_url"]:
        raise ValueError(
            f"Invalid document type: {doc_type}. "
            "Must be 'document_url', 'image_url', or 'file'"
        )

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

    if dynamic_api_key:
        api_key = dynamic_api_key
    if dynamic_api_base:
        api_base = dynamic_api_base

    ocr_provider_config = ProviderConfigManager.get_provider_ocr_config(
        model=model,
        provider=litellm.LlmProviders(custom_llm_provider),
    )

    if ocr_provider_config is None:
        raise ValueError(f"OCR is not supported for provider: {custom_llm_provider}")

    verbose_logger.debug(f"OCR call - model: {model}, provider: {custom_llm_provider}")

    litellm_params = GenericLiteLLMParams(**kwargs)

    supported_params = ocr_provider_config.get_supported_ocr_params(model=model)
    non_default_params = {}
    for param in supported_params:
        if param in kwargs:
            non_default_params[param] = kwargs.pop(param)

    optional_params = ocr_provider_config.map_ocr_params(
        non_default_params=non_default_params,
        optional_params={},
        model=model,
    )

    verbose_logger.debug(f"OCR optional_params after mapping: {optional_params}")

    effective_timeout = timeout or request_timeout

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
        extra_headers=cast(dict[str, object] | None, extra_headers),
        provider_config=ocr_provider_config,
        optional_params=cast(dict[str, object], optional_params),
        litellm_params=dict(litellm_params),
        effective_timeout=effective_timeout,
        litellm_logging_obj=litellm_logging_obj,
    )


def _prepare_rust_ocr_args(kwargs: dict[str, Any]) -> _PreparedRustOCRArgs:
    return _PreparedRustOCRArgs(
        optional_params={
            key: value
            for key, value in kwargs.items()
            if key not in _RUST_BRIDGE_INTERNAL_KWARGS
        }
    )


def _run_rust_ocr(
    rust_ocr: RustOcr,
    *,
    model: str,
    document: dict[str, Any],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, Any] | None,
    timeout: Union[float, httpx.Timeout] | None,
    kwargs: dict[str, Any],
) -> OCRResponse:
    """Forward raw public OCR arguments to Rust and wrap the response."""
    prepared = _prepare_rust_ocr_args(kwargs)
    return OCRResponse.model_validate(
        rust_ocr(
            model=model,
            document=cast(dict[str, object], document),
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=cast(dict[str, object] | None, extra_headers),
            optional_params=prepared.optional_params,
            timeout_seconds=_timeout_to_seconds(timeout),
        )
    )


async def _run_rust_aocr(
    rust_aocr: RustAocr,
    *,
    model: str,
    document: dict[str, Any],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, Any] | None,
    timeout: Union[float, httpx.Timeout] | None,
    kwargs: dict[str, Any],
) -> OCRResponse:
    prepared = _prepare_rust_ocr_args(kwargs)
    return OCRResponse.model_validate(
        await rust_aocr(
            model=model,
            document=cast(dict[str, object], document),
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=cast(dict[str, object] | None, extra_headers),
            optional_params=prepared.optional_params,
            timeout_seconds=_timeout_to_seconds(timeout),
        )
    )


@client
async def aocr(
    model: str,
    document: dict[str, Any],
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: Union[float, httpx.Timeout] | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: dict[str, Any] | None = None,
    **kwargs,
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
    completion_kwargs: dict[str, object] = {
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
        if rust_ocr_enabled():
            rust_aocr = load_rust_aocr()
            if rust_aocr is None:
                verbose_logger.debug(
                    "Async Rust OCR bridge unavailable; falling back to Python path"
                )
            else:
                return await _run_rust_aocr(
                    rust_aocr=rust_aocr,
                    model=model,
                    document=document,
                    api_key=api_key,
                    api_base=api_base,
                    custom_llm_provider=custom_llm_provider,
                    extra_headers=extra_headers,
                    timeout=timeout,
                    kwargs=kwargs,
                )

        prepared = _prepare_ocr_request(
            model=model,
            document=document,
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            kwargs=kwargs,
        )
        model = prepared.model
        custom_llm_provider = prepared.custom_llm_provider
        completion_kwargs.update(
            {"model": model, "custom_llm_provider": custom_llm_provider}
        )

        response = base_llm_http_handler.ocr(
            model=prepared.model,
            document=prepared.document,
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

        if asyncio.iscoroutine(response):
            response = await response

        if response is None:
            raise ValueError(
                f"Got an unexpected None response from the OCR API: {response}"
            )

        return response
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

_MIME_PATTERN = re.compile(r"^[\w.+-]+/[\w.+-]+$")

_MIME_TYPE_MAP = {
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
    ext = os.path.splitext(file_path)[1].lower()
    mime = _MIME_TYPE_MAP.get(ext)
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
    file_input = document.get("file")
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
        file_path = str(file_input)
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
            f"Unsupported file input type: {type(file_input)}. "
            "Expected pathlib.Path, bytes, or a file-like object."
        )

    if not file_bytes:
        raise ValueError("File is empty or could not be read")

    if "mime_type" in document:
        mime_type = document["mime_type"]

    if not _MIME_PATTERN.match(mime_type):
        raise ValueError(f"Invalid MIME type: {mime_type}")

    base64_data = base64.b64encode(file_bytes).decode("utf-8")
    data_uri = f"data:{mime_type};base64,{base64_data}"

    if mime_type.startswith("image/"):
        verbose_logger.debug(
            f"OCR file input: Converted file to image_url data URI "
            f"(mime={mime_type}, size={len(file_bytes)} bytes, name={file_name})"
        )
        return {"type": "image_url", "image_url": data_uri}

    verbose_logger.debug(
        f"OCR file input: Converted file to document_url data URI "
        f"(mime={mime_type}, size={len(file_bytes)} bytes, name={file_name})"
    )
    return {"type": "document_url", "document_url": data_uri}


@client
def ocr(
    model: str,
    document: dict[str, Any],
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: Union[float, httpx.Timeout] | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: dict[str, Any] | None = None,
    **kwargs,
) -> Union[OCRResponse, Coroutine[Any, Any, OCRResponse]]:
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
    completion_kwargs: dict[str, object] = {
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
        _is_async = kwargs.pop("aocr", False) is True
        completion_kwargs["aocr"] = _is_async
        if rust_ocr_enabled():
            rust_ocr = load_rust_ocr()
            if rust_ocr is None:
                verbose_logger.debug(
                    "Rust OCR bridge unavailable; falling back to Python path"
                )
            else:
                return _run_rust_ocr(
                    rust_ocr=rust_ocr,
                    model=model,
                    document=document,
                    api_key=api_key,
                    api_base=api_base,
                    custom_llm_provider=custom_llm_provider,
                    extra_headers=extra_headers,
                    timeout=timeout,
                    kwargs=kwargs,
                )

        prepared = _prepare_ocr_request(
            model=model,
            document=document,
            api_key=api_key,
            api_base=api_base,
            kwargs=kwargs,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            timeout=timeout,
        )
        model = prepared.model
        custom_llm_provider = prepared.custom_llm_provider
        completion_kwargs.update(
            {"model": model, "custom_llm_provider": custom_llm_provider}
        )

        response = base_llm_http_handler.ocr(
            model=prepared.model,
            document=prepared.document,
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

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=completion_kwargs,
            extra_kwargs=kwargs,
        )
