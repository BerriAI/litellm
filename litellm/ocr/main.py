"""
Main OCR function for LiteLLM.
"""

import base64
import mimetypes
import os
import re
from dataclasses import dataclass
from io import IOBase
from typing import Any, Callable, Coroutine, Union, cast

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.constants import request_timeout
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.ocr.rust_bridge import (
    RustAocr,
    RustOcr,
    load_rust_aocr,
    load_rust_ocr,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import client


@dataclass
class _PreparedOCRRequest:
    model: str
    document: dict[str, Any]
    api_key: str | None
    api_base: str | None
    custom_llm_provider: str
    extra_headers: dict[str, object] | None
    optional_params: dict[str, object]
    litellm_params: dict[str, object]
    effective_timeout: Union[float, httpx.Timeout]
    litellm_logging_obj: LiteLLMLoggingObj


@dataclass
class _PreparedRustOCRCall:
    api_key: str | None
    api_base: str | None
    headers: dict[str, object]
    optional_params: dict[str, object]


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

    verbose_logger.debug(f"OCR call - model: {model}, provider: {custom_llm_provider}")

    litellm_params = GenericLiteLLMParams(**kwargs)
    optional_params = dict(kwargs)

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
        optional_params=cast(dict[str, object], optional_params),
        litellm_params=dict(litellm_params),
        effective_timeout=effective_timeout,
        litellm_logging_obj=litellm_logging_obj,
    )


def _rust_bridge_optional_params(
    prepared_request: _PreparedOCRRequest,
    resolve_secret: Callable[[str], str | None],
) -> dict[str, object]:
    optional_params = dict(prepared_request.optional_params)
    if prepared_request.custom_llm_provider == "vertex_ai":
        vertex_project = (
            prepared_request.litellm_params.get("vertex_project")
            or prepared_request.litellm_params.get("vertex_ai_project")
            or litellm.vertex_project
            or resolve_secret("VERTEXAI_PROJECT")
        )
        vertex_location = (
            prepared_request.litellm_params.get("vertex_location")
            or prepared_request.litellm_params.get("vertex_ai_location")
            or litellm.vertex_location
            or resolve_secret("VERTEXAI_LOCATION")
            or resolve_secret("VERTEX_LOCATION")
        )
        if vertex_project is not None:
            optional_params["vertex_project"] = vertex_project
        if vertex_location is not None:
            optional_params["vertex_location"] = vertex_location
    return optional_params


def _is_azure_document_intelligence(prepared_request: _PreparedOCRRequest) -> bool:
    return prepared_request.custom_llm_provider == "azure_ai/doc-intelligence" or (
        prepared_request.custom_llm_provider == "azure_ai"
        and (
            "doc-intelligence" in prepared_request.model
            or "documentintelligence" in prepared_request.model
        )
    )


def _rust_bridge_api_base(
    prepared_request: _PreparedOCRRequest,
    resolve_secret: Callable[[str], str | None],
) -> str | None:
    if prepared_request.api_base is not None:
        return prepared_request.api_base
    if _is_azure_document_intelligence(prepared_request):
        return resolve_secret("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    if prepared_request.custom_llm_provider == "azure_ai":
        return resolve_secret("AZURE_AI_API_BASE")
    return None


def _rust_bridge_api_key(
    prepared_request: _PreparedOCRRequest,
    resolve_api_key: Callable[[str], str | None],
) -> str | None:
    if prepared_request.api_key is not None:
        return prepared_request.api_key

    provider = prepared_request.custom_llm_provider
    if provider == "mistral":
        return resolve_api_key("MISTRAL_API_KEY")
    if provider == "azure_ai/doc-intelligence" or _is_azure_document_intelligence(
        prepared_request
    ):
        return resolve_api_key("AZURE_DOCUMENT_INTELLIGENCE_API_KEY")
    if provider == "azure_ai":
        return resolve_api_key("AZURE_AI_API_KEY")
    if provider == "vertex_ai":
        return resolve_api_key("VERTEX_AI_API_KEY") or resolve_api_key("VERTEXAI_API_KEY")
    if provider == "reducto":
        return resolve_api_key("REDUCTO_API_KEY")
    return None


def _prepare_rust_ocr_call(
    prepared_request: _PreparedOCRRequest,
    resolve_api_key: Callable[[str], str | None],
) -> _PreparedRustOCRCall:
    resolved_api_key = _rust_bridge_api_key(prepared_request, resolve_api_key)
    rust_api_base = _rust_bridge_api_base(prepared_request, resolve_api_key)
    rust_optional_params = _rust_bridge_optional_params(
        prepared_request, resolve_api_key
    )
    resolved_headers = prepared_request.extra_headers or {}
    prepared_request.litellm_logging_obj.pre_call(
        input="OCR document processing",
        api_key=resolved_api_key,
        additional_args={
            "complete_input_dict": {
                "model": prepared_request.model,
                "document": prepared_request.document,
                **rust_optional_params,
            },
            "api_base": rust_api_base or prepared_request.api_base,
            "headers": resolved_headers,
        },
    )
    return _PreparedRustOCRCall(
        api_key=resolved_api_key,
        api_base=rust_api_base,
        headers=cast(dict[str, object], resolved_headers),
        optional_params=rust_optional_params,
    )


def _run_rust_ocr(
    rust_ocr: RustOcr,
    prepared_request: _PreparedOCRRequest,
    resolve_api_key: Callable[[str], str | None],
) -> OCRResponse:
    prepared = _prepare_rust_ocr_call(
        prepared_request=prepared_request,
        resolve_api_key=resolve_api_key,
    )
    return OCRResponse.model_validate(
        rust_ocr(
            model=prepared_request.model,
            document=cast(dict[str, object], prepared_request.document),
            api_key=prepared.api_key,
            api_base=prepared.api_base,
            custom_llm_provider=prepared_request.custom_llm_provider,
            extra_headers=prepared.headers,
            optional_params=prepared.optional_params,
            timeout_seconds=_timeout_to_seconds(prepared_request.effective_timeout),
        )
    )


def _missing_rust_bridge_error() -> RuntimeError:
    return RuntimeError(
        "Rust OCR bridge is required for litellm.ocr()/litellm.aocr(), but the native extension is unavailable"
    )


async def _run_rust_aocr(
    rust_aocr: RustAocr,
    prepared_request: _PreparedOCRRequest,
    resolve_api_key: Callable[[str], str | None],
) -> OCRResponse:
    prepared = _prepare_rust_ocr_call(
        prepared_request=prepared_request,
        resolve_api_key=resolve_api_key,
    )
    return OCRResponse.model_validate(
        await rust_aocr(
            model=prepared_request.model,
            document=cast(dict[str, object], prepared_request.document),
            api_key=prepared.api_key,
            api_base=prepared.api_base,
            custom_llm_provider=prepared_request.custom_llm_provider,
            extra_headers=prepared.headers,
            optional_params=prepared.optional_params,
            timeout_seconds=_timeout_to_seconds(prepared_request.effective_timeout),
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

        rust_aocr = load_rust_aocr()
        if rust_aocr is None:
            raise _missing_rust_bridge_error()

        from litellm.secret_managers.main import get_secret_str

        return await _run_rust_aocr(
            rust_aocr=rust_aocr,
            prepared_request=prepared,
            resolve_api_key=get_secret_str,
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

        rust_ocr = load_rust_ocr()
        if rust_ocr is None:
            raise _missing_rust_bridge_error()

        from litellm.secret_managers.main import get_secret_str

        return _run_rust_ocr(
            rust_ocr=rust_ocr,
            prepared_request=prepared,
            resolve_api_key=get_secret_str,
        )
    except Exception as e:
        raise litellm.exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=completion_kwargs,
            extra_kwargs=kwargs,
        )
