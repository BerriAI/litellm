"""
Main OCR function for LiteLLM.
"""
import asyncio
import base64
import contextvars
import mimetypes
import os
import re
from functools import partial
from io import IOBase
from pathlib import Path
from typing import Any, Coroutine, Dict, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.constants import request_timeout
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.ocr.transformation import BaseOCRConfig, OCRResponse
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager, client

####### ENVIRONMENT VARIABLES ###################
base_llm_http_handler = BaseLLMHTTPHandler()
#################################################


@client
async def aocr(
    model: str,
    document: Dict[str, Any],
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
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
    local_vars = locals()
    try:
        loop = asyncio.get_event_loop()
        kwargs["aocr"] = True

        # Get custom llm provider
        if custom_llm_provider is None:
            _, custom_llm_provider, _, _ = litellm.get_llm_provider(
                model=model, api_base=api_base
            )

        func = partial(
            ocr,
            model=model,
            document=document,
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            **kwargs,
        )

        ctx = contextvars.copy_context()
        func_with_context = partial(ctx.run, func)
        init_response = await loop.run_in_executor(None, func_with_context)

        if asyncio.iscoroutine(init_response):
            response = await init_response
        else:
            response = init_response

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
            completion_kwargs=local_vars,
            extra_kwargs=kwargs,
        )


@client
def ocr(
    model: str,
    document: Dict[str, Any],
    api_key: Optional[str] = None,
    api_base: Optional[str] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
    custom_llm_provider: Optional[str] = None,
    extra_headers: Optional[Dict[str, Any]] = None,
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
    local_vars = locals()
    try:
        litellm_logging_obj: LiteLLMLoggingObj = kwargs.pop("litellm_logging_obj")  # type: ignore
        litellm_call_id: Optional[str] = kwargs.get("litellm_call_id", None)
        _is_async = kwargs.pop("aocr", False) is True

        # Validate document parameter format
        if not isinstance(document, dict):
            raise ValueError(
                f"document must be a dict with 'type' and URL/file field, got {type(document)}"
            )

        doc_type = document.get("type")

        # Handle file type: convert to document_url/image_url with base64 data URI
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

        # Update with dynamic values if available
        if dynamic_api_key:
            api_key = dynamic_api_key
        if dynamic_api_base:
            api_base = dynamic_api_base

        # Get provider config
        ocr_provider_config: Optional[
            BaseOCRConfig
        ] = ProviderConfigManager.get_provider_ocr_config(
            model=model,
            provider=litellm.LlmProviders(custom_llm_provider),
        )

        if ocr_provider_config is None:
            raise ValueError(
                f"OCR is not supported for provider: {custom_llm_provider}"
            )

        verbose_logger.debug(
            f"OCR call - model: {model}, provider: {custom_llm_provider}"
        )

        # Get litellm params using GenericLiteLLMParams (same as responses API)
        litellm_params = GenericLiteLLMParams(**kwargs)

        # Extract OCR-specific parameters from kwargs
        supported_params = ocr_provider_config.get_supported_ocr_params(model=model)
        non_default_params = {}
        for param in supported_params:
            if param in kwargs:
                non_default_params[param] = kwargs.pop(param)

        # Map parameters to provider-specific format
        optional_params = ocr_provider_config.map_ocr_params(
            non_default_params=non_default_params,
            optional_params={},
            model=model,
        )

        verbose_logger.debug(f"OCR optional_params after mapping: {optional_params}")

        # Pre Call logging
        litellm_logging_obj.update_environment_variables(
            model=model,
            optional_params=optional_params,
            litellm_params={
                "litellm_call_id": litellm_call_id,
                "api_base": api_base,
            },
            custom_llm_provider=custom_llm_provider,
        )

        # Call the handler - pass document dict directly
        response = base_llm_http_handler.ocr(
            model=model,
            document=document,  # Pass the entire document dict
            optional_params=optional_params,
            timeout=timeout or request_timeout,
            logging_obj=litellm_logging_obj,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            aocr=_is_async,
            headers=extra_headers,
            provider_config=ocr_provider_config,
            litellm_params=dict(litellm_params),
        )

        return response
    except Exception as e:
        raise litellm.exception_type(
            model=model,
            custom_llm_provider=custom_llm_provider,
            original_exception=e,
            completion_kwargs=local_vars,
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


def convert_file_document_to_url_document(document: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert a file-type document dict to a document_url-type document dict
    with an inline base64 data URI.

    Accepts document dicts like:
        {"type": "file", "file": "/path/to/document.pdf"}        # file path string
        {"type": "file", "file": Path("/path/to/doc.pdf")}       # pathlib.Path
        {"type": "file", "file": <binary file-like object>}      # file-like object (BinaryIO)
        {"type": "file", "file": b"raw bytes"}                   # raw bytes

    Returns:
        {"type": "document_url", "document_url": "data:<mime>;base64,<data>"}
        or {"type": "image_url", "image_url": "data:<mime>;base64,<data>"}
    """
    file_input = document.get("file")
    if file_input is None:
        raise ValueError(
            "document with type='file' must include a 'file' field containing "
            "a file path (str), pathlib.Path, file-like object, or bytes"
        )

    file_bytes: bytes
    mime_type: str = "application/octet-stream"
    file_name: Optional[str] = None

    if isinstance(file_input, (str, Path)):
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
            "Expected str (file path), pathlib.Path, bytes, or a file-like object."
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
    else:
        verbose_logger.debug(
            f"OCR file input: Converted file to document_url data URI "
            f"(mime={mime_type}, size={len(file_bytes)} bytes, name={file_name})"
        )
        return {"type": "document_url", "document_url": data_uri}
