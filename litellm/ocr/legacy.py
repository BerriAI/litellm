"""
Main OCR function for LiteLLM.
"""

import asyncio
import base64
import mimetypes
import os
import re
from collections.abc import Coroutine, Mapping
from dataclasses import dataclass
from io import IOBase
from types import MappingProxyType
from typing import Final, cast  # noqa: TID251  # adapters preserve the legacy untyped contracts

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
from litellm.ocr.input import FileReader
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager, client

base_llm_http_handler: Final = BaseLLMHTTPHandler()


@dataclass(frozen=True, slots=True)
class _PreparedOCRRequest:
    model: str
    document: Mapping[str, object]
    api_key: str | None
    api_base: str | None
    custom_llm_provider: str
    extra_headers: dict[str, object] | None
    provider_config: BaseOCRConfig
    optional_params: dict[str, object]
    litellm_params: dict[str, object]
    effective_timeout: float | httpx.Timeout
    litellm_logging_obj: LiteLLMLoggingObj


def _prepare_ocr_request(
    model: str,
    document: Mapping[str, object],
    api_key: str | None,
    api_base: str | None,
    timeout: float | httpx.Timeout | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,
    kwargs: dict[str, object],
) -> _PreparedOCRRequest:
    litellm_logging_obj: Final = cast(  # cast-ok: @client supplies the logging object; preserve legacy failure behavior
        LiteLLMLoggingObj, kwargs.pop("litellm_logging_obj")
    )
    litellm_call_id: Final = cast(  # cast-ok: @client supplies the call id without coercion
        str | None, kwargs.get("litellm_call_id", None)
    )

    if not isinstance(document, dict):
        raise ValueError(f"document must be a dict with 'type' and URL/file field, got {type(document)}")

    doc_type = document.get("type")

    if doc_type == "file":
        document = convert_file_document_to_url_document(document)
        doc_type = document.get("type")

    if doc_type not in ["document_url", "image_url"]:
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
        optional_params=cast(
            dict[str, object], optional_params
        ),  # cast-ok: provider configs return heterogeneous OCR options
        litellm_params=dict(litellm_params),
        effective_timeout=effective_timeout,
        litellm_logging_obj=litellm_logging_obj,
    )


def _error_provider(model: str, custom_llm_provider: str | None) -> str | None:
    if custom_llm_provider is not None:
        return custom_llm_provider
    prefix: Final = model.partition("/")[0]
    if prefix in {"mistral", "azure_ai", "vertex_ai"}:
        return prefix
    return "mistral" if model.startswith("mistral-ocr") else None


@client
async def aocr(
    model: str,
    document: Mapping[str, object],
    api_key: str | None = None,
    api_base: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    custom_llm_provider: str | None = None,
    extra_headers: dict[str, object] | None = None,
    **kwargs: object,  # kwargs-ok: public OCR accepts provider-specific options
) -> OCRResponse:
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
        prepared: Final = _prepare_ocr_request(
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
        completion_kwargs.update({"model": model, "custom_llm_provider": custom_llm_provider})

        response = base_llm_http_handler.ocr(
            model=prepared.model,
            document=cast(  # cast-ok: preserve legacy document fields for provider validation
                dict[str, str], prepared.document
            ),
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
            raise ValueError(f"Got an unexpected None response from the OCR API: {response}")

        return response
    except Exception as e:
        error_provider: Final = _error_provider(model, custom_llm_provider)
        error_model: Final = model.removeprefix(f"{error_provider}/") if error_provider else model
        raise litellm.exception_type(
            model=error_model,
            custom_llm_provider=error_provider,
            original_exception=e,
            completion_kwargs=completion_kwargs,
            extra_kwargs=kwargs,
        )


_MIME_PATTERN: Final = re.compile(r"^[\w.+-]+/[\w.+-]+$")

_MIME_TYPE_MAP: Final = MappingProxyType(
    {
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
)


def get_mime_type(file_path: str) -> str:
    ext: Final = os.path.splitext(file_path)[1].lower()
    mime: Final = _MIME_TYPE_MAP.get(ext)
    if mime:
        return mime
    guessed, _ = mimetypes.guess_type(file_path)
    return guessed or "application/octet-stream"


def _read_file(file_input: object) -> tuple[bytes, str, str | None]:
    if isinstance(file_input, str):
        raise ValueError(
            "OCR file input does not accept bare str values. Pass bytes, "
            "a pathlib.Path, or a file-like object. To OCR a local file "
            "from a path, call open(path, 'rb') yourself."
        )
    if isinstance(file_input, os.PathLike):
        file_path: Final = str(cast(object, file_input))  # cast-ok: preserve staging's str(PathLike) conversion
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        mime_type: Final = get_mime_type(file_path)
        with open(file_path, "rb") as stream:
            return stream.read(), mime_type, os.path.basename(file_path)
    if isinstance(file_input, bytes):
        return file_input, "application/octet-stream", None
    if isinstance(file_input, IOBase) or hasattr(file_input, "read"):
        file_name: Final = cast(  # cast-ok: retain legacy validation and errors for file-like metadata
            str | None, getattr(file_input, "name", None)
        )
        inferred_mime: Final = get_mime_type(file_name) if file_name else "application/octet-stream"
        reader: Final = cast(FileReader, file_input)  # cast-ok: legacy accepts duck-typed file readers
        content: Final = reader.read()
        return content.encode("utf-8") if isinstance(content, str) else content, inferred_mime, file_name
    raise ValueError(
        f"Unsupported file input type: {type(file_input)}. Expected pathlib.Path, bytes, or a file-like object."
    )


def convert_file_document_to_url_document(document: Mapping[str, object]) -> dict[str, str]:
    file_input: Final = document.get("file")
    if file_input is None:
        raise ValueError(
            "document with type='file' must include a 'file' field containing "
            "a pathlib.Path, file-like object, or bytes"
        )
    file_bytes, inferred_mime, file_name = _read_file(file_input)
    if not file_bytes:
        raise ValueError("File is empty or could not be read")
    mime_type: Final = cast(  # cast-ok: keep staging's MIME validation errors
        str, document.get("mime_type", inferred_mime)
    )
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
    **kwargs: object,  # kwargs-ok: public OCR accepts provider-specific options
) -> OCRResponse | Coroutine[object, object, OCRResponse]:
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
        prepared: Final = _prepare_ocr_request(
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
        completion_kwargs.update({"model": model, "custom_llm_provider": custom_llm_provider})

        response: Final = base_llm_http_handler.ocr(
            model=prepared.model,
            document=cast(  # cast-ok: preserve legacy document fields for provider validation
                dict[str, str], prepared.document
            ),
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
        error_provider: Final = _error_provider(model, custom_llm_provider)
        error_model: Final = model.removeprefix(f"{error_provider}/") if error_provider else model
        raise litellm.exception_type(
            model=error_model,
            custom_llm_provider=error_provider,
            original_exception=e,
            completion_kwargs=completion_kwargs,
            extra_kwargs=kwargs,
        )
