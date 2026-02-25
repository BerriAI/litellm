#### OCR Endpoints #####

import json
from typing import Any, Dict, Optional

import orjson
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import ORJSONResponse
from starlette.datastructures import UploadFile

from litellm._logging import verbose_proxy_logger
from litellm.ocr.main import _convert_file_document_to_url_document, _get_mime_type
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

router = APIRouter()


def _build_document_from_upload(
    file_content: bytes,
    filename: Optional[str],
    content_type: Optional[str],
) -> Dict[str, str]:
    """
    Convert uploaded file bytes into a Mistral-format document dict with base64 data URI.

    Delegates to _convert_file_document_to_url_document after resolving MIME type
    from the upload's content_type header or filename.
    """
    mime_type = content_type
    if not mime_type or mime_type == "application/octet-stream":
        if filename:
            mime_type = _get_mime_type(filename)

    return _convert_file_document_to_url_document(
        {"type": "file", "file": file_content, "mime_type": mime_type or "application/octet-stream"}
    )


async def _parse_multipart_form(request: Request) -> Dict[str, Any]:
    """
    Extract OCR data from a multipart form request.

    Uses the cached form if already parsed by auth middleware,
    otherwise parses the form from the request.

    Returns:
        A dict with 'document', 'model', and any other OCR params.
    """
    try:
        form = await request.form()
    except Exception as e:
        raise ValueError(
            f"Failed to parse multipart form data: {str(e)}. "
            "When using curl with --form/-F, do NOT set the Content-Type header "
            "manually — curl will set it automatically with the required boundary."
        )

    uploaded_file = form.get("file")
    if uploaded_file is None or not isinstance(uploaded_file, UploadFile):
        raise ValueError(
            "Multipart OCR request must include a 'file' field with the document to process"
        )

    # Seek to start in case the file was already partially read by middleware
    await uploaded_file.seek(0)
    file_content = await uploaded_file.read()
    if not file_content:
        raise ValueError("Uploaded file is empty")

    document = _build_document_from_upload(
        file_content=file_content,
        filename=uploaded_file.filename,
        content_type=uploaded_file.content_type,
    )

    data: Dict[str, Any] = {"document": document}

    for field_name, field_value in form.items():
        if field_name == "file":
            continue  # Already handled
        # Try to parse JSON values (e.g. pages=[0,1,2])
        if isinstance(field_value, str):
            try:
                data[field_name] = json.loads(field_value)
            except (json.JSONDecodeError, ValueError):
                data[field_name] = field_value
        else:
            data[field_name] = field_value

    verbose_proxy_logger.debug(
        f"OCR multipart form request parsed - model: {data.get('model')}, "
        f"document_type: {document['type']}, "
        f"filename: {uploaded_file.filename}"
    )

    return data


async def _parse_ocr_request(request: Request) -> Dict[str, Any]:
    """
    Parse an OCR request, supporting both JSON and multipart form data.

    JSON body (existing behavior):
        {
            "model": "mistral/mistral-ocr-latest",
            "document": {"type": "document_url", "document_url": "https://..."}
        }

    Multipart form data (new):
        - file: the uploaded file
        - model: model name (form field)
        - Any other OCR params as form fields (pages, include_image_base64, etc.)

    Returns:
        A dict suitable for passing to the OCR processing pipeline.
    """
    content_type = request.headers.get("content-type", "")

    if "multipart/form-data" in content_type.lower():
        return await _parse_multipart_form(request)

    # --- JSON body (existing behavior) ---
    try:
        body = await request.body()
    except RuntimeError:
        # Body stream was consumed by auth middleware (e.g., form parsing).
        body = b""

    if not body:
        # The body may be empty because the auth middleware already parsed
        # it as form data (e.g., _read_request_body called request.form()).
        # Check if form data is available.
        if getattr(request, "_form", None) is not None:
            verbose_proxy_logger.debug(
                "OCR request body is empty but form data is available from middleware — "
                "processing as multipart form."
            )
            return await _parse_multipart_form(request)

        raise ValueError(
            "Empty request body. For file uploads, use multipart/form-data content type "
            "with a file field. When using curl with --form/-F, do NOT set the Content-Type "
            "header manually."
        )

    return orjson.loads(body)


@router.post(
    "/v1/ocr",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["ocr"],
)
@router.post(
    "/ocr",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["ocr"],
)
async def ocr(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    OCR endpoint for extracting text from documents and images.

    Supports two input modes:

    **1. JSON body** (Mistral OCR API compatible):
    ```bash
    curl -X POST "http://localhost:4000/v1/ocr" \
        -H "Authorization: Bearer sk-1234" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "mistral-ocr",
            "document": {
                "type": "document_url",
                "document_url": "https://arxiv.org/pdf/2201.04234"
            }
        }'
    ```

    **2. Multipart form file upload**:
    ```bash
    curl -X POST "http://localhost:4000/v1/ocr" \
        -H "Authorization: Bearer sk-1234" \
        -F "model=mistral-ocr" \
        -F "file=@document.pdf"
    ```
    """
    from litellm.proxy.proxy_server import (
        general_settings,
        llm_router,
        proxy_config,
        proxy_logging_obj,
        select_data_generator,
        user_api_base,
        user_max_tokens,
        user_model,
        user_request_timeout,
        user_temperature,
        version,
    )

    data: dict = {}
    try:
        # Parse request body (JSON or multipart form)
        data = await _parse_ocr_request(request)

        # Process request using ProxyBaseLLMRequestProcessing
        processor = ProxyBaseLLMRequestProcessing(data=data)

        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aocr",
            proxy_logging_obj=proxy_logging_obj,
            llm_router=llm_router,
            general_settings=general_settings,
            proxy_config=proxy_config,
            select_data_generator=select_data_generator,
            model=None,
            user_model=user_model,
            user_temperature=user_temperature,
            user_request_timeout=user_request_timeout,
            user_max_tokens=user_max_tokens,
            user_api_base=user_api_base,
            version=version,
        )
    except Exception as e:
        processor = ProxyBaseLLMRequestProcessing(data=data)
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )

