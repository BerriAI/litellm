"""
RAG Ingest Endpoints for LiteLLM Proxy.

Provides an all-in-one API for document ingestion:
Upload -> (OCR) -> Chunk -> Embed -> Vector Store
"""

import base64
from typing import Any, Dict, Optional, Tuple

import orjson
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import ORJSONResponse

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_utils.http_parsing_utils import (
    _read_request_body,
    _safe_get_request_headers,
    get_form_data,
)

router = APIRouter()


async def parse_rag_ingest_request(
    request: Request,
) -> Tuple[Dict[str, Any], Optional[Tuple[str, bytes, str]], Optional[str], Optional[str]]:
    """
    Parse RAG ingest request.

    Supports:
    - Form: file + request JSON in form field
    - JSON body for URL-based ingestion

    Returns:
        Tuple of (ingest_options, file_data, file_url, file_id)
    """
    headers = _safe_get_request_headers(request)
    content_type = headers.get("content-type", "")

    file_data = None
    file_url = None
    file_id = None
    ingest_options: Dict[str, Any] = {}

    if "multipart/form-data" in content_type:
        # Form upload
        form_data = await get_form_data(request)

        # Get file
        file_obj = form_data.get("file")
        if file_obj is not None and hasattr(file_obj, "read"):
            file_content = await file_obj.read()
            file_data = (file_obj.filename, file_content, file_obj.content_type)

        # Parse JSON from 'request' form field (contains full request body as JSON)
        request_json_str = form_data.get("request")
        if request_json_str:
            request_data = orjson.loads(request_json_str)
            ingest_options = request_data.get("ingest_options", {})
            file_url = request_data.get("file_url")
            file_id = request_data.get("file_id")

    else:
        # JSON body
        data = await _read_request_body(request)
        ingest_options = data.get("ingest_options", {})
        file_url = data.get("file_url")
        file_id = data.get("file_id")

        # Handle base64-encoded file in JSON body
        file_obj = data.get("file")
        if file_obj and isinstance(file_obj, dict):
            filename = file_obj.get("filename")
            content_b64 = file_obj.get("content")
            content_type = file_obj.get("content_type", "application/octet-stream")

            if filename and content_b64:
                try:
                    file_content = base64.b64decode(content_b64)
                    file_data = (filename, file_content, content_type)
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail={"error": f"Invalid base64 content: {e}"},
                    )

    # Validate
    if file_data is None and file_url is None and file_id is None:
        raise HTTPException(
            status_code=400,
            detail={"error": "Must provide file, file_url, or file_id"},
        )

    if "vector_store" not in ingest_options:
        raise HTTPException(
            status_code=400,
            detail={"error": "ingest_options must contain 'vector_store' configuration"},
        )

    return ingest_options, file_data, file_url, file_id


@router.post(
    "/v1/rag/ingest",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["rag"],
)
@router.post(
    "/rag/ingest",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["rag"],
)
async def rag_ingest(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    RAG Ingest endpoint - all-in-one document ingestion pipeline.

    Supports form upload (for files) or JSON body (for URLs).

    ## Form upload (for files):
    ```bash
    curl -X POST "http://localhost:4000/v1/rag/ingest" \\
        -H "Authorization: Bearer sk-1234" \\
        -F file="@document.pdf" \\
        -F 'ingest_options={"vector_store": {"custom_llm_provider": "openai"}}'
    ```

    ## JSON body (for URLs):
    ```bash
    curl -X POST "http://localhost:4000/v1/rag/ingest" \\
        -H "Authorization: Bearer sk-1234" \\
        -H "Content-Type: application/json" \\
        -d '{
            "file_url": "https://example.com/document.pdf",
            "ingest_options": {"vector_store": {"custom_llm_provider": "openai"}}
        }'
    ```

    ## Bedrock:
    ```bash
    curl -X POST "http://localhost:4000/v1/rag/ingest" \\
        -H "Authorization: Bearer sk-1234" \\
        -F file="@document.pdf" \\
        -F 'ingest_options={"vector_store": {"custom_llm_provider": "bedrock"}}'
    ```
    """
    from litellm.proxy.proxy_server import (
        add_litellm_data_to_request,
        general_settings,
        llm_router,
        proxy_config,
        version,
    )

    try:
        # Parse request
        ingest_options, file_data, file_url, file_id = await parse_rag_ingest_request(request)

        # Add litellm data
        request_data: Dict[str, Any] = {}
        request_data = await add_litellm_data_to_request(
            data=request_data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        verbose_proxy_logger.debug(f"RAG Ingest - options: {ingest_options}")

        # Call ingest
        response = await litellm.aingest(
            ingest_options=ingest_options,
            file_data=file_data,
            file_url=file_url,
            file_id=file_id,
            router=llm_router,
            **request_data,
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"RAG Ingest failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": str(e)},
        )
