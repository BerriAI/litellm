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
    Parse RAG ingest request - supports both Form upload and JSON body.

    Returns:
        Tuple of (ingest_options, file_data, file_url, file_id)
    """
    headers = _safe_get_request_headers(request)
    content_type = headers.get("content-type", "")

    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
        # Form-based upload (like OpenAI files API)
        form_data = await get_form_data(request)

        ingest_options_str = form_data.get("ingest_options")
        if not ingest_options_str:
            raise HTTPException(
                status_code=400,
                detail={"error": "ingest_options is required"},
            )

        ingest_options = orjson.loads(ingest_options_str)

        # Handle file upload
        file_data = None
        file_obj = form_data.get("file")
        if file_obj is not None and hasattr(file_obj, "read"):
            file_content = await file_obj.read()
            file_data = (file_obj.filename, file_content, file_obj.content_type)

        file_url = form_data.get("file_url")
        file_id = form_data.get("file_id")

    else:
        # JSON body
        data = await _read_request_body(request)

        ingest_options = data.get("ingest_options", {})
        file_url = data.get("file_url")
        file_id = data.get("file_id")
        file_data = None

        # Handle inline file content (base64)
        if "file" in data and data["file"]:
            file_obj = data["file"]
            filename = file_obj.get("filename", "document")
            content_b64 = file_obj.get("content", "")
            content_type_str = file_obj.get("content_type", "application/octet-stream")
            content_bytes = base64.b64decode(content_b64)
            file_data = (filename, content_bytes, content_type_str)

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

    Supports two formats:
    1. Form upload (like OpenAI files API) - simple file upload
    2. JSON body - for programmatic use with base64 content or URLs

    ## Form Upload (Recommended for files):
    ```bash
    curl -X POST "http://localhost:4000/v1/rag/ingest" \\
        -H "Authorization: Bearer sk-1234" \\
        -F file="@document.pdf" \\
        -F 'ingest_options={"vector_store": {"custom_llm_provider": "openai"}}'
    ```

    ## Form Upload for Bedrock:
    ```bash
    curl -X POST "http://localhost:4000/v1/rag/ingest" \\
        -H "Authorization: Bearer sk-1234" \\
        -F file="@document.pdf" \\
        -F 'ingest_options={"vector_store": {"custom_llm_provider": "bedrock"}}'
    ```

    ## JSON Body (for URLs or base64):
    ```bash
    curl -X POST "http://localhost:4000/v1/rag/ingest" \\
        -H "Authorization: Bearer sk-1234" \\
        -H "Content-Type: application/json" \\
        -d '{
            "file_url": "https://example.com/document.pdf",
            "ingest_options": {
                "vector_store": {"custom_llm_provider": "openai"}
            }
        }'
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
        # Parse request (form or JSON)
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
