"""
RAG Endpoints for LiteLLM Proxy.

Provides:
- /rag/ingest: All-in-one document ingestion pipeline (Upload -> Chunk -> Embed -> Vector Store)
- /rag/query: RAG query pipeline (Search -> Rerank -> LLM Completion)
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


@router.post(
    "/v1/rag/query",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["rag"],
)
@router.post(
    "/rag/query",
    dependencies=[Depends(user_api_key_auth)],
    response_class=ORJSONResponse,
    tags=["rag"],
)
async def rag_query(
    request: Request,
    fastapi_response: Response,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    RAG Query endpoint - search vector store, optionally rerank, and generate LLM response.

    This endpoint:
    1. Extracts the query from the last user message
    2. Searches the vector store for relevant context
    3. Optionally reranks the results
    4. Generates an LLM response with the retrieved context

    ## Example Request:
    ```bash
    curl -X POST "http://localhost:4000/v1/rag/query" \\
        -H "Authorization: Bearer sk-1234" \\
        -H "Content-Type: application/json" \\
        -d '{
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "What is LiteLLM?"}],
            "retrieval_config": {
                "vector_store_id": "vs_abc123",
                "custom_llm_provider": "openai",
                "top_k": 5
            }
        }'
    ```

    ## With Reranking:
    ```bash
    curl -X POST "http://localhost:4000/v1/rag/query" \\
        -H "Authorization: Bearer sk-1234" \\
        -H "Content-Type: application/json" \\
        -d '{
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "What is LiteLLM?"}],
            "retrieval_config": {
                "vector_store_id": "vs_abc123",
                "custom_llm_provider": "openai",
                "top_k": 10
            },
            "rerank": {
                "enabled": true,
                "model": "cohere/rerank-english-v3.0",
                "top_n": 3
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
        # Parse request body
        data = await _read_request_body(request)

        # Extract required fields
        model = data.get("model")
        messages = data.get("messages")
        retrieval_config = data.get("retrieval_config")
        rerank = data.get("rerank")
        stream = data.get("stream", False)

        # Validate required fields
        if not model:
            raise HTTPException(
                status_code=400,
                detail={"error": "model is required"},
            )
        if not messages:
            raise HTTPException(
                status_code=400,
                detail={"error": "messages is required"},
            )
        if not retrieval_config:
            raise HTTPException(
                status_code=400,
                detail={"error": "retrieval_config is required"},
            )
        if "vector_store_id" not in retrieval_config:
            raise HTTPException(
                status_code=400,
                detail={"error": "retrieval_config must contain 'vector_store_id'"},
            )

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

        verbose_proxy_logger.debug(
            f"RAG Query - model: {model}, retrieval_config: {retrieval_config}"
        )

        # Call query
        response = await litellm.aquery(
            model=model,
            messages=messages,
            retrieval_config=retrieval_config,
            rerank=rerank,
            stream=stream,
            router=llm_router,
            **request_data,
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception(f"RAG Query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": str(e)},
        )
