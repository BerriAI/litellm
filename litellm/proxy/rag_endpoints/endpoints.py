"""
RAG Ingest Endpoints for LiteLLM Proxy.

Provides an all-in-one API for document ingestion:
Upload -> (OCR) -> Chunk -> Embed -> Vector Store
"""

import orjson
from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import ORJSONResponse

from litellm.proxy._types import *
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

router = APIRouter()


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

    Pipeline stages:
    1. Upload - Accept file data (URL, base64, or file_id)
    2. OCR (optional) - Extract text from images/PDFs
    3. Chunking - Split text into chunks
    4. Embedding - Generate embeddings (if needed by provider)
    5. Vector Store - Persist to vector store

    Example with file URL:
    ```bash
    curl -X POST "http://localhost:4000/v1/rag/ingest" \
        -H "Authorization: Bearer sk-1234" \
        -H "Content-Type: application/json" \
        -d '{
            "file_url": "https://example.com/document.pdf",
            "ingest_options": {
                "vector_store": {"custom_llm_provider": "openai"}
            }
        }'
    ```

    Example with base64 file content:
    ```bash
    curl -X POST "http://localhost:4000/v1/rag/ingest" \
        -H "Authorization: Bearer sk-1234" \
        -H "Content-Type: application/json" \
        -d '{
            "file": {
                "filename": "document.txt",
                "content": "base64_encoded_content_here",
                "content_type": "text/plain"
            },
            "ingest_options": {
                "vector_store": {"custom_llm_provider": "openai"}
            }
        }'
    ```

    Example with Bedrock (auto-creates KB):
    ```bash
    curl -X POST "http://localhost:4000/v1/rag/ingest" \
        -H "Authorization: Bearer sk-1234" \
        -H "Content-Type: application/json" \
        -d '{
            "file_url": "https://example.com/document.pdf",
            "ingest_options": {
                "vector_store": {"custom_llm_provider": "bedrock"}
            }
        }'
    ```

    Request body:
    ```json
    {
        "file_url": "https://...",  // URL to fetch file from
        "file_id": "file-abc123",   // OR existing file ID
        "file": {                    // OR inline file content
            "filename": "doc.pdf",
            "content": "base64...",
            "content_type": "application/pdf"
        },
        "ingest_options": {
            "name": "my-pipeline",  // optional
            "ocr": {"model": "mistral/mistral-ocr-latest"},  // optional
            "chunking_strategy": {"chunk_size": 1000, "chunk_overlap": 100},  // optional
            "vector_store": {
                "custom_llm_provider": "openai",  // or "bedrock"
                "vector_store_id": "vs_xxx"  // optional - creates new if not provided
            }
        }
    }
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

    # Read request body
    body = await request.body()
    data = orjson.loads(body)

    # Process request using ProxyBaseLLMRequestProcessing
    processor = ProxyBaseLLMRequestProcessing(data=data)
    try:
        return await processor.base_process_llm_request(
            request=request,
            fastapi_response=fastapi_response,
            user_api_key_dict=user_api_key_dict,
            route_type="aingest",
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
        raise await processor._handle_llm_api_exception(
            e=e,
            user_api_key_dict=user_api_key_dict,
            proxy_logging_obj=proxy_logging_obj,
            version=version,
        )
