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


def _build_file_metadata_entry(
    response: Any,
    file_data: Optional[Tuple[str, bytes, str]] = None,
    file_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build a file metadata entry for storing in vector_store_metadata.
    
    Args:
        response: The response from litellm.aingest containing file_id
        file_data: Optional tuple of (filename, content, content_type)
        file_url: Optional URL if file was ingested from URL
    
    Returns:
        Dictionary with file metadata (file_id, filename, file_url, ingested_at, etc.)
    """
    from datetime import datetime, timezone

    # Extract file_id from response
    file_id = None
    if hasattr(response, "get"):
        file_id = response.get("file_id")
    elif hasattr(response, "file_id"):
        file_id = response.file_id
    
    # Extract file information from file_data tuple
    filename = None
    file_size = None
    content_type = None
    
    if file_data:
        filename = file_data[0]
        file_size = len(file_data[1]) if len(file_data) > 1 else None
        content_type = file_data[2] if len(file_data) > 2 else None
    
    # Build file metadata entry
    file_entry = {
        "file_id": file_id,
        "filename": filename,
        "file_url": file_url,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    
    # Add optional fields if available
    if file_size is not None:
        file_entry["file_size"] = file_size
    if content_type is not None:
        file_entry["content_type"] = content_type
    
    return file_entry


async def _save_vector_store_to_db_from_rag_ingest(
    response: Any,
    ingest_options: Dict[str, Any],
    prisma_client,
    user_api_key_dict: UserAPIKeyAuth,
    file_data: Optional[Tuple[str, bytes, str]] = None,
    file_url: Optional[str] = None,
) -> None:
    """
    Helper function to save a newly created vector store from RAG ingest to the database.
    
    This function:
    - Extracts vector store ID and config from the ingest response
    - Checks if the vector store already exists in the database
    - Creates a new database entry if it doesn't exist
    - Adds the vector store to the registry
    - Tracks team_id and user_id for access control
    
    Args:
        response: The response from litellm.aingest()
        ingest_options: The ingest options containing vector store config
        prisma_client: The Prisma database client
        user_api_key_dict: User API key authentication info
    """
    from litellm.proxy.vector_store_endpoints.management_endpoints import (
        create_vector_store_in_db,
    )

    # Handle both dict and object responses
    if hasattr(response, "get"):
        vector_store_id = response.get("vector_store_id")
    elif hasattr(response, "vector_store_id"):
        vector_store_id = response.vector_store_id
    else:
        verbose_proxy_logger.warning(
            f"Unable to extract vector_store_id from response type: {type(response)}"
        )
        return

    if vector_store_id is None or not isinstance(vector_store_id, str):
        verbose_proxy_logger.warning(
            "Vector store ID is None or not a string, skipping database save"
        )
        return

    vector_store_config = ingest_options.get("vector_store", {})
    custom_llm_provider = vector_store_config.get("custom_llm_provider")
    
    # Extract litellm_vector_store_params for custom name and description
    litellm_vector_store_params = ingest_options.get("litellm_vector_store_params", {})
    custom_vector_store_name = litellm_vector_store_params.get("vector_store_name")
    custom_vector_store_description = litellm_vector_store_params.get("vector_store_description")
    
    # Extract provider-specific params from vector_store_config to save as litellm_params
    # This ensures params like aws_region_name, embedding_model, etc. are available for search
    provider_specific_params = {}
    excluded_keys = {"custom_llm_provider", "vector_store_id"}
    for key, value in vector_store_config.items():
        if key not in excluded_keys and value is not None:
            provider_specific_params[key] = value

    # Build file metadata entry using helper
    file_entry = _build_file_metadata_entry(
        response=response,
        file_data=file_data,
        file_url=file_url,
    )

    try:
        # Check if vector store already exists in database
        existing_vector_store = (
            await prisma_client.db.litellm_managedvectorstorestable.find_unique(
                where={"vector_store_id": vector_store_id}
            )
        )

        # Only create if it doesn't exist
        if existing_vector_store is None:
            verbose_proxy_logger.info(
                f"Saving newly created vector store {vector_store_id} to database"
            )

            # Initialize metadata with first file
            initial_metadata = {
                "ingested_files": [file_entry]
            }
            
            # Use custom name if provided, otherwise default
            vector_store_name = custom_vector_store_name or f"RAG Vector Store - {vector_store_id[:8]}"
            vector_store_description = custom_vector_store_description or "Created via RAG ingest endpoint"

            await create_vector_store_in_db(
                vector_store_id=vector_store_id,
                custom_llm_provider=custom_llm_provider or "openai",
                prisma_client=prisma_client,
                vector_store_name=vector_store_name,
                vector_store_description=vector_store_description,
                vector_store_metadata=initial_metadata,
                litellm_params=provider_specific_params if provider_specific_params else None,
                team_id=user_api_key_dict.team_id,
                user_id=user_api_key_dict.user_id,
            )

            verbose_proxy_logger.info(
                f"Vector store {vector_store_id} saved to database successfully"
            )
        else:
            verbose_proxy_logger.info(
                f"Vector store {vector_store_id} already exists, appending file to metadata"
            )
            
            # Update existing vector store with new file
            existing_metadata = existing_vector_store.vector_store_metadata or {}
            if isinstance(existing_metadata, str):
                import json
                existing_metadata = json.loads(existing_metadata)
            
            ingested_files = existing_metadata.get("ingested_files", [])
            ingested_files.append(file_entry)
            existing_metadata["ingested_files"] = ingested_files
            
            # Update the vector store
            from litellm.proxy.utils import safe_dumps
            await prisma_client.db.litellm_managedvectorstorestable.update(
                where={"vector_store_id": vector_store_id},
                data={"vector_store_metadata": safe_dumps(existing_metadata)}
            )
            
            verbose_proxy_logger.info(
                f"Added file {file_entry.get('filename') or file_entry.get('file_url', 'Unknown')} to vector store {vector_store_id} metadata"
            )
    except Exception as db_error:
        # Log the error but don't fail the request since ingestion succeeded
        verbose_proxy_logger.exception(
            f"Failed to save vector store {vector_store_id} to database: {db_error}"
        )


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
        prisma_client,
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

        # Save vector store to database if it was newly created and prisma_client is available
        verbose_proxy_logger.debug(
            f"RAG Ingest - Checking database save conditions: prisma_client={prisma_client is not None}, response={response is not None}, response_type={type(response)}"
        )
        
        if prisma_client is not None and response is not None:
            await _save_vector_store_to_db_from_rag_ingest(
                response=response,
                ingest_options=ingest_options,
                prisma_client=prisma_client,
                user_api_key_dict=user_api_key_dict,
                file_data=file_data,
                file_url=file_url,
            )
        else:
            verbose_proxy_logger.warning(
                f"Skipping database save: prisma_client={prisma_client is not None}, response={response is not None}"
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
