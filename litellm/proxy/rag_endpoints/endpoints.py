"""
RAG Endpoints for LiteLLM Proxy.

Provides:
- /rag/ingest: All-in-one document ingestion pipeline (Upload -> Chunk -> Embed -> Vector Store)
- /rag/query: RAG query pipeline (Search -> Rerank -> LLM Completion)
"""

import base64
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

import orjson
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import ORJSONResponse, StreamingResponse

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import DEFAULT_MAX_RECURSE_DEPTH
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.proxy._types import *
from litellm.proxy.auth.auth_utils import is_request_body_safe
from litellm.proxy.auth.user_api_key_auth import UserAPIKeyAuth, user_api_key_auth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.common_utils.http_parsing_utils import (
    _read_request_body,
    _safe_get_request_headers,
    get_form_data,
)
from litellm.proxy.rag_endpoints.upload_security import (
    MAX_UPLOAD_SIZE_BYTES,
    EicarTestMalwareScanner,
    MalwareScanner,
    RejectedUpload,
    validate_upload,
)
from litellm.proxy.vector_store_endpoints.utils import (
    assert_user_can_access_vector_store_id,
)
from litellm.repositories.table_repositories import ManagedVectorStoresRepository

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

router: Final = APIRouter()


def _raise_vector_store_scan_depth_exceeded() -> None:
    raise HTTPException(
        status_code=400,
        detail={"error": f"Max depth of {DEFAULT_MAX_RECURSE_DEPTH} exceeded while scanning vector_store_id values"},
    )


def _append_payload_to_scan_stack(
    payload_stack: list[tuple[Any, int]],
    value: Any,
    next_depth: int,
) -> None:
    if isinstance(value, dict):
        if next_depth > DEFAULT_MAX_RECURSE_DEPTH:
            _raise_vector_store_scan_depth_exceeded()
        payload_stack.append((value, next_depth))
    elif isinstance(value, list):
        if next_depth > DEFAULT_MAX_RECURSE_DEPTH:
            if any(isinstance(item, (dict, list)) for item in value):
                _raise_vector_store_scan_depth_exceeded()
            return
        payload_stack.append((value, next_depth))


def _collect_vector_store_ids_from_payload(payload: object) -> set[str]:
    vector_store_ids: Final[set[str]] = set()
    payload_stack: Final = [(payload, 0)]

    while payload_stack:
        current_payload, depth = payload_stack.pop()
        if depth > DEFAULT_MAX_RECURSE_DEPTH:
            _raise_vector_store_scan_depth_exceeded()

        if isinstance(current_payload, dict):
            for key, value in current_payload.items():
                if key == "vector_store_id":
                    if not isinstance(value, str) or not value:
                        raise HTTPException(
                            status_code=400,
                            detail={"error": "vector_store_id must be a non-empty string"},
                        )
                    vector_store_ids.add(value)
                    continue
                if isinstance(value, (dict, list)):
                    _append_payload_to_scan_stack(
                        payload_stack=payload_stack,
                        value=value,
                        next_depth=depth + 1,
                    )
        elif isinstance(current_payload, list):
            for item in current_payload:
                _append_payload_to_scan_stack(
                    payload_stack=payload_stack,
                    value=item,
                    next_depth=depth + 1,
                )

    return vector_store_ids


async def _authorize_nested_vector_store_ids(
    payload: object,
    user_api_key_dict: UserAPIKeyAuth,
) -> None:
    for vector_store_id in sorted(_collect_vector_store_ids_from_payload(payload)):
        await assert_user_can_access_vector_store_id(
            vector_store_id=vector_store_id,
            user_api_key_dict=user_api_key_dict,
        )


def _build_file_metadata_entry(
    response: Any,
    file_data: tuple[str, bytes, str] | None = None,
    file_url: str | None = None,
) -> Mapping[str, str | int | None]:
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
    file_entry: Final = {
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
    ingest_options: Mapping[str, dict[str, str | None]],
    prisma_client: "PrismaClient",
    user_api_key_dict: UserAPIKeyAuth,
    file_data: tuple[str, bytes, str] | None = None,
    file_url: str | None = None,
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
        verbose_proxy_logger.warning("Unable to extract vector_store_id from response type: %s", type(response))
        return

    if vector_store_id is None or not isinstance(vector_store_id, str):
        verbose_proxy_logger.warning("Vector store ID is None or not a string, skipping database save")
        return

    vector_store_config: Final = ingest_options.get("vector_store", {})
    custom_llm_provider: Final = vector_store_config.get("custom_llm_provider")

    # Extract litellm_vector_store_params for custom name and description
    litellm_vector_store_params: Final = ingest_options.get("litellm_vector_store_params", {})
    custom_vector_store_name: Final = litellm_vector_store_params.get("vector_store_name")
    custom_vector_store_description: Final = litellm_vector_store_params.get("vector_store_description")

    # Extract provider-specific params from vector_store_config to save as litellm_params
    # This ensures params like aws_region_name, embedding_model, etc. are available for search
    provider_specific_params: Final = {}
    excluded_keys: Final = {"custom_llm_provider", "vector_store_id"}
    for key, value in vector_store_config.items():
        if key not in excluded_keys and value is not None:
            provider_specific_params[key] = value

    # Build file metadata entry using helper
    file_entry: Final = _build_file_metadata_entry(
        response=response,
        file_data=file_data,
        file_url=file_url,
    )

    try:
        # Check if vector store already exists in database
        existing_vector_store: Final = await ManagedVectorStoresRepository(prisma_client).table.find_unique(
            where={"vector_store_id": vector_store_id}
        )

        # Only create if it doesn't exist
        if existing_vector_store is None:
            verbose_proxy_logger.info("Saving newly created vector store %s to database", vector_store_id)

            # Initialize metadata with first file
            initial_metadata: Final = {"ingested_files": [file_entry]}

            # Use custom name if provided, otherwise default
            vector_store_name: Final = custom_vector_store_name or f"RAG Vector Store - {vector_store_id[:8]}"
            vector_store_description: Final = custom_vector_store_description or "Created via RAG ingest endpoint"

            await create_vector_store_in_db(
                vector_store_id=vector_store_id,
                custom_llm_provider=custom_llm_provider or "openai",
                prisma_client=prisma_client,
                vector_store_name=vector_store_name,
                vector_store_description=vector_store_description,
                vector_store_metadata=initial_metadata,
                litellm_params=(provider_specific_params if provider_specific_params else None),
                team_id=user_api_key_dict.team_id,
                user_id=user_api_key_dict.user_id,
            )

            verbose_proxy_logger.info("Vector store %s saved to database successfully", vector_store_id)
        else:
            verbose_proxy_logger.info("Vector store %s already exists, appending file to metadata", vector_store_id)

            # Update existing vector store with new file
            existing_metadata = existing_vector_store.vector_store_metadata or {}
            if isinstance(existing_metadata, str):
                import json

                existing_metadata = json.loads(existing_metadata)

            ingested_files: Final = existing_metadata.get("ingested_files", [])
            ingested_files.append(file_entry)
            existing_metadata["ingested_files"] = ingested_files

            # Update the vector store
            from litellm.proxy.utils import safe_dumps

            await ManagedVectorStoresRepository(prisma_client).table.update(
                where={"vector_store_id": vector_store_id},
                data={"vector_store_metadata": safe_dumps(existing_metadata)},
            )

            verbose_proxy_logger.info(
                "Added file %s to vector store %s metadata",
                file_entry.get("filename") or file_entry.get("file_url", "Unknown"),
                vector_store_id,
            )
    except Exception as db_error:
        # Log the error but don't fail the request since ingestion succeeded
        verbose_proxy_logger.exception("Failed to save vector store %s to database: %s", vector_store_id, db_error)


def _secure_uploaded_file(
    file_data: tuple[str, bytes, str],
    scanner: MalwareScanner,
) -> tuple[str, bytes, str]:
    validation: Final = validate_upload(content=file_data[1], scanner=scanner)
    if isinstance(validation, RejectedUpload):
        raise HTTPException(
            status_code=400,
            detail={"error": validation.message, "reason": validation.reason.value},
        )
    return validation.safe_filename, file_data[1], validation.content_type


async def parse_rag_ingest_request(
    request: Request,
    scanner: MalwareScanner,
) -> tuple[dict[str, Any], tuple[str, bytes, str] | None, str | None, str | None]:
    """
    Parse RAG ingest request.

    Supports:
    - Form: file + request JSON in form field
    - JSON body for URL-based ingestion

    Uploaded file bytes are validated against the vector-store upload controls
    (size limit, format allowlist with content inspection, archive rejection,
    and the injected malware scanner) and given a server-generated filename
    before they are returned.

    Returns:
        Tuple of (ingest_options, file_data, file_url, file_id)
    """
    headers: Final = _safe_get_request_headers(request)
    content_type = headers.get("content-type", "")

    file_data: tuple[str, bytes, str] | None = None
    file_url: str | None = None
    file_id: str | None = None
    ingest_options: dict[str, Any] = {}

    if "multipart/form-data" in content_type:
        # Form upload
        form_data: Final = await get_form_data(request)

        # Get file
        file_obj = form_data.get("file")
        if file_obj is not None and hasattr(file_obj, "read"):
            file_content = await file_obj.read(MAX_UPLOAD_SIZE_BYTES + 1)
            file_data = (file_obj.filename, file_content, file_obj.content_type)

        # Parse JSON from 'request' form field (contains full request body as JSON)
        request_json_str: Final[str | bytes | None] = form_data.get("request")
        if request_json_str:
            request_data: Final = orjson.loads(request_json_str)
            ingest_options = request_data.get("ingest_options", {})
            file_url = request_data.get("file_url")
            file_id = request_data.get("file_id")

    else:
        # JSON body
        data: Final = await _read_request_body(request)
        ingest_options = data.get("ingest_options", {})
        file_url = data.get("file_url")
        file_id = data.get("file_id")

        # Handle base64-encoded file in JSON body
        file_obj = data.get("file")
        if file_obj and isinstance(file_obj, dict):
            filename: Final = file_obj.get("filename")
            content_b64: Final = file_obj.get("content")
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

    secured_file_data: Final[tuple[str, bytes, str] | None] = (
        _secure_uploaded_file(file_data, scanner) if file_data is not None else None
    )

    if "vector_store" not in ingest_options:
        raise HTTPException(
            status_code=400,
            detail={"error": "ingest_options must contain 'vector_store' configuration"},
        )

    # Credential fields must come from server configuration, not user requests.
    # Accepting user-supplied credentials (e.g. vertex_credentials with
    # type=external_account + credential_source.file=/proc/1/environ) allows
    # any authenticated user to exfiltrate host secrets via SSRF through
    # google-auth's identity_pool credential refresh.
    # api_base is also blocked: a user-controlled base URL causes the server
    # to send its configured provider credentials to an attacker endpoint.
    _BLOCKED_VECTOR_STORE_CREDENTIAL_PARAMS: Final = {
        "vertex_credentials",
        "vertex_ai_credentials",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_session_token",
        "aws_web_identity_token",
        "aws_role_name",
        "aws_session_name",
        "aws_profile_name",
        "aws_sts_endpoint",
        "aws_external_id",
        "azure_ad_token",
        "api_key",
        "api_base",
    }
    vector_store_opts: Final[object] = ingest_options.get("vector_store", {})
    if isinstance(vector_store_opts, dict):
        for field in _BLOCKED_VECTOR_STORE_CREDENTIAL_PARAMS:
            if field in vector_store_opts:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": f"'{field}' cannot be set in ingest_options.vector_store. "
                        "Credentials must be configured server-side."
                    },
                )

    return ingest_options, secured_file_data, file_url, file_id


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
        ingest_options, file_data, file_url, file_id = await parse_rag_ingest_request(
            request, scanner=EicarTestMalwareScanner()
        )

        # INTERNAL_USER_VIEW_ONLY can ingest to existing vector stores only
        if user_api_key_dict.user_role == LitellmUserRoles.INTERNAL_USER_VIEW_ONLY.value and not ingest_options.get(
            "vector_store", {}
        ).get("vector_store_id"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "internal_user_viewer role can only ingest files to an existing vector store. "
                    "Provide 'vector_store_id' in ingest_options.vector_store."
                },
            )

        await _authorize_nested_vector_store_ids(
            payload=ingest_options,
            user_api_key_dict=user_api_key_dict,
        )

        try:
            is_request_body_safe(
                request_body=ingest_options.get("vector_store", {}),
                general_settings=general_settings,
                llm_router=llm_router,
                model="",
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail={"error": str(e)})

        # Add litellm data
        request_data: dict[str, Any] = {}
        request_data = await add_litellm_data_to_request(
            data=request_data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        verbose_proxy_logger.debug("RAG Ingest - options: %s", ingest_options)

        # Call ingest
        response: Final = await litellm.aingest(
            ingest_options=ingest_options,
            file_data=file_data,
            file_url=file_url,
            file_id=file_id,
            router=llm_router,
            **request_data,
        )

        # Save vector store to database if it was newly created and prisma_client is available
        verbose_proxy_logger.debug(
            "RAG Ingest - Checking database save conditions: prisma_client=%s, response=%s, response_type=%s",
            prisma_client is not None,
            response is not None,
            type(response),
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
                "Skipping database save: prisma_client=%s, response=%s", prisma_client is not None, response is not None
            )

        return response

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("RAG Ingest failed: %s", e)
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
        select_data_generator,
        version,
    )

    try:
        # Parse request body
        data: Final = await _read_request_body(request)

        # Extract required fields
        model: Final = data.get("model")
        messages: Final = data.get("messages")
        retrieval_config: Final = data.get("retrieval_config")
        rerank: Final = data.get("rerank")
        stream: Final = data.get("stream", False)

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
        if not isinstance(retrieval_config, dict):
            raise HTTPException(
                status_code=400,
                detail={"error": "retrieval_config must be an object"},
            )
        if "vector_store_id" not in retrieval_config:
            raise HTTPException(
                status_code=400,
                detail={"error": "retrieval_config must contain 'vector_store_id'"},
            )
        await _authorize_nested_vector_store_ids(
            payload=retrieval_config,
            user_api_key_dict=user_api_key_dict,
        )

        # Add litellm data
        request_data: dict[str, object] = {}
        request_data = await add_litellm_data_to_request(
            data=request_data,
            request=request,
            general_settings=general_settings,
            user_api_key_dict=user_api_key_dict,
            version=version,
            proxy_config=proxy_config,
        )

        verbose_proxy_logger.debug("RAG Query - model: %s, retrieval_config: %s", model, retrieval_config)

        # Call query
        response: Final = await litellm.aquery(
            model=model,
            messages=messages,
            retrieval_config=retrieval_config,
            rerank=rerank,
            stream=stream,
            router=llm_router,
            **request_data,
        )

        hidden_params: Final = getattr(response, "_hidden_params", {}) or {}
        custom_headers: Final = ProxyBaseLLMRequestProcessing.get_custom_headers(
            user_api_key_dict=user_api_key_dict,
            call_id=hidden_params.get("litellm_call_id", None) or "",
            model_id=hidden_params.get("model_id", None) or "",
            cache_key=hidden_params.get("cache_key", None) or "",
            api_base=hidden_params.get("api_base", None) or "",
            version=version,
            response_cost=hidden_params.get("response_cost", None),
            request_data=request_data,
        )

        if isinstance(response, CustomStreamWrapper):
            return StreamingResponse(
                select_data_generator(
                    response=response,
                    user_api_key_dict=user_api_key_dict,
                    request_data=request_data,
                    request=request,
                ),
                media_type="text/event-stream",
                headers=custom_headers,
            )

        fastapi_response.headers.update(custom_headers)
        return response

    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("RAG Query failed: %s", e)
        raise HTTPException(
            status_code=500,
            detail={"error": str(e)},
        )
