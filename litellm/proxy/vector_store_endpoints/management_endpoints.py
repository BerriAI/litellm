"""
VECTOR STORE MANAGEMENT

All /vector_store management endpoints

/vector_store/new
/vector_store/delete
/vector_store/list
"""

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

if TYPE_CHECKING:
    from prisma.models import LiteLLM_ManagedVectorStoresTable as _VectorStoreRow

    from litellm.proxy.utils import PrismaClient
import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import MILVUS_ADMIN_CONFIGURED_CONNECTION, REDACTED_BY_LITELM_STRING
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.proxy._types import (
    LiteLLM_ManagedVectorStoresTable,
    ResponseLiteLLM_ManagedVectorStore,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.rbac_utils import check_feature_access_for_user
from litellm.proxy.vector_store_endpoints.utils import (
    can_user_access_vector_store,
    filter_listable_vector_stores,
    prepare_vector_store_connection_for_persistence,
)
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.table_repositories import ManagedVectorStoresRepository
from litellm.types.vector_stores import (
    LiteLLM_ManagedVectorStore,
    LiteLLM_ManagedVectorStoreListResponse,
    VectorStoreDeleteRequest,
    VectorStoreInfoRequest,
    VectorStoreUpdateRequest,
)
from litellm.vector_stores.vector_store_registry import (
    VectorStoreRegistry,
    deserialize_litellm_params,
)

router: Final = APIRouter()


def _vector_store_table(prisma_client: "PrismaClient") -> "TableActions[_VectorStoreRow]":
    return ManagedVectorStoresRepository(prisma_client).table


def _row_to_vector_store(row: "_VectorStoreRow") -> LiteLLM_ManagedVectorStore:
    return LiteLLM_ManagedVectorStore(**row.model_dump())


_LITELLM_PARAMS_MASKER: Final = SensitiveDataMasker(extra_sensitive_patterns=frozenset(("connection",)))


_REDACT_LITELLM_PARAMS_MAX_DEPTH: Final = 10


def _redact_sensitive_litellm_params(litellm_params: object, _depth: int = 0) -> Any:
    """
    Replace credential-bearing values in ``litellm_params`` with
    ``REDACTED_BY_LITELM`` while preserving non-secret keys (``api_base``,
    ``region``, ``model``, ``api_version``).

    Handles three input shapes:

    * ``dict`` — recurse into nested dicts (e.g. ``litellm_embedding_config``
      which itself carries ``api_key`` / ``aws_*`` / ``vertex_credentials``).
    * ``str`` — the in-memory registry occasionally holds the params as a
      JSON-serialized string. Parse, redact, re-serialize. If parsing
      fails, return the redaction sentinel rather than echo the value
      back verbatim.
    * Anything else, or ``None`` — passed through.

    Recursion depth is bounded by ``_REDACT_LITELLM_PARAMS_MAX_DEPTH`` —
    matching the convention of other allowlisted recursive helpers in the
    repo (see ``tests/code_coverage_tests/recursive_detector.py``).
    """
    if _depth >= _REDACT_LITELLM_PARAMS_MAX_DEPTH:
        return REDACTED_BY_LITELM_STRING
    if litellm_params is None:
        return None
    if isinstance(litellm_params, str):
        try:
            parsed: Final = json.loads(litellm_params)
        except (TypeError, ValueError):
            return REDACTED_BY_LITELM_STRING
        return json.dumps(_redact_sensitive_litellm_params(parsed, _depth + 1))
    if not isinstance(litellm_params, dict):
        return litellm_params
    out: Final[dict[str, object]] = {}
    for k, v in litellm_params.items():
        if k == MILVUS_ADMIN_CONFIGURED_CONNECTION:
            continue
        if _LITELLM_PARAMS_MASKER.is_sensitive_key(k):
            out[k] = REDACTED_BY_LITELM_STRING
        elif isinstance(v, dict):
            out[k] = _redact_sensitive_litellm_params(v, _depth + 1)
        else:
            out[k] = v
    return out


def _restore_redacted_litellm_params(supplied: object, existing: object, _depth: int = 0) -> object:
    if supplied == REDACTED_BY_LITELM_STRING:
        return existing
    if _depth >= _REDACT_LITELLM_PARAMS_MAX_DEPTH or not isinstance(supplied, dict) or not isinstance(existing, dict):
        return supplied
    supplied_params: Final = deserialize_litellm_params(supplied)
    existing_params: Final = deserialize_litellm_params(existing)
    return {
        key: _restore_redacted_litellm_params(value, existing_params.get(key), _depth + 1)
        for key, value in supplied_params.items()
        if value != REDACTED_BY_LITELM_STRING or key in existing_params
    }


def _reuses_redacted_secrets(supplied: object, _depth: int = 0) -> bool:
    if supplied == REDACTED_BY_LITELM_STRING:
        return True
    if _depth >= _REDACT_LITELLM_PARAMS_MAX_DEPTH or not isinstance(supplied, dict):
        return False
    return any(_reuses_redacted_secrets(value, _depth + 1) for value in deserialize_litellm_params(supplied).values())


def _litellm_params_validation_detail(error: ValidationError) -> str:
    return "; ".join(
        f"{'.'.join(str(location) for location in issue['loc'])}: {issue['msg']}" for issue in error.errors()
    )


def _validated_litellm_params(
    litellm_params: Mapping[str, object],
) -> Mapping[str, object]:
    from litellm.types.router import GenericLiteLLMParams

    try:
        return GenericLiteLLMParams.model_validate(litellm_params).model_dump(exclude_none=True)
    except ValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid litellm_params: {_litellm_params_validation_detail(e)}",
        ) from e


def _reject_config_vector_store_id(vector_store_id: str) -> None:
    registry: Final = litellm.vector_store_registry
    if registry is None or vector_store_id not in registry.config_vector_store_ids:
        return
    raise HTTPException(
        status_code=400,
        detail=f"Vector store ID {vector_store_id} is defined in proxy configuration and cannot be managed through the API",
    )


async def _fetch_and_authorize_vector_store(
    vector_store_id: str,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: "PrismaClient",
    reject_config_defined_id: bool = False,
) -> "LiteLLM_ManagedVectorStore":
    """
    Look up a vector store by id and confirm the caller can access it.
    Raises HTTPException(404) on miss and HTTPException(403) on access
    denial.
    """
    row: Final = await _vector_store_table(prisma_client).find_unique(where={"vector_store_id": vector_store_id})
    if row is None:
        if reject_config_defined_id:
            _reject_config_vector_store_id(vector_store_id)
        raise HTTPException(
            status_code=404,
            detail=f"Vector store with ID {vector_store_id} not found",
        )
    typed: Final = _row_to_vector_store(row)
    if not await _check_vector_store_access(typed, user_api_key_dict):
        raise HTTPException(
            status_code=403,
            detail="Access denied: You do not have permission to access this vector store",
        )
    return typed


########################################################
# Helper Functions
########################################################
async def _check_vector_store_access(
    vector_store: LiteLLM_ManagedVectorStore,
    user_api_key_dict: UserAPIKeyAuth,
) -> bool:
    """
    Check if the user has access to the vector store.

    Delegates to :func:`can_user_access_vector_store`, which honors:
    - PROXY_ADMIN bypass
    - legacy vector stores with no team_id
    - key-level and team-level ``object_permission.vector_stores`` allowlists
    - team_id match between key and store
    """
    return await can_user_access_vector_store(vector_store=vector_store, user_api_key_dict=user_api_key_dict)


def _vector_store_create_data(
    vector_store_id: str,
    custom_llm_provider: str,
    vector_store_name: str | None,
    vector_store_description: str | None,
    vector_store_metadata: Mapping[str, object] | None,
    litellm_params: Mapping[str, object] | None,
    litellm_credential_name: str | None,
    team_id: str | None,
    user_id: str | None,
) -> dict[str, object]:  # mutable-ok: Prisma create requires a mutable data dict
    serialized_params: Final = safe_dumps(_validated_litellm_params(litellm_params)) if litellm_params else "{}"
    return {  # mutable-ok: Prisma create requires a mutable data dict
        "vector_store_id": vector_store_id,
        "custom_llm_provider": custom_llm_provider,
        **{
            key: value
            for key, value in (
                ("vector_store_name", vector_store_name),
                ("vector_store_description", vector_store_description),
                (
                    "vector_store_metadata",
                    safe_dumps(vector_store_metadata) if vector_store_metadata is not None else None,
                ),
                ("litellm_credential_name", litellm_credential_name),
                ("team_id", team_id),
                ("user_id", user_id),
            )
            if value is not None
        },
        "litellm_params": serialized_params,
    }


async def create_vector_store_in_db(
    vector_store_id: str,
    custom_llm_provider: str,
    prisma_client: "PrismaClient | None",
    vector_store_name: str | None = None,
    vector_store_description: str | None = None,
    vector_store_metadata: dict | None = None,
    litellm_params: dict | None = None,
    litellm_credential_name: str | None = None,
    team_id: str | None = None,
    user_id: str | None = None,
) -> LiteLLM_ManagedVectorStore:
    """
    Helper function to create a vector store in the database.

    This function handles:
    - Checking if vector store already exists
    - Creating the vector store in the database
    - Adding it to the vector store registry

    Returns:
        LiteLLM_ManagedVectorStore: The created vector store object

    Raises:
        HTTPException: If vector store already exists or database error occurs
    """
    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    _reject_config_vector_store_id(vector_store_id)

    # Check if vector store already exists
    existing_vector_store: Final = await _vector_store_table(prisma_client).find_unique(
        where={"vector_store_id": vector_store_id}
    )
    if existing_vector_store is not None:
        raise HTTPException(
            status_code=400,
            detail=f"Vector store with ID {vector_store_id} already exists",
        )

    data_to_create: Final = _vector_store_create_data(
        vector_store_id=vector_store_id,
        custom_llm_provider=custom_llm_provider,
        vector_store_name=vector_store_name,
        vector_store_description=vector_store_description,
        vector_store_metadata=vector_store_metadata,
        litellm_params=litellm_params,
        litellm_credential_name=litellm_credential_name,
        team_id=team_id,
        user_id=user_id,
    )

    # Create in database
    _new_vector_store: Final = await _vector_store_table(prisma_client).create(data=data_to_create)

    new_vector_store: Final[LiteLLM_ManagedVectorStore] = _row_to_vector_store(_new_vector_store)

    # Add vector store to registry
    if litellm.vector_store_registry is not None:
        litellm.vector_store_registry.add_vector_store_to_registry(vector_store=new_vector_store)

    verbose_proxy_logger.info("Vector store %s created in database successfully", vector_store_id)

    return new_vector_store


########################################################
# Management Endpoints
########################################################
@router.post(
    "/vector_store/new",
    tags=["vector store management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def new_vector_store(
    vector_store: LiteLLM_ManagedVectorStore,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Create a new vector store.

    Parameters:
    - vector_store_id: str - Unique identifier for the vector store
    - custom_llm_provider: str - Provider of the vector store
    - vector_store_name: Optional[str] - Name of the vector store
    - vector_store_description: Optional[str] - Description of the vector store
    - vector_store_metadata: Optional[Dict] - Additional metadata for the vector store
    """
    await check_feature_access_for_user(user_api_key_dict, "vector_stores")

    from litellm.proxy.proxy_server import prisma_client

    try:
        vector_store_id: Final = vector_store.get("vector_store_id")
        custom_llm_provider: Final = vector_store.get("custom_llm_provider")

        if not vector_store_id or not custom_llm_provider:
            raise HTTPException(
                status_code=400,
                detail="vector_store_id and custom_llm_provider are required",
            )

        prepared_litellm_params: Final = prepare_vector_store_connection_for_persistence(
            custom_llm_provider=custom_llm_provider,
            litellm_params=vector_store.get("litellm_params"),
            user_api_key_dict=user_api_key_dict,
        )

        # Extract and validate metadata
        metadata: Final = vector_store.get("vector_store_metadata")
        validated_metadata: dict | None = None
        if metadata is not None and isinstance(metadata, dict):
            validated_metadata = metadata

        new_vector_store: Final = await create_vector_store_in_db(
            vector_store_id=vector_store_id,
            custom_llm_provider=custom_llm_provider,
            prisma_client=prisma_client,
            vector_store_name=vector_store.get("vector_store_name"),
            vector_store_description=vector_store.get("vector_store_description"),
            vector_store_metadata=validated_metadata,
            litellm_params=prepared_litellm_params,
            litellm_credential_name=vector_store.get("litellm_credential_name"),
            team_id=user_api_key_dict.team_id,
            user_id=user_api_key_dict.user_id,
        )

        return {
            "status": "success",
            "message": f"Vector store {vector_store.get('vector_store_id')} created successfully",
            "vector_store": _redact_vector_store(new_vector_store),
        }
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error creating vector store: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


def _vector_stores_by_id(
    vector_stores: Sequence[LiteLLM_ManagedVectorStore],
) -> Mapping[str, LiteLLM_ManagedVectorStore]:
    return {  # mutable-ok: registry synchronization requires ID-keyed replacement semantics
        vector_store_id: vector_store
        for vector_store in vector_stores
        if (vector_store_id := vector_store.get("vector_store_id"))
    }


def _synchronize_vector_store_registry(
    vector_stores_from_db: Sequence[LiteLLM_ManagedVectorStore],
) -> Mapping[str, LiteLLM_ManagedVectorStore]:
    database_stores: Final = _vector_stores_by_id(vector_stores_from_db)
    registry: Final = litellm.vector_store_registry
    if registry is None:
        return database_stores

    memory_stores: Final = _vector_stores_by_id(registry.vector_stores)
    config_ids: Final = registry.config_vector_store_ids
    stale_ids: Final = tuple(
        vector_store_id
        for vector_store_id in memory_stores
        if vector_store_id not in config_ids and vector_store_id not in database_stores
    )
    for vector_store_id in stale_ids:
        registry.delete_vector_store_from_registry(vector_store_id=vector_store_id)
        verbose_proxy_logger.debug("Removed deleted vector store %s from in-memory registry", vector_store_id)
    for vector_store_id, vector_store in database_stores.items():
        if vector_store_id not in config_ids:
            registry.update_vector_store_in_registry(vector_store_id=vector_store_id, updated_data=vector_store)

    return {  # mutable-ok: listing and access filtering require an ID-keyed dict
        **database_stores,
        **{
            vector_store_id: vector_store
            for vector_store_id, vector_store in memory_stores.items()
            if vector_store_id in config_ids
        },
    }


def _redact_vector_store(vector_store: LiteLLM_ManagedVectorStore) -> LiteLLM_ManagedVectorStore:
    return {**vector_store, "litellm_params": _redact_sensitive_litellm_params(vector_store.get("litellm_params"))}


@router.get(
    "/vector_store/list",
    tags=["vector store management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=LiteLLM_ManagedVectorStoreListResponse,
)
@router.get(
    "/v1/vector_store/list",
    tags=["vector store management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=LiteLLM_ManagedVectorStoreListResponse,
)
async def list_vector_stores(
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    page: int = 1,
    page_size: int = 100,
):
    """
    List all available vector stores with optional filtering and pagination.
    Combines both in-memory vector stores and those stored in the database.
    Config entries remain authoritative; database-backed entries sync from the database.

    Parameters:
    - page: int - Page number for pagination (default: 1)
    - page_size: int - Number of items per page (default: 100)
    """
    await check_feature_access_for_user(user_api_key_dict, "vector_stores")

    from litellm.proxy.proxy_server import prisma_client

    try:
        vector_stores_from_db: Final = await VectorStoreRegistry._get_vector_stores_from_db(prisma_client=prisma_client)
        vector_store_map: Final = _synchronize_vector_store_registry(vector_stores_from_db)
        accessible_vector_stores: Final = [  # mutable-ok: response model requires a list
            _redact_vector_store(vector_store)
            for vector_store in await filter_listable_vector_stores(vector_store_map.values(), user_api_key_dict)
        ]

        total_count: Final = len(accessible_vector_stores)
        total_pages: Final = (total_count + page_size - 1) // page_size

        response: Final = LiteLLM_ManagedVectorStoreListResponse(
            object="list",
            data=accessible_vector_stores,
            total_count=total_count,
            current_page=page,
            total_pages=total_pages,
        )

        return response
    except Exception as e:
        verbose_proxy_logger.exception("Error listing vector stores: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


async def _vector_store_delete_target(
    vector_store_id: str,
    prisma_client: "PrismaClient",
) -> tuple[LiteLLM_ManagedVectorStore, bool, bool]:
    row: Final = await _vector_store_table(prisma_client).find_unique(where={"vector_store_id": vector_store_id})
    registry: Final = litellm.vector_store_registry
    memory_store: Final = (
        registry.get_litellm_managed_vector_store_from_registry(vector_store_id=vector_store_id)
        if registry is not None
        else None
    )
    target: Final = _row_to_vector_store(row) if row is not None else memory_store
    if target is None:
        raise HTTPException(status_code=404, detail=f"Vector store with ID {vector_store_id} not found")
    return target, row is not None, memory_store is not None


@router.post(
    "/vector_store/delete",
    tags=["vector store management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def delete_vector_store(
    data: VectorStoreDeleteRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Delete a vector store from both database and in-memory registry.

    Parameters:
    - vector_store_id: str - ID of the vector store to delete
    """
    await check_feature_access_for_user(user_api_key_dict, "vector_stores")

    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        vector_store, database_exists, memory_exists = await _vector_store_delete_target(
            data.vector_store_id,
            prisma_client,
        )
        if not database_exists:
            _reject_config_vector_store_id(data.vector_store_id)
        if not await _check_vector_store_access(vector_store, user_api_key_dict):
            raise HTTPException(
                status_code=403,
                detail="Access denied: You do not have permission to delete this vector store",
            )
        if database_exists:
            await _vector_store_table(prisma_client).delete(where={"vector_store_id": data.vector_store_id})
        if memory_exists:
            registry: Final = litellm.vector_store_registry
            if registry is not None:
                registry.delete_vector_store_from_registry(vector_store_id=data.vector_store_id)

        return {
            "status": "success",
            "message": f"Vector store {data.vector_store_id} deleted successfully",
        }
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error deleting vector store: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/vector_store/info",
    tags=["vector store management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ResponseLiteLLM_ManagedVectorStore,
)
async def get_vector_store_info(
    data: VectorStoreInfoRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Return a single vector store's details"""
    await check_feature_access_for_user(user_api_key_dict, "vector_stores")

    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        if litellm.vector_store_registry is not None:
            vector_store: Final = litellm.vector_store_registry.get_litellm_managed_vector_store_from_registry(
                vector_store_id=data.vector_store_id
            )
            if vector_store is not None:
                # Check access control
                if not await _check_vector_store_access(vector_store, user_api_key_dict):
                    raise HTTPException(
                        status_code=403,
                        detail="Access denied: You do not have permission to access this vector store",
                    )

                vector_store_metadata: Final = vector_store.get("vector_store_metadata")
                # Parse metadata if it's a JSON string
                parsed_metadata: dict | None = None
                if isinstance(vector_store_metadata, str):
                    parsed_metadata = json.loads(vector_store_metadata)
                elif isinstance(vector_store_metadata, dict):
                    parsed_metadata = vector_store_metadata

                vector_store_pydantic_obj: Final = LiteLLM_ManagedVectorStoresTable(
                    vector_store_id=vector_store.get("vector_store_id") or "",
                    custom_llm_provider=vector_store.get("custom_llm_provider") or "",
                    vector_store_name=vector_store.get("vector_store_name") or None,
                    vector_store_description=vector_store.get("vector_store_description") or None,
                    vector_store_metadata=parsed_metadata,
                    created_at=vector_store.get("created_at") or None,
                    updated_at=vector_store.get("updated_at") or None,
                    litellm_credential_name=vector_store.get("litellm_credential_name"),
                    litellm_params=_redact_sensitive_litellm_params(vector_store.get("litellm_params")),
                    team_id=vector_store.get("team_id") or None,
                    user_id=vector_store.get("user_id") or None,
                )
                return {"vector_store": vector_store_pydantic_obj}

        vector_store_typed: Final = await _fetch_and_authorize_vector_store(
            vector_store_id=data.vector_store_id,
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
        )
        return {
            "vector_store": _redact_vector_store(vector_store_typed)
            if "litellm_params" in vector_store_typed
            else dict(vector_store_typed)
        }
    except HTTPException:
        # Preserve 403/404 from the access-control / not-found checks above;
        # the catch-all below would otherwise rewrite them as 500.
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error getting vector store info: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/vector_store/update",
    tags=["vector store management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_vector_store(
    data: VectorStoreUpdateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Update vector store details in both database and in-memory registry.
    The updated data is immediately synchronized to the in-memory registry.
    """
    await check_feature_access_for_user(user_api_key_dict, "vector_stores")

    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    try:
        update_data: Final = data.model_dump(exclude_unset=True)
        vector_store_id: Final[str] = update_data.pop("vector_store_id")

        # Per-store access control: anyone authenticated who passes the
        # premium-feature gate could otherwise update *any* vector store —
        # including stores belonging to other teams.
        existing_vector_store: Final = await _fetch_and_authorize_vector_store(
            vector_store_id=vector_store_id,
            user_api_key_dict=user_api_key_dict,
            prisma_client=prisma_client,
            reject_config_defined_id=True,
        )

        existing_litellm_params: Final = deserialize_litellm_params(existing_vector_store.get("litellm_params"))
        supplied_litellm_params: Final = _restore_redacted_litellm_params(
            update_data.get("litellm_params"), existing_litellm_params
        )
        effective_provider: Final = update_data.get("custom_llm_provider") or existing_vector_store.get(
            "custom_llm_provider"
        )
        effective_litellm_params: Final = prepare_vector_store_connection_for_persistence(
            custom_llm_provider=effective_provider,
            litellm_params=supplied_litellm_params,
            user_api_key_dict=user_api_key_dict,
            existing_custom_llm_provider=existing_vector_store.get("custom_llm_provider"),
            existing_litellm_params=existing_litellm_params,
            litellm_credential_name=update_data.get("litellm_credential_name"),
            existing_litellm_credential_name=existing_vector_store.get("litellm_credential_name"),
            litellm_credential_name_supplied="litellm_credential_name" in update_data,
            reuses_stored_credentials=_reuses_redacted_secrets(update_data.get("litellm_params")),
        )

        # Handle metadata serialization
        if update_data.get("vector_store_metadata") is not None:
            update_data["vector_store_metadata"] = safe_dumps(update_data["vector_store_metadata"])

        # Handle litellm_params if provided. As with the create path, the
        # embedding-config auto-resolve previously persisted cleartext
        # credentials into the row; each search embeds the query through
        # the router at request time, so this row only ever stores the
        # user-supplied ``litellm_embedding_model`` reference.
        if "litellm_params" in update_data or effective_litellm_params != existing_litellm_params:
            update_data["litellm_params"] = safe_dumps(_validated_litellm_params(effective_litellm_params))

        # Update in database
        updated: Final = await _vector_store_table(prisma_client).update(
            where={"vector_store_id": vector_store_id},
            data=update_data,
        )

        if updated is None:
            raise HTTPException(
                status_code=404,
                detail=f"Vector store with ID {vector_store_id} not found",
            )

        updated_vs: Final = _row_to_vector_store(updated)

        # Immediately update in-memory registry to keep it in sync
        if litellm.vector_store_registry is not None:
            litellm.vector_store_registry.update_vector_store_in_registry(
                vector_store_id=vector_store_id,
                updated_data=updated_vs,
            )
            verbose_proxy_logger.debug(
                "Updated vector store %s in both database and in-memory registry", vector_store_id
            )

        return {
            "status": "success",
            "message": f"Vector store {vector_store_id} updated successfully",
            "vector_store": _redact_vector_store(updated_vs),
        }
    except HTTPException:
        # Preserve 403/404 responses from the access-control / not-found
        # checks above; the catch-all below would otherwise rewrite them
        # as 500 with the original status code embedded in the detail.
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error updating vector store: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
