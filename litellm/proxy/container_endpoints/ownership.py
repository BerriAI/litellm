import os
from typing import Any, Dict, List, Optional, Set, Tuple

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.resource_ownership import (
    get_primary_resource_owner_scope,
    get_resource_owner_scopes,
    is_proxy_admin,
    user_can_access_resource_owner,
)
from litellm.responses.utils import ResponsesAPIRequestUtils

CONTAINER_OBJECT_PURPOSE = "container"
ALLOW_UNTRACKED_CONTAINER_ACCESS_ENV = "LITELLM_ALLOW_UNTRACKED_CONTAINER_ACCESS"
_IN_MEMORY_CONTAINER_OWNERS: Dict[str, str] = {}


def _allow_untracked_container_access() -> bool:
    return os.getenv(ALLOW_UNTRACKED_CONTAINER_ACCESS_ENV, "").lower() in {
        "1",
        "true",
        "yes",
    }


def _container_model_object_id(
    original_container_id: str,
    custom_llm_provider: str,
) -> str:
    return f"{CONTAINER_OBJECT_PURPOSE}:{custom_llm_provider}:{original_container_id}"


def decode_container_id_for_ownership(
    container_id: str,
    custom_llm_provider: str,
) -> Tuple[str, str]:
    decoded = ResponsesAPIRequestUtils._decode_container_id(container_id)
    original_container_id = decoded.get("response_id", container_id)
    decoded_provider = decoded.get("custom_llm_provider")
    if decoded_provider and custom_llm_provider == "openai":
        custom_llm_provider = decoded_provider
    return original_container_id, custom_llm_provider


def get_container_forwarding_params(
    container_id: str,
    original_container_id: str,
    custom_llm_provider: str,
) -> Dict[str, str]:
    params = {
        "container_id": original_container_id,
        "custom_llm_provider": custom_llm_provider,
    }
    decoded = ResponsesAPIRequestUtils._decode_container_id(container_id)
    model_id = decoded.get("model_id")
    if isinstance(model_id, str) and model_id:
        params["model_id"] = model_id
    return params


def _get_response_id(response: Any) -> Optional[str]:
    if response is None:
        return None
    if isinstance(response, dict):
        value = response.get("id")
    else:
        value = getattr(response, "id", None)
    return value if isinstance(value, str) else None


def _dump_response(response: Any) -> Dict[str, Any]:
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "dict"):
        return response.dict()
    return {"id": _get_response_id(response)}


async def _get_prisma_client():
    from litellm.proxy.proxy_server import prisma_client

    return prisma_client


async def record_container_owner(
    response: Any,
    user_api_key_dict: UserAPIKeyAuth,
    custom_llm_provider: str,
) -> Any:
    container_id = _get_response_id(response)
    owner = get_primary_resource_owner_scope(user_api_key_dict)
    if is_proxy_admin(user_api_key_dict) and (container_id is None or owner is None):
        return response
    if container_id is None:
        verbose_proxy_logger.warning(
            "Skipping container ownership tracking because provider response has no id"
        )
        return response
    if owner is None:
        raise HTTPException(status_code=500, detail="Unable to track container")

    original_container_id, resolved_provider = decode_container_id_for_ownership(
        container_id,
        custom_llm_provider,
    )
    model_object_id = _container_model_object_id(
        original_container_id,
        resolved_provider,
    )
    file_object = _dump_response(response)
    file_object["custom_llm_provider"] = resolved_provider
    file_object["provider_container_id"] = original_container_id

    try:
        prisma_client = await _get_prisma_client()
        if prisma_client is None:
            existing_owner = _IN_MEMORY_CONTAINER_OWNERS.get(model_object_id)
            if existing_owner is not None and not user_can_access_resource_owner(
                existing_owner, user_api_key_dict
            ):
                raise HTTPException(status_code=403, detail="Forbidden")
            _IN_MEMORY_CONTAINER_OWNERS[model_object_id] = owner
            return response

        existing = await prisma_client.db.litellm_managedobjecttable.find_unique(
            where={"model_object_id": model_object_id}
        )
        if existing is not None:
            if getattr(existing, "file_purpose", None) != CONTAINER_OBJECT_PURPOSE:
                raise HTTPException(status_code=500, detail="Unable to track container")
            if not user_can_access_resource_owner(
                getattr(existing, "created_by", None), user_api_key_dict
            ):
                raise HTTPException(status_code=403, detail="Forbidden")
            await prisma_client.db.litellm_managedobjecttable.update(
                where={"model_object_id": model_object_id},
                data={
                    "unified_object_id": container_id,
                    "file_object": file_object,
                    "updated_by": owner,
                },
            )
        else:
            await prisma_client.db.litellm_managedobjecttable.create(
                data={
                    "unified_object_id": container_id,
                    "model_object_id": model_object_id,
                    "file_object": file_object,
                    "file_purpose": CONTAINER_OBJECT_PURPOSE,
                    "created_by": owner,
                    "updated_by": owner,
                }
            )
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.warning(
            "Failed to persist container ownership for container_id=%s; "
            "falling back to in-process tracking: %s",
            model_object_id,
            e,
        )
        _IN_MEMORY_CONTAINER_OWNERS[model_object_id] = owner

    return response


async def _get_container_owner(
    original_container_id: str,
    custom_llm_provider: str,
) -> Optional[str]:
    model_object_id = _container_model_object_id(
        original_container_id,
        custom_llm_provider,
    )
    try:
        prisma_client = await _get_prisma_client()
        if prisma_client is None:
            return _IN_MEMORY_CONTAINER_OWNERS.get(model_object_id)

        row = await prisma_client.db.litellm_managedobjecttable.find_first(
            where={
                "model_object_id": model_object_id,
                "file_purpose": CONTAINER_OBJECT_PURPOSE,
            }
        )
        return getattr(row, "created_by", None) if row is not None else None
    except Exception as e:
        verbose_proxy_logger.warning(
            "Failed to load container ownership for container_id=%s; "
            "falling back to in-process tracking: %s",
            model_object_id,
            e,
        )
        return _IN_MEMORY_CONTAINER_OWNERS.get(model_object_id)


async def assert_user_can_access_container(
    container_id: str,
    user_api_key_dict: UserAPIKeyAuth,
    custom_llm_provider: str,
) -> Tuple[str, str]:
    original_container_id, resolved_provider = decode_container_id_for_ownership(
        container_id,
        custom_llm_provider,
    )

    if is_proxy_admin(user_api_key_dict):
        return original_container_id, resolved_provider

    owner = await _get_container_owner(original_container_id, resolved_provider)
    if owner is None and _allow_untracked_container_access():
        verbose_proxy_logger.warning(
            "Allowing untracked container access because %s is enabled",
            ALLOW_UNTRACKED_CONTAINER_ACCESS_ENV,
        )
        return original_container_id, resolved_provider

    if not user_can_access_resource_owner(owner, user_api_key_dict):
        raise HTTPException(status_code=403, detail="Forbidden")

    return original_container_id, resolved_provider


def _get_container_list_data(response: Any) -> Optional[List[Any]]:
    if response is None:
        return None
    if isinstance(response, dict):
        data = response.get("data")
    else:
        data = getattr(response, "data", None)
    return data if isinstance(data, list) else None


def _set_container_list_data(
    response: Any, data: List[Any], removed_filtered_items: bool = False
) -> Any:
    if isinstance(response, dict):
        response["data"] = data
        if data:
            response["first_id"] = _get_response_id(data[0])
            response["last_id"] = _get_response_id(data[-1])
        else:
            response["first_id"] = None
            response["last_id"] = None
            response["has_more"] = False
        if removed_filtered_items:
            response["has_more"] = False
        return response

    response.data = data
    response.first_id = _get_response_id(data[0]) if data else None
    response.last_id = _get_response_id(data[-1]) if data else None
    if not data and hasattr(response, "has_more"):
        response.has_more = False
    if removed_filtered_items and hasattr(response, "has_more"):
        response.has_more = False
    return response


async def _get_allowed_container_ids(
    user_api_key_dict: UserAPIKeyAuth,
    custom_llm_provider: str,
) -> Set[str]:
    owner_scopes = get_resource_owner_scopes(user_api_key_dict)
    if not owner_scopes:
        return set()

    in_memory_allowed_ids = {
        model_object_id
        for model_object_id, owner in _IN_MEMORY_CONTAINER_OWNERS.items()
        if owner in owner_scopes
    }
    try:
        prisma_client = await _get_prisma_client()
        if prisma_client is None:
            return in_memory_allowed_ids

        rows = await prisma_client.db.litellm_managedobjecttable.find_many(
            where={
                "file_purpose": CONTAINER_OBJECT_PURPOSE,
                "created_by": {"in": owner_scopes},
            }
        )
        return {
            row.model_object_id
            for row in rows
            if getattr(row, "model_object_id", None) is not None
        }
    except Exception as e:
        verbose_proxy_logger.warning(
            "Failed to load allowed container ids; falling back to in-process "
            "tracking: %s",
            e,
        )
        return in_memory_allowed_ids


async def filter_container_list_response(
    response: Any,
    user_api_key_dict: UserAPIKeyAuth,
    custom_llm_provider: str,
) -> Any:
    if is_proxy_admin(user_api_key_dict):
        return response

    data = _get_container_list_data(response)
    if data is None:
        return response

    allowed_container_ids = await _get_allowed_container_ids(
        user_api_key_dict,
        custom_llm_provider,
    )
    filtered: List[Any] = []
    for item in data:
        container_id = _get_response_id(item)
        if container_id is None:
            continue
        original_container_id, resolved_provider = decode_container_id_for_ownership(
            container_id,
            custom_llm_provider,
        )
        if (
            _container_model_object_id(original_container_id, resolved_provider)
            in allowed_container_ids
        ):
            filtered.append(item)

    return _set_container_list_data(
        response,
        filtered,
        removed_filtered_items=len(filtered) != len(data),
    )
