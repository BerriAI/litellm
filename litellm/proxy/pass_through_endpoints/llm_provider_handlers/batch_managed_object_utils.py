"""
Shared helper for persisting a passthrough-created batch as a managed object.

Vertex AI and Anthropic passthrough both create a provider batch job and then
need to register it for asynchronous cost tracking by CheckBatchCost. The
creating key's identity (hashed key, team, tags) is captured here so the
eventual batch-cost spend log can be attributed back to it, mirroring the
attribution a non-batch request receives.
"""

import asyncio
from typing import TYPE_CHECKING, Protocol, cast

from litellm._logging import verbose_proxy_logger
from litellm.types.utils import LiteLLMBatch

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth


class _StoreUnifiedObjectHook(Protocol):
    async def store_unified_object_id(
        self,
        unified_object_id: str,
        file_object: LiteLLMBatch,
        litellm_parent_otel_span: None,
        model_object_id: str,
        file_purpose: str,
        user_api_key_dict: "UserAPIKeyAuth",
        request_tags: list[str] | None,
    ) -> None: ...


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_str_list(value: object) -> list[str] | None:
    if isinstance(value, list):
        items = cast(list[object], value)  # cast-ok: isinstance-narrowed; element type unknown
        return [str(tag) for tag in items]
    return None


def _request_tags(request_metadata: dict[str, object]) -> list[str] | None:
    """Tags for the batch-cost spend row: the request's own tags when it sent any,
    otherwise the key's tags, which auth exposes in-memory as
    user_api_key_auth_metadata (a tagged key does not put its tags in the
    top-level metadata "tags" on the passthrough path)
    """
    tags = _optional_str_list(request_metadata.get("tags"))
    if tags:
        return tags
    key_auth_metadata = request_metadata.get("user_api_key_auth_metadata")
    if isinstance(key_auth_metadata, dict):
        typed_key_auth_metadata = cast(dict[str, object], key_auth_metadata)  # cast-ok: isinstance-narrowed dict
        return _optional_str_list(typed_key_auth_metadata.get("tags"))
    return None


def store_batch_managed_object(
    unified_object_id: str,
    batch_object: LiteLLMBatch,
    model_object_id: str,
    request_metadata: dict[str, object],
) -> None:
    """
    Register a provider batch job as a managed object so CheckBatchCost can
    later compute its cost and write an attributed spend log.

    request_metadata is the request's ``litellm_params["metadata"]``, which
    carries the authenticated key's ``user_api_key`` hash, user/team ids and
    request tags.
    """
    try:
        from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
        from litellm.proxy.proxy_server import proxy_logging_obj

        managed_files_hook = proxy_logging_obj.get_proxy_hook("managed_files")
        if managed_files_hook is None or not hasattr(managed_files_hook, "store_unified_object_id"):
            verbose_proxy_logger.warning(
                "Managed files hook not available, cannot store batch object for cost tracking"
            )
            return

        user_api_key_dict = UserAPIKeyAuth(
            user_id=_optional_str(request_metadata.get("user_api_key_user_id")) or "default-user",
            api_key=_optional_str(request_metadata.get("user_api_key")),
            team_id=_optional_str(request_metadata.get("user_api_key_team_id")),
            team_alias=_optional_str(request_metadata.get("user_api_key_team_alias")),
            key_alias=_optional_str(request_metadata.get("user_api_key_alias")),
            user_role=LitellmUserRoles.CUSTOMER,
        )

        hook = cast(_StoreUnifiedObjectHook, managed_files_hook)  # cast-ok: presence checked via hasattr above
        asyncio.create_task(
            hook.store_unified_object_id(
                unified_object_id=unified_object_id,
                file_object=batch_object,
                litellm_parent_otel_span=None,
                model_object_id=model_object_id,
                file_purpose="batch",
                user_api_key_dict=user_api_key_dict,
                request_tags=_request_tags(request_metadata),
            )
        )

        verbose_proxy_logger.info(
            f"Stored batch managed object with unified_object_id={unified_object_id}, batch_id={model_object_id}"
        )
    except Exception as e:
        verbose_proxy_logger.error(f"Error storing batch managed object: {e}")
