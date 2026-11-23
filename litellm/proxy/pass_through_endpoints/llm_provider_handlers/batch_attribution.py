"""Spend attribution for batches created through a passthrough endpoint.

The creating key and its tags are read off the passthrough request's metadata and
persisted on the managed object row, because the batch cost lands hours later in a
background poll that has no request to read them from.
"""

import asyncio
import base64
from collections.abc import Coroutine, Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Final, Literal, Protocol, runtime_checkable

from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.safe_json_dumps import strip_null_bytes
from litellm.types.utils import LiteLLMBatch, SpecialEnums

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth


@runtime_checkable
class BatchRegistry(Protocol):
    """The managed-files hook's write path, the one method passthrough batch attribution needs from it."""

    def store_unified_object_id(
        self,
        unified_object_id: str,
        file_object: LiteLLMBatch,
        litellm_parent_otel_span: None,
        model_object_id: str,
        file_purpose: Literal["batch"],
        user_api_key_dict: "UserAPIKeyAuth",
        request_tags: Sequence[str] | None,
        persist_attribution: bool,
    ) -> Coroutine[object, object, None]: ...


def optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _sanitized_str_tuple(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        return None
    items: Final[Sequence[object]] = value
    return tuple(strip_null_bytes(tag) for tag in items if isinstance(tag, str))


def is_collection_route(url_route: str, collection_suffix: str) -> bool:
    """Whether the route addresses the batch collection itself rather than one batch.
    A POST to the collection is the create; every id-scoped route is a retrieve,
    results or cancel.
    """
    return url_route.split("?")[0].rstrip("/").endswith(collection_suffix)


def request_tags_from_metadata(request_metadata: Mapping[str, object]) -> tuple[str, ...] | None:
    """Tags for the batch-cost spend row: the request's own tags when it sent any,
    otherwise the key's tags, which auth exposes as user_api_key_auth_metadata
    """
    tags: Final = _sanitized_str_tuple(request_metadata.get("tags"))
    if tags:
        return tags
    key_auth_metadata: Final = request_metadata.get("user_api_key_auth_metadata")
    if isinstance(key_auth_metadata, dict):
        return _sanitized_str_tuple(key_auth_metadata.get("tags"))
    return None


_METADATA_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])


def request_tags_from_request_data(data: Mapping[str, object]) -> tuple[str, ...] | None:
    """Tags of a request whose metadata travels as litellm_metadata (batches, responses)"""
    metadata: Final = data.get("litellm_metadata")
    if metadata is None:
        return None
    return request_tags_from_metadata(_METADATA_ADAPTER.validate_python(metadata))


def log_batch_registration_result(
    finished: asyncio.Task[None],
    provider: str,
    unified_object_id: str,
    model_object_id: str,
    is_batch_create: bool,
) -> None:
    """Report the outcome of the fire-and-forget managed object write. A create that
    fails is not retried by a later poll, so its cost is never tracked at all.
    """
    error: Final = finished.exception() if not finished.cancelled() else None
    if finished.cancelled() or error is not None:
        consequence: Final = (
            "its cost will not be tracked" if is_batch_create else "its status and output file may be stale"
        )
        verbose_proxy_logger.error(
            "Failed to store %s batch managed object with unified_object_id=%s, batch_id=%s; %s: %s",
            provider,
            unified_object_id,
            model_object_id,
            consequence,
            error,
        )
        return
    verbose_proxy_logger.info(
        "Stored %s batch managed object with unified_object_id=%s, batch_id=%s",
        provider,
        unified_object_id,
        model_object_id,
    )


def unified_batch_object_id(model_id: str, batch_id: str) -> str:
    """The managed object id CheckBatchCost decodes back into the deployment and the provider batch id."""
    unified: Final = SpecialEnums.LITELLM_MANAGED_BATCH_COMPLETE_STR.value.format(model_id, batch_id)
    return base64.urlsafe_b64encode(unified.encode()).decode().rstrip("=")


def register_passthrough_batch(
    provider: str,
    unified_object_id: str,
    batch_object: LiteLLMBatch,
    model_object_id: str,
    request_metadata: Mapping[str, object],
) -> None:
    """Record a batch created through a passthrough create for the calling key, team and tags, so
    CheckBatchCost prices it on completion. Fire and forget, the passthrough response is already on its way."""
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
    from litellm.proxy.proxy_server import proxy_logging_obj

    managed_files_hook: Final = proxy_logging_obj.get_proxy_hook("managed_files")
    if not isinstance(managed_files_hook, BatchRegistry):
        verbose_proxy_logger.warning(
            "Managed files hook not available, cannot store %s batch %s for cost tracking", provider, model_object_id
        )
        return
    user_api_key_dict: Final = UserAPIKeyAuth(
        user_id=optional_str(request_metadata.get("user_api_key_user_id")),
        api_key=optional_str(request_metadata.get("user_api_key")),
        team_id=optional_str(request_metadata.get("user_api_key_team_id")),
        user_role=LitellmUserRoles.CUSTOMER,
        spend=0.0,
        models=[],
        metadata={},
        permissions={},
        model_max_budget={},
        model_spend={},
    )
    task: Final[asyncio.Task[None]] = asyncio.create_task(
        managed_files_hook.store_unified_object_id(
            unified_object_id=unified_object_id,
            file_object=batch_object,
            litellm_parent_otel_span=None,
            model_object_id=model_object_id,
            file_purpose="batch",
            user_api_key_dict=user_api_key_dict,
            request_tags=request_tags_from_metadata(request_metadata),
            persist_attribution=True,
        )
    )
    task.add_done_callback(
        partial(
            log_batch_registration_result,
            provider=provider,
            unified_object_id=unified_object_id,
            model_object_id=model_object_id,
            is_batch_create=True,
        )
    )
