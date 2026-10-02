"""Spend attribution for batches created through a passthrough endpoint.

The creating key and its tags are read off the passthrough request's metadata and
persisted on the managed object row, because the batch cost lands hours later in a
background poll that has no request to read them from.
"""

import asyncio
from collections.abc import Mapping, Sequence
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.safe_json_dumps import strip_null_bytes


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
    otherwise the key's tags, which auth exposes as user_api_key_auth_metadata (a
    tagged key does not put its tags in the top-level metadata "tags" on the
    passthrough path)
    """
    tags: Final = _sanitized_str_tuple(request_metadata.get("tags"))
    if tags:
        return tags
    key_auth_metadata: Final = request_metadata.get("user_api_key_auth_metadata")
    if isinstance(key_auth_metadata, dict):
        return _sanitized_str_tuple(key_auth_metadata.get("tags"))
    return None


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
