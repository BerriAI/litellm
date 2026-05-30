"""
Rewrite passthrough-managed IDs in pass-through endpoint requests and responses.

OUTPUT (response) path
----------------------
``rewrite_response_ids()`` is called after the upstream response is received.
It looks up the (provider, method, path) combination in ``BUILTIN_OUTPUT_ID_FIELD_MAP``,
mints a managed ID for each listed field whose raw provider value is present,
stores / reuses a DB row (dedup), and swaps the value in the body before the
response is returned to the client.

INPUT (request) path
--------------------
``rewrite_path_ids()``, ``rewrite_query_ids()``, and ``rewrite_body_ids()``
are called just before the request is forwarded upstream.  Each one walks its
respective location (URL path, query params, JSON body) and calls
``_resolve_one()`` for every string that looks like a passthrough managed ID
(decode-first detection).  ``_resolve_one()`` enforces:

  1. Cross-route check: the provider embedded in the ID must match the current
     route's provider, else HTTPException(404).
  2. DB existence check: unknown / forged IDs raise HTTPException(404); the
     raw string is NEVER forwarded to upstream.
  3. Access check: ``can_access_resource()`` raises HTTPException(403) on
     mismatch.

When a value does not decode as a passthrough managed ID it is passed through
untouched (deliberate opt-out for raw OpenAI IDs).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, FrozenSet, List, Optional, Tuple
from urllib.parse import quote, unquote

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.llms.base_llm.managed_resources.isolation import (
    build_owner_filter,
    can_access_resource,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.llms.openai import OpenAIFileObject

from .managed_id_codec import ManagedIdPayload, decode, is_managed, new_managed_id

# ---------------------------------------------------------------------------
# Field map
# ---------------------------------------------------------------------------

_FieldSpec = Tuple[str, str]  # (field_name, expected_raw_id_prefix)
_MapKey = Tuple[str, str, str]  # (provider, HTTP_METHOD, canonical_path)

# ``canonical_path`` uses ``/v1/...`` form without any ``/openai/`` prefix.
# Both ``/openai/...`` and ``/openai_passthrough/...`` are normalised by
# ``_canonical_path()`` before the lookup so only one set of entries is needed.
BUILTIN_OUTPUT_ID_FIELD_MAP: Dict[_MapKey, List[_FieldSpec]] = {
    # ------------------------------------------------------------------ files
    ("openai", "POST", "/v1/files"): [
        ("id", "file-"),
    ],
    ("openai", "GET", "/v1/files/{file_id}"): [
        ("id", "file-"),
    ],
    ("openai", "DELETE", "/v1/files/{file_id}"): [
        ("id", "file-"),
    ],
    # ----------------------------------------------------------------- batches
    ("openai", "POST", "/v1/batches"): [
        ("id", "batch_"),
        ("input_file_id", "file-"),
        ("output_file_id", "file-"),
        ("error_file_id", "file-"),
    ],
    ("openai", "GET", "/v1/batches/{batch_id}"): [
        ("id", "batch_"),
        ("input_file_id", "file-"),
        ("output_file_id", "file-"),
        ("error_file_id", "file-"),
    ],
    ("openai", "POST", "/v1/batches/{batch_id}/cancel"): [
        ("id", "batch_"),
        ("input_file_id", "file-"),
        ("output_file_id", "file-"),
        ("error_file_id", "file-"),
    ],
    # --------------------------------------------------------------- responses
    ("openai", "POST", "/v1/responses"): [
        ("id", "resp_"),
    ],
    ("openai", "GET", "/v1/responses/{response_id}"): [
        ("id", "resp_"),
    ],
    ("openai", "DELETE", "/v1/responses/{response_id}"): [
        ("id", "resp_"),
    ],
    # ================================================================ azure
    # Azure OpenAI exposes the same files/batches surface as OpenAI.
    # IDs are scoped to "azure" so they are never confused with "openai" ones.
    # ------------------------------------------------------------------ files
    ("azure", "POST", "/v1/files"): [
        ("id", "file-"),
    ],
    ("azure", "GET", "/v1/files/{file_id}"): [
        ("id", "file-"),
    ],
    ("azure", "DELETE", "/v1/files/{file_id}"): [
        ("id", "file-"),
    ],
    # ----------------------------------------------------------------- batches
    ("azure", "POST", "/v1/batches"): [
        ("id", "batch_"),
        ("input_file_id", "file-"),
        ("output_file_id", "file-"),
        ("error_file_id", "file-"),
    ],
    ("azure", "GET", "/v1/batches/{batch_id}"): [
        ("id", "batch_"),
        ("input_file_id", "file-"),
        ("output_file_id", "file-"),
        ("error_file_id", "file-"),
    ],
    ("azure", "POST", "/v1/batches/{batch_id}/cancel"): [
        ("id", "batch_"),
        ("input_file_id", "file-"),
        ("output_file_id", "file-"),
        ("error_file_id", "file-"),
    ],
    # --------------------------------------------------------------- responses
    ("azure", "POST", "/v1/responses"): [
        ("id", "resp_"),
    ],
    ("azure", "GET", "/v1/responses/{response_id}"): [
        ("id", "resp_"),
    ],
    ("azure", "DELETE", "/v1/responses/{response_id}"): [
        ("id", "resp_"),
    ],
}

# Prefixes that live in the *file* table rather than the object table.
_FILE_PREFIXES: FrozenSet[str] = frozenset({"file-"})

# Guards request-body rewriting against stack exhaustion from adversarially
# deep payloads. Real OpenAI files/batches bodies nest only a few levels.
_MAX_BODY_REWRITE_DEPTH = 64

# ---------------------------------------------------------------------------
# List routes — GET requests that return a paginated {object:"list", data:[…]}
# These are intercepted and served entirely from the DB rather than forwarded
# to the upstream provider, so each caller only sees IDs they own.
# ---------------------------------------------------------------------------

# Maps (provider, canonical_path) -> "files" | "batches"
_LIST_ROUTE_TABLE: Dict[Tuple[str, str], str] = {
    ("openai", "/v1/files"): "files",
    ("openai", "/v1/batches"): "batches",
    ("azure", "/v1/files"): "files",
    ("azure", "/v1/batches"): "batches",
}


# Sentinel model_id written to model_mappings for passthrough-created rows.
# Prevents the unified-endpoint deployment-resolution path from ever finding a
# real deployment, so a passthrough ID replayed on a unified endpoint fails
# cleanly (no silent raw-ID leak).
def _passthrough_sentinel_model_id(provider: str) -> str:
    return f"_passthrough_{provider}"


def _managed_id_matches_provider(unified_id: str, provider: str) -> bool:
    payload = decode(unified_id)
    return payload is not None and payload.provider == provider


# Strip /openai or /openai_passthrough prefix to produce canonical /v1/... path.
# Strips provider-specific passthrough prefixes before the /v1/... path:
#   /openai_passthrough/v1/files  -> /v1/files
#   /openai/v1/files              -> /v1/files
#   /azure/openai/files           -> /files  (_canonical_path then prepends /v1/)
#   /azure_ai/openai/files        -> /files
_PASSTHROUGH_PREFIX_RE = re.compile(
    r"^/(?:azure(?:_ai)?/)?openai(?:_passthrough)?(?=/|$)"
)


def _canonical_path(route: str) -> str:
    """
    Normalise a passthrough route to a bare /v1/... path for map lookup.

    Examples:
      /openai_passthrough/v1/files   -> /v1/files
      /openai/v1/files               -> /v1/files
      /azure/openai/files            -> /v1/files   (Azure omits /v1/)
      /azure/openai/batches/batch_x  -> /v1/batches/batch_x
    """
    stripped = _PASSTHROUGH_PREFIX_RE.sub("", route) or "/"
    # Azure API paths don't include /v1/ — add it so they match the map keys.
    if not stripped.startswith("/v1/") and stripped != "/":
        stripped = "/v1" + stripped
    return stripped


# ---------------------------------------------------------------------------
# Shared resolver — used by all INPUT path extractors
# ---------------------------------------------------------------------------


async def _resolve_one(
    managed_id: str,
    provider: str,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: Any,
    managed_files_hook: Any,
) -> str:
    """
    Resolve a single value that may be a passthrough managed ID.

    Returns the raw provider ID on success.
    Returns *managed_id* unchanged when it is NOT a managed ID so callers
    need not pre-filter.
    Raises ``HTTPException(403)`` on access denial.
    Raises ``HTTPException(404)`` on unknown / forged managed IDs — never
    forwarded upstream as a literal string.
    """
    payload: Optional[ManagedIdPayload] = decode(managed_id)
    if payload is None:
        return managed_id  # not a passthrough managed ID; pass through
    verbose_proxy_logger.debug(
        "managed_id_rewriter: resolving managed id provider=%s raw_prefix=%s",
        provider,
        (
            payload.raw_provider_id.split("_", 1)[0]
            if "_" in payload.raw_provider_id
            else payload.raw_provider_id.split("-", 1)[0]
        ),
    )

    # 1. Cross-route (cross-provider) check
    if payload.provider != provider:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Managed ID was minted for provider '{payload.provider}', "
                f"not '{provider}'."
            ),
        )

    row_created_by: Optional[str] = None
    row_team_id: Optional[str] = None
    found = False

    raw_id = payload.raw_provider_id

    # 2. DB lookup — pick table based on raw ID prefix
    if any(raw_id.startswith(p) for p in _FILE_PREFIXES):
        # File table — use hook's internal cache for speed when available
        if managed_files_hook is not None:
            try:
                file_row = await managed_files_hook.get_unified_file_id(
                    managed_id,
                    litellm_parent_otel_span=None,
                )
                if file_row is not None:
                    row_created_by = file_row.created_by
                    row_team_id = file_row.team_id
                    found = True
            except Exception:
                verbose_proxy_logger.debug(
                    "managed_id_rewriter._resolve_one: file hook lookup failed",
                    exc_info=True,
                )
        if not found and prisma_client is not None:
            try:
                db_row = await prisma_client.db.litellm_managedfiletable.find_first(
                    where={"unified_file_id": managed_id}
                )
                if db_row is not None:
                    row_created_by = db_row.created_by
                    row_team_id = db_row.team_id
                    found = True
            except Exception:
                verbose_proxy_logger.debug(
                    "managed_id_rewriter._resolve_one: file DB lookup failed",
                    exc_info=True,
                )
    else:
        # Object table (batches, responses)
        if prisma_client is not None:
            try:
                obj_row = await prisma_client.db.litellm_managedobjecttable.find_first(
                    where={"unified_object_id": managed_id}
                )
                if obj_row is not None:
                    row_created_by = obj_row.created_by
                    row_team_id = obj_row.team_id
                    found = True
            except Exception:
                verbose_proxy_logger.debug(
                    "managed_id_rewriter._resolve_one: object DB lookup failed",
                    exc_info=True,
                )

    # 3. Hard 404 for unknown / forged IDs — NEVER forward to upstream
    if not found:
        raise HTTPException(
            status_code=404,
            detail="Managed resource not found.",
        )

    # 4. Access check
    if not can_access_resource(user_api_key_dict, row_created_by, row_team_id):
        raise HTTPException(
            status_code=403,
            detail="Access denied to managed resource.",
        )

    return payload.raw_provider_id


# ---------------------------------------------------------------------------
# OUTPUT path — helpers for minting and storing managed IDs
# ---------------------------------------------------------------------------


def _build_managed_file_object(
    snapshot: Optional[Dict[str, Any]], managed_id: str
) -> Optional[OpenAIFileObject]:
    """Build an ``OpenAIFileObject`` (with the managed ID swapped in) from an
    upstream file response so the DB-served list returns the same metadata as a
    direct file GET.  Returns ``None`` when no usable snapshot is available, in
    which case the row is stored without metadata (previous behaviour)."""
    if not snapshot:
        return None
    try:
        return OpenAIFileObject(**{**snapshot, "id": managed_id})
    except Exception:
        verbose_proxy_logger.debug(
            "managed_id_rewriter: file object snapshot incomplete; "
            "storing file row without list metadata",
            exc_info=True,
        )
        return None


async def _mint_or_reuse_file(
    raw_id: str,
    provider: str,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: Any,
    managed_files_hook: Any,
    file_object_snapshot: Optional[Dict[str, Any]] = None,
    is_create_route: bool = True,
) -> str:
    """Return an existing managed file ID or mint + store a new one."""
    if prisma_client is None and managed_files_hook is None:
        return raw_id  # no persistence available; leave raw

    # Dedup + cross-tenant guard.  Look up existing passthrough rows for this
    # raw id WITHOUT scoping to the caller, so a raw file id that belongs to a
    # different tenant is denied rather than re-minted under the caller.  A raw
    # id only reaches this OUTPUT path by skipping the managed-id input gate (raw
    # provider ids are opt-out), so a row owned by someone else means the caller
    # is touching another tenant's upstream file.  flat_model_file_ids uses array
    # containment (no index, acceptable at the scale managed-file features run).
    #
    # The file table has no provider column, so the same raw id can map to one
    # row per provider (OpenAI and Azure both use the ``file-`` format).  Fetch
    # all matches and filter to this provider in the application layer, picking
    # the oldest match deterministically so two providers issuing the same raw id
    # reuse a stable row instead of minting duplicate rows on every call.
    if prisma_client is not None:
        try:
            candidates = await prisma_client.db.litellm_managedfiletable.find_many(
                where={"flat_model_file_ids": {"has": raw_id}},
                order={"created_at": "asc"},
            )
        except Exception:
            candidates = []
            verbose_proxy_logger.debug(
                "managed_id_rewriter: file dedup lookup failed", exc_info=True
            )
        provider_rows = [
            row
            for row in (candidates or [])
            if _managed_id_matches_provider(row.unified_file_id, provider)
        ]
        owned_row = next(
            (
                row
                for row in provider_rows
                if can_access_resource(user_api_key_dict, row.created_by, row.team_id)
            ),
            None,
        )
        if owned_row is not None:
            verbose_proxy_logger.debug(
                "managed_id_rewriter: reusing existing managed file id for raw prefix=%s",
                raw_id.split("-", 1)[0],
            )
            return owned_row.unified_file_id
        if provider_rows:
            if not is_create_route:
                # Retrieve / delete: the caller supplied another owner's raw file
                # id, so deny instead of minting a fresh managed id that would
                # grant them cross-tenant access.
                raise HTTPException(
                    status_code=404,
                    detail="Managed resource not found.",
                )
            # Create only: the caller's own upstream upload reused a raw id a
            # different owner already holds (two upstream accounts under one
            # provider name); the file is the caller's, so leave it unmanaged.
            verbose_proxy_logger.debug(
                "managed_id_rewriter: file dedup hit different owner on create; "
                "leaving raw id unmanaged for prefix=%s",
                raw_id.split("-", 1)[0],
            )
            return raw_id

    # No existing row — mint a new managed ID and store it.
    managed_id = new_managed_id(provider, raw_id)
    verbose_proxy_logger.debug(
        "managed_id_rewriter: minted new managed file id for raw prefix=%s",
        raw_id.split("-", 1)[0],
    )
    if managed_files_hook is not None:
        try:
            await managed_files_hook.store_unified_file_id(
                file_id=managed_id,
                file_object=_build_managed_file_object(
                    file_object_snapshot, managed_id
                ),
                litellm_parent_otel_span=None,
                model_mappings={_passthrough_sentinel_model_id(provider): raw_id},
                user_api_key_dict=user_api_key_dict,
            )
        except Exception:
            verbose_proxy_logger.warning(
                "managed_id_rewriter: could not persist file row; "
                "ID minted but not stored",
                exc_info=True,
            )
    return managed_id


async def _mint_or_reuse_object(
    raw_id: str,
    provider: str,
    file_purpose: str,
    body_snapshot: dict,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: Any,
    is_create_route: bool,
) -> str:
    """Return an existing managed object ID (batch/response) or mint + store one."""
    if prisma_client is None:
        return raw_id

    # Namespace raw_id with provider so two providers that happen to issue
    # the same raw batch/response ID get distinct rows.  The @unique constraint
    # on model_object_id would otherwise cause a UniqueConstraintViolation when
    # the second provider tries to insert, silently losing the persisted mapping
    # and causing every subsequent _resolve_one for that ID to return 404.
    # This mirrors the pattern in container_endpoints/ownership.py which uses
    # f"{purpose}:{provider}:{raw_id}" for the same reason.
    namespaced_model_object_id = f"passthrough:{provider}:{raw_id}"

    # Dedup: look up by the namespaced key — guaranteed unique per provider.
    try:
        existing = await prisma_client.db.litellm_managedobjecttable.find_first(
            where={"model_object_id": namespaced_model_object_id}
        )
    except Exception:
        verbose_proxy_logger.debug(
            "managed_id_rewriter: object dedup lookup failed", exc_info=True
        )
        existing = None

    if existing is not None:
        if not can_access_resource(
            user_api_key_dict, existing.created_by, existing.team_id
        ):
            if not is_create_route:
                # Retrieve / cancel / delete: the caller supplied a raw ID whose
                # managed row belongs to someone else.  A raw ID only reaches the
                # upstream by bypassing the managed-ID input gate, so deny here
                # instead of echoing another owner's object back to the caller.
                raise HTTPException(
                    status_code=404,
                    detail="Managed resource not found.",
                )
            # Create only: the caller's upstream create just succeeded under a
            # raw id a different owner already holds (two upstream accounts under
            # one provider name). The object is the caller's own, so leave the raw
            # id unmanaged rather than 404 a successful create; a new row can't be
            # minted because model_object_id is @unique.
            verbose_proxy_logger.debug(
                "managed_id_rewriter: object dedup hit different owner on create; "
                "leaving raw id unmanaged for prefix=%s",
                raw_id.split("_", 1)[0],
            )
            return raw_id
        # Refresh the stored snapshot so DB-served list responses reflect
        # the batch's latest state (e.g. output_file_id / error_file_id that
        # were null at creation but populated once the batch completed).
        try:
            await prisma_client.db.litellm_managedobjecttable.update(
                where={"unified_object_id": existing.unified_object_id},
                data={
                    "file_object": json.dumps(body_snapshot),
                    "updated_by": user_api_key_dict.user_id,
                },
            )
        except Exception:
            verbose_proxy_logger.debug(
                "managed_id_rewriter: object snapshot refresh failed",
                exc_info=True,
            )
        verbose_proxy_logger.debug(
            "managed_id_rewriter: reusing existing managed object id for raw prefix=%s",
            raw_id.split("_", 1)[0],
        )
        return existing.unified_object_id

    # No existing row — mint and upsert.
    managed_id = new_managed_id(provider, raw_id)
    verbose_proxy_logger.debug(
        "managed_id_rewriter: minted new managed object id for raw prefix=%s",
        raw_id.split("_", 1)[0],
    )
    try:
        await prisma_client.db.litellm_managedobjecttable.upsert(
            where={"unified_object_id": managed_id},
            data={
                "create": {
                    "unified_object_id": managed_id,
                    "file_object": json.dumps(body_snapshot),
                    "model_object_id": namespaced_model_object_id,
                    "file_purpose": file_purpose,
                    "created_by": user_api_key_dict.user_id,
                    "team_id": user_api_key_dict.team_id,
                    "updated_by": user_api_key_dict.user_id,
                },
                "update": {
                    "updated_by": user_api_key_dict.user_id,
                },
            },
        )
    except Exception:
        verbose_proxy_logger.warning(
            "managed_id_rewriter: could not persist object row; "
            "ID minted but not stored",
            exc_info=True,
        )
    return managed_id


async def rewrite_response_ids(
    provider: str,
    method: str,
    route: str,
    body: dict,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: Any,
    managed_files_hook: Any,
) -> dict:
    """
    Mint managed IDs for raw provider values listed in
    ``BUILTIN_OUTPUT_ID_FIELD_MAP`` and swap them into *body*.

    Returns the same *body* object (unchanged) when no map entry exists for
    this ``(provider, method, route)`` combination.
    Returns a shallow-copy of *body* with swapped values when any field is
    rewritten.
    """
    from litellm.proxy.auth.auth_utils import normalize_request_route

    # Strip passthrough prefix then normalize to get e.g. /v1/batches/{batch_id}
    canonical = normalize_request_route(_canonical_path(route))
    field_specs = BUILTIN_OUTPUT_ID_FIELD_MAP.get((provider, method, canonical))
    if field_specs is None:
        verbose_proxy_logger.debug(
            "managed_id_rewriter: no output rewrite map for provider=%s method=%s route=%s",
            provider,
            method,
            canonical,
        )
        return body

    # Collection endpoints (POST /v1/batches, /v1/responses) carry no resource
    # id in the path; everything else (retrieve / cancel / delete) does.  Only
    # creates may degrade to a raw id on a cross-owner collision.
    is_create_route = "{" not in canonical

    mutated = dict(body)  # shallow copy; only return if something changed
    changed = False

    def _record(field_name: str, raw_value: str, managed_id: str) -> None:
        nonlocal changed
        if managed_id != raw_value:
            mutated[field_name] = managed_id
            changed = True
            verbose_proxy_logger.debug(
                "managed_id_rewriter: output field rewritten field=%s route=%s method=%s",
                field_name,
                canonical,
                method,
            )

    # File fields are rewritten first so that nested references (e.g. a batch's
    # input_file_id) are already managed IDs when the object snapshot is
    # captured below — keeping the DB-served list in sync with a direct GET.
    for field_name, expected_prefix in field_specs:
        if expected_prefix not in _FILE_PREFIXES:
            continue
        raw_value = mutated.get(field_name)
        if not isinstance(raw_value, str) or not raw_value.startswith(expected_prefix):
            continue
        managed_id = await _mint_or_reuse_file(
            raw_value,
            provider,
            user_api_key_dict,
            prisma_client,
            managed_files_hook,
            # The file's own ``id`` carries the full upstream metadata; nested
            # references do not, so only the former is persisted as a snapshot.
            file_object_snapshot=body if field_name == "id" else None,
            is_create_route=is_create_route,
        )
        _record(field_name, raw_value, managed_id)

    for field_name, expected_prefix in field_specs:
        if expected_prefix in _FILE_PREFIXES:
            continue
        raw_value = mutated.get(field_name)
        if not isinstance(raw_value, str) or not raw_value.startswith(expected_prefix):
            continue
        purpose = "batch" if raw_value.startswith("batch_") else "response"
        managed_id = await _mint_or_reuse_object(
            raw_value,
            provider,
            purpose,
            mutated,
            user_api_key_dict,
            prisma_client,
            is_create_route,
        )
        _record(field_name, raw_value, managed_id)

    verbose_proxy_logger.debug(
        "managed_id_rewriter: output rewrite completed changed=%s provider=%s method=%s route=%s",
        changed,
        provider,
        method,
        canonical,
    )
    return mutated if changed else body


# ---------------------------------------------------------------------------
# List-route interception — serve listing entirely from DB
# ---------------------------------------------------------------------------


def is_passthrough_list_route(provider: str, method: str, route: str) -> bool:
    """Return True when this is a GET list route whose results should be served
    from the DB (user-scoped) rather than forwarded upstream."""
    if method != "GET":
        return False
    from litellm.proxy.auth.auth_utils import normalize_request_route

    canonical = normalize_request_route(_canonical_path(route))
    return (provider, canonical) in _LIST_ROUTE_TABLE


def _parse_file_object(file_object: Any) -> Any:
    """Prisma may return ``Json`` columns as either a parsed dict or the raw
    JSON string (depending on driver / row source). Mirror the handling used
    elsewhere (see ``openai_files_endpoints/common_utils.py``) so callers can
    treat the result uniformly.
    """
    if isinstance(file_object, str):
        try:
            return json.loads(file_object)
        except (TypeError, ValueError):
            return None
    return file_object


def _empty_list_response() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [],
        "first_id": None,
        "last_id": None,
        "has_more": False,
    }


def _parse_list_limit(query_params: Optional[Dict[str, Any]]) -> Tuple[int, int]:
    params = query_params or {}
    try:
        raw_limit = int(params.get("limit", 20))
    except (TypeError, ValueError):
        raw_limit = 20
    # Fetch one extra to cheaply detect has_more.
    return raw_limit, min(raw_limit, 100) + 1


async def _build_list_where_with_cursor(
    prisma_client: Any,
    resource_kind: str,
    provider: str,
    owner_filter: Dict[str, Any],
    query_params: Optional[Dict[str, Any]],
) -> Tuple[Dict[str, Any], str]:
    """Return a Prisma ``where`` clause and fetch order for a list query."""
    params = query_params or {}
    after_id: Optional[str] = params.get("after")
    before_id: Optional[str] = params.get("before")
    where: Dict[str, Any] = dict(owner_filter)
    fetch_order = "desc"

    cursor_id = after_id or before_id
    # A cursor minted for a different provider would resolve to that provider's
    # created_at boundary and silently skip/repeat this provider's rows, so
    # ignore it and serve the unscoped first page instead.
    if not cursor_id or not _managed_id_matches_provider(cursor_id, provider):
        return where, fetch_order

    cursor_table = (
        prisma_client.db.litellm_managedfiletable
        if resource_kind == "files"
        else prisma_client.db.litellm_managedobjecttable
    )
    cursor_field = (
        "unified_file_id" if resource_kind == "files" else "unified_object_id"
    )
    try:
        cursor_row = await cursor_table.find_first(
            where={**owner_filter, cursor_field: cursor_id}
        )
        if cursor_row is not None:
            if after_id:
                where["created_at"] = {"lt": cursor_row.created_at}
            else:
                where["created_at"] = {"gt": cursor_row.created_at}
                fetch_order = "asc"
    except Exception:
        pass
    return where, fetch_order


async def _fetch_list_rows(
    prisma_client: Any,
    resource_kind: str,
    where: Dict[str, Any],
    fetch_order: str,
    fetch_limit: int,
    skip: int = 0,
) -> Optional[List[Any]]:
    # created_at is not unique, so a second sort on the unique id column gives a
    # total order; the scan loop relies on it to advance by offset without
    # skipping or repeating rows that share a created_at timestamp.
    try:
        if resource_kind == "files":
            return await prisma_client.db.litellm_managedfiletable.find_many(
                where=where,
                order=[{"created_at": fetch_order}, {"unified_file_id": fetch_order}],
                take=fetch_limit,
                skip=skip,
            )
        return await prisma_client.db.litellm_managedobjecttable.find_many(
            where={**where, "file_purpose": "batch"},
            order=[{"created_at": fetch_order}, {"unified_object_id": fetch_order}],
            take=fetch_limit,
            skip=skip,
        )
    except Exception:
        verbose_proxy_logger.warning(
            "managed_id_rewriter: list DB query failed", exc_info=True
        )
        return None


async def _fetch_provider_scoped_list_rows(
    prisma_client: Any,
    resource_kind: str,
    provider: str,
    where: Dict[str, Any],
    fetch_order: str,
    raw_limit: int,
    fetch_limit: int,
) -> Tuple[List[Any], bool]:
    """Fetch list rows scoped to *provider* by decoding each managed ID.

    Always returns a (page, has_more) tuple — never ``None``.  A DB failure
    returns the rows matched so far (fail-closed) rather than ``None``, which
    would cause the caller to fall through to the upstream provider.
    """
    matched: List[Any] = []
    scan_where = dict(where)
    id_field = "unified_file_id" if resource_kind == "files" else "unified_object_id"
    # Object rows store model_object_id as "passthrough:{provider}:{raw}" (see
    # _mint_or_reuse_object), so the provider scope is pushed down to the indexed
    # DB column and the scan collapses to one query. File rows have no provider
    # column and fall back to the application-layer decode filter below.
    if resource_kind != "files":
        scan_where["model_object_id"] = {"startswith": f"passthrough:{provider}:"}
    max_scans = 20
    last_batch_full = False  # True when the last DB page was full-sized
    scanned = 0  # offset into the totally-ordered result set

    for _ in range(max_scans):
        batch = await _fetch_list_rows(
            prisma_client, resource_kind, scan_where, fetch_order, fetch_limit, scanned
        )
        if batch is None:
            # DB error — fail closed: return the rows matched so far.
            verbose_proxy_logger.warning(
                "managed_id_rewriter: list DB error mid-scan provider=%s kind=%s "
                "— returning partial results to fail closed",
                provider,
                resource_kind,
            )
            last_batch_full = False
            break
        if not batch:
            last_batch_full = False
            break

        scanned += len(batch)
        last_batch_full = len(batch) >= fetch_limit

        for row in batch:
            if _managed_id_matches_provider(getattr(row, id_field), provider):
                matched.append(row)

        if len(matched) >= fetch_limit:
            break

        if not last_batch_full:
            break

    effective_limit = min(raw_limit, 100)
    # has_more is True when:
    # 1. We accumulated more rows than the requested limit (normal path), OR
    # 2. The scan cap was hit while the last DB page was still full-sized —
    #    meaning unscanned rows almost certainly remain.  Without this,
    #    high-mixed-provider pools would return has_more=False on a truncated page.
    scan_cap_hit = len(matched) < fetch_limit and last_batch_full
    has_more = len(matched) > effective_limit or scan_cap_hit
    page = matched[:effective_limit]
    if fetch_order == "asc":
        page = list(reversed(page))
    return page, has_more


def _serialize_file_list_item(row: Any) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "id": row.unified_file_id,
        "object": "file",
        "created_at": int(row.created_at.timestamp()) if row.created_at else None,
    }
    file_object = _parse_file_object(row.file_object)
    if isinstance(file_object, dict):
        item.update(file_object)
    item["id"] = row.unified_file_id  # managed ID always wins over stored raw id
    return item


def _serialize_batch_list_item(row: Any) -> Dict[str, Any]:
    item: Dict[str, Any] = {}
    file_object = _parse_file_object(row.file_object)
    if isinstance(file_object, dict):
        item.update(file_object)
    item["id"] = row.unified_object_id  # managed ID always wins
    item["object"] = "batch"
    return item


def _list_boundary_ids(
    rows: List[Any], resource_kind: str
) -> Tuple[Optional[str], Optional[str]]:
    if not rows:
        return None, None
    id_attr = "unified_file_id" if resource_kind == "files" else "unified_object_id"
    return getattr(rows[0], id_attr), getattr(rows[-1], id_attr)


async def list_passthrough_ids_from_db(
    provider: str,
    route: str,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: Any,
    query_params: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Query the DB for managed IDs the caller owns and return an OpenAI-style
    paginated list response.

    Returns ``None`` when ``prisma_client`` is unavailable or the route is not
    a recognised list route (caller should fall through to upstream).

    Pagination params ``after``, ``before``, and ``limit`` are read from
    ``query_params`` to match the OpenAI Batches / Files list API.

    Ownership scoping:
    - Proxy admins / master key: see **all** rows.
    - Regular users: only rows matching their ``user_id`` / ``team_id``.
    """
    if prisma_client is None:
        return None

    from litellm.proxy.auth.auth_utils import normalize_request_route

    canonical = normalize_request_route(_canonical_path(route))
    resource_kind = _LIST_ROUTE_TABLE.get((provider, canonical))
    if resource_kind is None:
        return None

    owner_filter = build_owner_filter(user_api_key_dict)
    if owner_filter is None:
        verbose_proxy_logger.warning(
            "managed_id_rewriter: list denied — caller has no user_id or team_id"
        )
        return _empty_list_response()

    raw_limit, fetch_limit = _parse_list_limit(query_params)
    where, fetch_order = await _build_list_where_with_cursor(
        prisma_client, resource_kind, provider, owner_filter, query_params
    )
    page, has_more = await _fetch_provider_scoped_list_rows(
        prisma_client,
        resource_kind,
        provider,
        where,
        fetch_order,
        raw_limit,
        fetch_limit,
    )
    if resource_kind == "files":
        data = [_serialize_file_list_item(row) for row in page]
    else:
        data = [_serialize_batch_list_item(row) for row in page]

    first_id, last_id = _list_boundary_ids(page, resource_kind)
    verbose_proxy_logger.debug(
        "managed_id_rewriter: list served from DB provider=%s kind=%s count=%d admin=%s",
        provider,
        resource_kind,
        len(data),
        owner_filter == {},
    )
    return {
        "object": "list",
        "data": data,
        "first_id": first_id,
        "last_id": last_id,
        "has_more": has_more,
    }


# ---------------------------------------------------------------------------
# INPUT path extractors — all delegate to _resolve_one
# ---------------------------------------------------------------------------


async def rewrite_path_ids(
    path: str,
    provider: str,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: Any,
    managed_files_hook: Any,
) -> str:
    """
    Walk URL path segments and resolve any passthrough managed IDs to raw
    provider IDs.  Returns *path* unchanged when no managed IDs are found.
    """
    segments = path.split("/")
    new_segments: List[str] = []
    changed = False
    for seg in segments:
        decoded_seg = unquote(seg)
        if is_managed(decoded_seg):
            raw = await _resolve_one(
                decoded_seg,
                provider,
                user_api_key_dict,
                prisma_client,
                managed_files_hook,
            )
            new_segments.append(quote(raw, safe="-_.~"))
            changed = True
        else:
            new_segments.append(seg)
    if changed:
        verbose_proxy_logger.debug(
            "managed_id_rewriter: path ids rewritten provider=%s", provider
        )
    return "/".join(new_segments) if changed else path


async def rewrite_query_ids(
    params: Optional[Dict[str, Any]],
    provider: str,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: Any,
    managed_files_hook: Any,
) -> Optional[Dict[str, Any]]:
    """
    Walk query param values and resolve any passthrough managed IDs.
    Returns *params* unchanged (same object) when nothing is resolved.
    """
    if not params:
        return params
    mutated = dict(params)
    changed = False
    for key, val in list(mutated.items()):
        if isinstance(val, str) and is_managed(val):
            mutated[key] = await _resolve_one(
                val, provider, user_api_key_dict, prisma_client, managed_files_hook
            )
            changed = True
    if changed:
        verbose_proxy_logger.debug(
            "managed_id_rewriter: query ids rewritten provider=%s keys=%s",
            provider,
            [k for k, v in mutated.items() if isinstance(v, str)],
        )
    return mutated if changed else params


async def rewrite_body_ids(
    body: Optional[Dict[str, Any]],
    provider: str,
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: Any,
    managed_files_hook: Any,
) -> Optional[Dict[str, Any]]:
    """
    Recursively walk a request body dict/list and resolve any passthrough
    managed IDs.  Skips litellm internal keys (``litellm_*``).
    Returns *body* unchanged (same object) when nothing is resolved.
    """
    if not body:
        return body

    async def _walk(node: Any, depth: int) -> Any:
        if depth >= _MAX_BODY_REWRITE_DEPTH:
            return node
        if isinstance(node, dict):
            result: Dict[str, Any] = {}
            changed_inner = False
            for k, v in node.items():
                # Skip litellm internal injection keys (e.g. litellm_logging_obj)
                if isinstance(k, str) and k.startswith("litellm_"):
                    result[k] = v
                    continue
                new_v = await _walk(v, depth + 1)
                result[k] = new_v
                if new_v is not v:
                    changed_inner = True
            return result if changed_inner else node
        elif isinstance(node, list):
            new_list = [await _walk(item, depth + 1) for item in node]
            if any(n is not o for n, o in zip(new_list, node)):
                return new_list
            return node
        elif isinstance(node, str) and is_managed(node):
            return await _resolve_one(
                node, provider, user_api_key_dict, prisma_client, managed_files_hook
            )
        return node

    rewritten = await _walk(body, 0)
    if rewritten is not body:
        verbose_proxy_logger.debug(
            "managed_id_rewriter: body ids rewritten provider=%s", provider
        )
    return rewritten
