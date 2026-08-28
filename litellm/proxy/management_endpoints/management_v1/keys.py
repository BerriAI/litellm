"""`PATCH /management/v1/keys/{key_id}`."""

from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Annotated, Final

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, model_validator

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    CommonProxyErrors,
    ProxyException,
    UpdateKeyRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.json_merge_patch import apply_json_merge_patch
from litellm.proxy.management_endpoints.key_management_endpoints import (
    _get_and_validate_existing_key,  # pyright: ignore[reportPrivateUsage]  # shared with POST /key/update on purpose, so the two routes cannot drift apart
    update_key_fn,
)
from litellm.proxy.management_endpoints.management_v1.common import (
    MANAGEMENT_V1_PREFIX,
    PROBLEM_TYPE_BASE,
    ManagementProblem,
    reject_unknown_query_params,
)
from litellm.proxy.utils import PrismaClient
from litellm.types.proxy.management_endpoints.management_v1 import (
    ItemResponse,
    ProblemDetail,
)

router: Final = APIRouter(prefix=MANAGEMENT_V1_PREFIX)

# A JSON column that the schema declares NOT NULL with a `{}` default, so it is always present on
# the wire. `Mapping` keeps it read-only to callers. The factory is unavoidable: pydantic deep-copies
# field defaults, and a `MappingProxyType` cannot be deep-copied, so an immutable default raises at
# validation time. Declared once here rather than repeated on each of the seven fields that use it.
_JsonObject = Annotated[
    Mapping[str, JsonValue],
    Field(default_factory=dict),  # mutable-ok: pydantic hands each instance its own copy, so no state is shared
]


class KeyResource(BaseModel):
    """A key as every `/management/v1/keys` operation returns it.

    One representation, shared by list, read, create and update, so a form seeded from any of them
    holds exactly the fields the server stores. A per-operation projection is what lets a form
    compute its dirty-field delta against a value the server never sent.

    The plaintext secret is structurally absent rather than filtered: it is not a declared field and
    extras are ignored, so it cannot appear here however the row was assembled. `key_id` is the
    hashed token, which is what identifies a key everywhere else, and `key_name` is the masked
    display form safe to show in a UI.
    """

    model_config = ConfigDict(extra="ignore")

    key_id: str
    key_name: str | None = None
    key_alias: str | None = None
    key_type: str | None = None
    user_id: str | None = None
    team_id: str | None = None
    agent_id: str | None = None
    project_id: str | None = None
    organization_id: str | None = None
    budget_id: str | None = None
    object_permission_id: str | None = None
    models: tuple[str, ...] = ()
    policies: tuple[str, ...] = ()
    access_group_ids: tuple[str, ...] = ()
    allowed_cache_controls: tuple[str, ...] = ()
    allowed_routes: tuple[str, ...] = ()
    aliases: _JsonObject
    config: _JsonObject
    permissions: _JsonObject
    metadata: _JsonObject
    model_spend: _JsonObject
    model_max_budget: _JsonObject
    budget_fallbacks: _JsonObject
    router_settings: Mapping[str, JsonValue] | None = None
    budget_limits: Mapping[str, JsonValue] | None = None
    spend: float = 0.0
    max_budget: float | None = None
    max_parallel_requests: int | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    budget_duration: str | None = None
    budget_reset_at: datetime | None = None
    blocked: bool | None = None
    expires: datetime | None = None
    auto_rotate: bool | None = None
    rotation_interval: str | None = None
    rotation_count: int | None = None
    last_rotation_at: datetime | None = None
    key_rotation_at: datetime | None = None
    last_active: datetime | None = None
    settings_updated_at: datetime | None = None
    created_at: datetime | None = None
    created_by: str | None = None
    updated_at: datetime | None = None
    updated_by: str | None = None


class KeyPatchRequest(UpdateKeyRequest):
    """Body of `PATCH /management/v1/keys/{key_id}`.

    Unknown fields are rejected rather than ignored: on a merge patch the set of fields present
    *is* the request, so a misspelled field has to fail loudly instead of silently no-op'ing.
    """

    model_config = ConfigDict(extra="forbid")

    key_id: str | None = None

    @model_validator(mode="after")
    def validate_key_identifier(self) -> "KeyPatchRequest":
        """The path supplies the identifier, and `key_id` is this surface's only spelling of it.

        `key` is the legacy route's spelling, inherited from `UpdateKeyRequest`. Accepting both
        would put two names for one field on a surface whose whole point is that there is one.
        """
        if self.key is not None:
            raise ValueError("`key` is not a field on this resource; the identifier is `key_id`, taken from the path")
        return self


_KEY_RESOURCE: Final = TypeAdapter(KeyResource)
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
# `object`, not `JsonValue`: a database row carries datetimes, which JsonValue does not admit.
_ROW: Final = TypeAdapter(dict[str, object])

_PROXY_ERROR_PROBLEMS: Final[Mapping[int, tuple[str, str]]] = MappingProxyType(
    {  # mutable-ok: an immutable mapping has no literal form; MappingProxyType freezes this one and it never escapes
        400: ("bad-request", "Bad request"),
        401: ("unauthorized", "Unauthorized"),
        403: ("forbidden", "Forbidden"),
        404: ("key-not-found", "Key not found"),
    }
)


def _problem(slug: str, title: str, status_code: int, detail: str) -> ProblemDetail:
    return ProblemDetail(type=f"{PROBLEM_TYPE_BASE}{slug}", title=title, status=status_code, detail=detail)


def _problem_from_proxy_exception(exc: ProxyException) -> ProblemDetail:
    """Translate the legacy write path's OpenAI-shaped error into a problem document.

    The write core is shared with `POST /key/update`, which must keep raising `ProxyException`, so
    the translation happens here rather than by changing what that core raises.
    """
    code: Final = str(exc.code)
    status_code: Final = int(code) if code.isdigit() else 400
    slug, title = _PROXY_ERROR_PROBLEMS.get(status_code, ("key-update-failed", "Key update failed"))
    return _problem(slug=slug, title=title, status_code=status_code, detail=exc.message)


def to_key_resource(row: Mapping[str, object]) -> KeyResource:
    """`key_id` comes from the row's own hashed token, never from the path.

    A caller may address a key by its plaintext secret, and echoing the path value back would put
    that secret in the response body.
    """
    return _KEY_RESOURCE.validate_python({**row, "key_id": row.get("token")})


async def _merge_key_metadata(
    key_id: str,
    prisma_client: PrismaClient | None,
    metadata_patch: JsonValue,
) -> JsonValue:
    """Deep-merge a metadata patch onto the key's stored metadata, per RFC 7396."""
    existing_key_row: Final = await _get_and_validate_existing_key(token=key_id, prisma_client=prisma_client)
    existing_metadata: Final = _JSON_OBJECT.validate_python(
        existing_key_row.metadata or {}  # pyright: ignore[reportUnknownMemberType]  # unannotated on the row model; the validate_python call around it is what types it
    )
    return apply_json_merge_patch(existing_metadata, metadata_patch)


@router.patch(
    "/keys/{key_id}",
    tags=["key management"],
    dependencies=[Depends(user_api_key_auth), Depends(reject_unknown_query_params)],
    response_model=ItemResponse[KeyResource],
)
async def patch_key(
    key_id: str,
    data: KeyPatchRequest,
    request: Request,
    user_api_key_dict: Annotated[UserAPIKeyAuth, Depends(user_api_key_auth)],
    litellm_changed_by: Annotated[
        str | None,
        Header(
            description="The litellm-changed-by header enables tracking of actions performed by authorized users on behalf of other users, providing an audit trail for accountability",
        ),
    ] = None,
) -> ItemResponse[KeyResource]:
    """
    Partially update a key, using RFC 7396 JSON Merge Patch semantics.

    `key_id` is taken from the path; a `key_id` in the body is accepted only when it matches.
    Omitting a field preserves it, `null` clears it, and any other value overwrites it. `metadata`
    merges rather than replacing: an omitted entry is preserved, `entry: null` deletes it, and a
    nested object recurses. Arrays replace wholesale, which RFC 7396 is explicit about. An unknown
    field is a 422 rather than a silent no-op.

    Answers with the full key under `data`, the same representation every other keys operation
    serves. The key's plaintext secret is never in that representation.

    ```
    curl --location --request PATCH 'http://0.0.0.0:4000/management/v1/keys/<key_id>' \
    --header 'Authorization: Bearer sk-1234' \
    --header 'Content-Type: application/json' \
    --data-raw '{
        "metadata": {"cost_center": "1234", "deprecated_entry": null}
    }'
    ```
    """
    try:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise ManagementProblem(
                _problem(
                    slug="database-not-connected",
                    title="Database not connected",
                    status_code=503,
                    detail=CommonProxyErrors.db_not_connected_error.value,
                )
            )

        if data.key_id is not None and data.key_id != key_id:
            raise ManagementProblem(
                _problem(
                    slug="identifier-mismatch",
                    title="Identifier mismatch",
                    status_code=400,
                    detail="`key_id` in the body does not match the `key_id` in the path.",
                )
            )

        patch_fields: Final = _JSON_OBJECT.validate_python(
            data.model_dump(exclude_unset=True, exclude={"key_id", "key"}, mode="json")
        )
        merged_fields: Final = (
            {**patch_fields, "metadata": await _merge_key_metadata(key_id, prisma_client, patch_fields["metadata"])}
            if "metadata" in patch_fields
            else patch_fields
        )

        updated: Final = _ROW.validate_python(
            await update_key_fn(
                request=request,
                data=UpdateKeyRequest.model_validate({"key": key_id, **merged_fields}),
                user_api_key_dict=user_api_key_dict,
                litellm_changed_by=litellm_changed_by,
            )
        )
        return ItemResponse(data=to_key_resource(updated))

    except ManagementProblem:
        raise
    except ProxyException as e:
        raise ManagementProblem(_problem_from_proxy_exception(e))
    except Exception as e:  # noqa: BLE001  # a driver error answers as a problem document, not the OpenAI error shape
        verbose_proxy_logger.exception(
            "litellm.proxy.management_endpoints.management_v1.keys.patch_key(): Exception occured - %s", e
        )
        raise ManagementProblem(
            _problem(
                slug="internal-server-error",
                title="Internal server error",
                status_code=500,
                detail="Failed to update key.",
            )
        )
