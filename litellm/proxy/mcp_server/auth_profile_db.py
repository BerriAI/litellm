import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.models.mcp_server import LiteLLM_MCPServerTable
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper, get_salt_key
from litellm.proxy.mcp_server.auth_profiles import (
    AuthProfileRefusal,
    MalformedProfile,
    StoredProfileId,
    ValidatedProfiles,
    describe_refusal,
    first_duplicate_profile_id,
    profile_host_refusal,
    profile_missing_credential,
    stored_profile_ids,
    validate_auth_profiles,
)
from litellm.repositories.table_repositories import MCPAuthProfileRepository
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.mcp_server_manager import MCPAuthProfile, MCPServer, auth_profile_from_row

if TYPE_CHECKING:
    from prisma import models as prisma_models

    from litellm.repositories.prisma_protocols import TableActions

_BLOB: Final = TypeAdapter(Mapping[str, str])


@dataclass(frozen=True, slots=True)
class SubmittedProfiles:
    raw: Sequence[Mapping[str, object]] | None


_SECRET_BLOB_KEYS: Final = ("client_id", "client_secret", "auth_value")


def _encrypted(value: str | None) -> str | None:
    if not value:
        return None
    return encrypt_value_helper(value=value, new_encryption_key=get_salt_key())


def stored_blob(credentials: object) -> Mapping[str, str]:
    decoded: Final = json.loads(credentials) if isinstance(credentials, str) else credentials
    return _BLOB.validate_python(decoded or {})


def _inherits(profile: MCPAuthProfile, stored: Mapping[str, object] | None) -> bool:
    return (
        stored is not None
        and stored["auth_type"] == profile.auth_type.value
        and stored["oauth2_flow"] == profile.oauth2_flow
        and ("token_url" not in profile.model_fields_set or profile.token_url == stored["token_url"])
    )


def _credentials_blob(profile: MCPAuthProfile, stored: Mapping[str, object] | None) -> Mapping[str, str]:
    kept: Final[Mapping[str, str]] = (
        stored_blob(stored["credentials"]) if stored is not None and _inherits(profile, stored) else {}
    )
    submitted: Final = {
        "client_id": _encrypted(profile.client_id),
        "client_secret": _encrypted(profile.client_secret),
        "auth_value": _encrypted(profile.authentication_token),
        "token_endpoint_auth_method": profile.token_endpoint_auth_method,
    }
    merged: Final[Mapping[str, str]] = {
        **{key: kept[key] for key in _SECRET_BLOB_KEYS if key in kept},
        **{key: value for key, value in submitted.items() if value is not None},
    }
    return merged


def profile_row(
    profile: MCPAuthProfile, touched_by: str, stored: Mapping[str, object] | None = None
) -> Mapping[str, object]:
    return {
        "profile_id": profile.id,
        "profile_id_normalized": profile.id.lower(),
        "auth_type": profile.auth_type.value,
        "oauth2_flow": profile.oauth2_flow,
        "credentials": safe_dumps(_credentials_blob(profile, stored)),
        "token_url": profile.token_url,
        "scopes": list(profile.scopes or ()),
        "updated_by": touched_by,
        **(
            {"created_by": touched_by}
            if stored is None
            else {"created_by": stored["created_by"], "created_at": stored["created_at"]}
        ),
    }


def create_profiles_write(profiles: Sequence[MCPAuthProfile], touched_by: str) -> Mapping[str, object]:
    return {"auth_profiles": {"create": [profile_row(profile, touched_by) for profile in profiles]}}


def in_stored_spelling(
    profiles: Sequence[MCPAuthProfile], stored: Mapping[str, Mapping[str, object]]
) -> tuple[MCPAuthProfile, ...]:
    spelling: Final = {profile_id.lower(): profile_id for profile_id in stored}
    return tuple(
        profile.model_copy(update={"id": spelling.get(profile.id.lower(), profile.id)}) for profile in profiles
    )


_KEPT_WHEN_OMITTED: Final = ("token_url", "scopes", "token_endpoint_auth_method")


def _with_omitted_settings_kept(profile: MCPAuthProfile, stored: Mapping[str, object] | None) -> MCPAuthProfile:
    if stored is None or not _inherits(profile, stored):
        return profile
    previous: Final = auth_profile_from_row(stored)
    return profile.model_copy(
        update={
            field: getattr(previous, field) for field in _KEPT_WHEN_OMITTED if field not in profile.model_fields_set
        }
    )


def replacement_profile_rows(
    profiles: Sequence[MCPAuthProfile], touched_by: str, stored: Mapping[str, Mapping[str, object]]
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        profile_row(_with_omitted_settings_kept(profile, stored.get(profile.id)), touched_by, stored.get(profile.id))
        for profile in in_stored_spelling(profiles, stored)
    )


def _required_secret(profile: MCPAuthProfile) -> tuple[str, str | None]:
    if profile.auth_type == MCPAuth.oauth2:
        return "client_secret", profile.client_secret
    return "authentication_token", profile.authentication_token


_PROFILE_FIELD_OF_BLOB_KEY: Final = (
    ("client_id", "client_id"),
    ("client_secret", "client_secret"),
    ("auth_value", "authentication_token"),
)


def _secrets_kept(profile: MCPAuthProfile, stored: Mapping[str, object] | None) -> frozenset[str]:
    if stored is None or not _inherits(profile, stored):
        return frozenset()
    blob: Final = stored_blob(stored["credentials"])
    return frozenset(field for key, field in _PROFILE_FIELD_OF_BLOB_KEY if blob.get(key))


def missing_secret(
    server_ref: str, profiles: Sequence[MCPAuthProfile], stored: Mapping[str, Mapping[str, object]]
) -> MalformedProfile | None:
    spelled: Final = in_stored_spelling(profiles, stored)
    for position, profile in enumerate(spelled):
        field, value = _required_secret(profile)
        if (previous := stored.get(profile.id)) is not None and not _inherits(profile, previous) and not value:
            return MalformedProfile(
                server_ref,
                position,
                f"profile '{profile.id}' changes its auth settings, so {field} must be supplied",
            )
    return profile_missing_credential(
        server_ref,
        tuple(_with_omitted_settings_kept(profile, stored.get(profile.id)) for profile in spelled),
        {profile.id: _secrets_kept(profile, stored.get(profile.id)) for profile in spelled},
    )


async def profile_rows_of(
    table: "TableActions[prisma_models.LiteLLM_MCPAuthProfileTable]", server_id: str
) -> Mapping[str, Mapping[str, object]]:
    rows: Final = await table.find_many(where={"server_id": server_id})
    return {row.profile_id: row.model_dump() for row in rows}


async def stored_profile_rows(prisma_client: object, server_id: str) -> Mapping[str, Mapping[str, object]]:
    return await profile_rows_of(MCPAuthProfileRepository(prisma_client, use_writer=True).table, server_id)


async def checked_profiles(
    prisma_client: object,
    server_ref: str,
    raw: object,
    config_profile_ids: Iterable[StoredProfileId],
    *,
    use_writer: bool = False,
) -> tuple[MCPAuthProfile, ...] | AuthProfileRefusal:
    own: Final = validate_auth_profiles({server_ref: raw}, ())
    if not isinstance(own, ValidatedProfiles):
        return own
    submitted: Final = own.by_server[server_ref]
    stored: Final = (
        await MCPAuthProfileRepository(prisma_client, use_writer=use_writer).table.find_many(
            # bounded-ok: the profiles one server write submits
            where={"profile_id_normalized": {"in": [profile.id.lower() for profile in submitted]}}
        )
        if submitted
        else ()
    )
    checked: Final = validate_auth_profiles(
        {server_ref: raw},
        tuple(StoredProfileId(row.profile_id, row.server_id) for row in stored),
        config_claims=config_profile_ids,
    )
    if isinstance(checked, ValidatedProfiles):
        return checked.by_server[server_ref]
    return checked


async def written_profiles(prisma_client: object, server_id: str) -> tuple[MCPAuthProfile, ...]:
    table: Final = MCPAuthProfileRepository(prisma_client, use_writer=True).table
    rows: Final = await table.find_many(where={"server_id": server_id})
    return tuple(auth_profile_from_row(row.model_dump()) for row in rows)


async def with_stored_profiles_of(
    prisma_client: object, tables: Sequence[LiteLLM_MCPServerTable]
) -> tuple[LiteLLM_MCPServerTable, ...]:
    profiles: Final = await stored_profiles_by_server(prisma_client, [table.server_id for table in tables])
    return tuple(table.model_copy(update={"additional_auth": profiles.get(table.server_id)}) for table in tables)


async def stored_profiles_by_server(
    prisma_client: object, server_ids: Sequence[str], *, use_writer: bool = False
) -> Mapping[str, tuple[MCPAuthProfile, ...]]:
    try:
        rows: Final = (
            await MCPAuthProfileRepository(prisma_client, use_writer=use_writer).table.find_many(
                # bounded-ok: one id per MCP server row the registry loads
                where={"server_id": {"in": list(server_ids)}}
            )
            if server_ids
            else ()
        )
    except Exception as error:  # noqa: BLE001  # profile storage is additive, a read failure must not fail the server read
        verbose_proxy_logger.warning("could not read MCP auth profiles for %s servers: %s", len(server_ids), error)
        return {}
    return {
        server_id: tuple(auth_profile_from_row(row.model_dump()) for row in rows if row.server_id == server_id)
        for server_id in {row.server_id for row in rows}
    }


async def with_stored_profiles(
    prisma_client: object, table: LiteLLM_MCPServerTable, *, use_writer: bool = False
) -> LiteLLM_MCPServerTable:
    profiles: Final = await stored_profiles_by_server(prisma_client, (table.server_id,), use_writer=use_writer)
    return table.model_copy(update={"additional_auth": profiles.get(table.server_id)})


async def startup_profile_refusal(config_servers: Iterable[MCPServer], prisma_client: object) -> str | None:
    from litellm.proxy.mcp_server.db import LOADABLE_MCP_SERVER_FILTER

    try:
        rows: Final = await MCPAuthProfileRepository(prisma_client, use_writer=True).table.find_many(
            where={"server": {"is": LOADABLE_MCP_SERVER_FILTER}}, include={"server": True}
        )
    except Exception as error:  # noqa: BLE001  # a transient read failure must not stop startup; writes still check
        verbose_proxy_logger.warning(
            "Skipped the MCP auth profile startup check, the profile table could not be read: %s", error
        )
        return None
    host_refusal: Final = next(
        (
            refusal
            for row in rows
            if row.server is not None
            and (
                refusal := profile_host_refusal(
                    row.server.server_name or row.server_id,
                    transport=row.server.transport,
                    is_byok=row.server.is_byok,
                    spec_path=row.server.spec_path,
                )
            )
            is not None
        ),
        None,
    )
    duplicate: Final = host_refusal or first_duplicate_profile_id(
        (
            *stored_profile_ids(config_servers),
            *(
                StoredProfileId(row.profile_id, (row.server.server_name if row.server else None) or row.server_id)
                for row in rows
            ),
        )
    )
    return None if duplicate is None else f"Invalid MCP auth profiles: {describe_refusal(duplicate)}"
