from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.models.mcp_server import LiteLLM_MCPServerTable
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper, get_salt_key
from litellm.proxy.mcp_server.auth_profiles import (
    AuthProfileRefusal,
    StoredProfileId,
    ValidatedProfiles,
    validate_auth_profiles,
)
from litellm.repositories.table_repositories import MCPAuthProfileRepository
from litellm.types.mcp_server.mcp_server_manager import MCPAuthProfile, auth_profile_from_row


@dataclass(frozen=True, slots=True)
class SubmittedProfiles:
    raw: Sequence[Mapping[str, object]] | None


def _encrypted(value: str | None) -> str | None:
    if not value:
        return None
    return encrypt_value_helper(value=value, new_encryption_key=get_salt_key())


def _credentials_blob(profile: MCPAuthProfile) -> Mapping[str, str]:
    submitted: Final = {
        "client_id": _encrypted(profile.client_id),
        "client_secret": _encrypted(profile.client_secret),
        "auth_value": _encrypted(profile.authentication_token),
        "token_endpoint_auth_method": profile.token_endpoint_auth_method,
    }
    return {key: value for key, value in submitted.items() if value is not None}


def profile_row(profile: MCPAuthProfile, touched_by: str) -> Mapping[str, object]:
    return {
        "profile_id": profile.id,
        "profile_id_normalized": profile.id.lower(),
        "auth_type": profile.auth_type.value,
        "oauth2_flow": profile.oauth2_flow,
        "credentials": safe_dumps(_credentials_blob(profile)),
        "token_url": profile.token_url,
        "scopes": list(profile.scopes or ()),
        "created_by": touched_by,
        "updated_by": touched_by,
    }


def create_profiles_write(profiles: Sequence[MCPAuthProfile], touched_by: str) -> Mapping[str, object]:
    return {"auth_profiles": {"create": [profile_row(profile, touched_by) for profile in profiles]}}


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
        rows: Final = await MCPAuthProfileRepository(prisma_client, use_writer=use_writer).table.find_many(
            # bounded-ok: one id per MCP server row the registry loads
            where={"server_id": {"in": list(server_ids)}}
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
