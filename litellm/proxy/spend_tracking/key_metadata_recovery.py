from collections.abc import Awaitable, Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from types import MappingProxyType
from typing import Final, TypeVar

from pydantic import BaseModel, TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import is_valid_sha256_hash
from litellm.proxy.utils import PrismaClient
from litellm.repositories.user_repository import UserRepository

_T = TypeVar("_T")

_ACTIVE_TOKEN_DIGEST_SQL: Final = """
SELECT encode(sha256(convert_to(token, 'UTF8')), 'hex') AS digest, key_alias, team_id, user_id
FROM "LiteLLM_VerificationToken"
WHERE encode(sha256(convert_to(token, 'UTF8')), 'hex') = ANY($1::text[])
"""

_DELETED_TOKEN_DIGEST_SQL: Final = """
SELECT DISTINCT ON (token)
    encode(sha256(convert_to(token, 'UTF8')), 'hex') AS digest, key_alias, team_id, user_id
FROM "LiteLLM_DeletedVerificationToken"
WHERE encode(sha256(convert_to(token, 'UTF8')), 'hex') = ANY($1::text[])
ORDER BY token, deleted_at DESC
"""


class KeyMetadataDict(TypedDict, total=False):
    key_alias: ReadOnly[str | None]
    team_id: ReadOnly[str | None]
    user_id: ReadOnly[str | None]
    user_email: ReadOnly[str | None]


class _TokenDigestRow(BaseModel):
    digest: str
    key_alias: str | None = None
    team_id: str | None = None
    user_id: str | None = None


_TOKEN_DIGEST_ROWS: Final = TypeAdapter(tuple[_TokenDigestRow, ...])
_EMPTY_KEY_METADATA: Final[Mapping[str, KeyMetadataDict]] = MappingProxyType({})
_EMPTY_EMAILS: Final[Mapping[str, str]] = MappingProxyType({})


async def _db_or_empty(
    load: Callable[[], Awaitable[_T]],
    warning: str,
    count: int,
) -> _T | None:
    from prisma.errors import PrismaError

    try:
        return await load()
    except PrismaError as e:
        verbose_proxy_logger.warning(warning, count, e)
        return None


async def _reverse_hash_key_metadata(
    prisma_client: PrismaClient,
    sql: str,
    wanted: AbstractSet[str],
    *,
    warning: str,
) -> Mapping[str, KeyMetadataDict]:
    rows: Final = await _db_or_empty(
        lambda: prisma_client.db.query_raw(sql, sorted(wanted)),
        warning,
        len(wanted),
    )
    if rows is None:
        return _EMPTY_KEY_METADATA
    return MappingProxyType(
        {
            row.digest: KeyMetadataDict(key_alias=row.key_alias, team_id=row.team_id, user_id=row.user_id)
            for row in _TOKEN_DIGEST_ROWS.validate_python(rows)
            if row.digest in wanted
        }
    )


async def _emails_for_user_ids(
    prisma_client: PrismaClient,
    user_ids: AbstractSet[str],
) -> Mapping[str, str]:
    if not user_ids:
        return _EMPTY_EMAILS
    users: Final = await _db_or_empty(
        lambda: UserRepository(prisma_client).table.find_many(
            where={"user_id": {"in": list(user_ids)}},  # mutable-ok: Prisma find_many where= is a dict
        ),
        "Failed user_email recovery for %d user ids: %s",
        len(user_ids),
    )
    if users is None:
        return _EMPTY_EMAILS
    return MappingProxyType(
        {
            user.user_id: user.user_email
            for user in users
            if getattr(user, "user_id", None) and getattr(user, "user_email", None)
        }
    )


def _meta_with_email(meta: KeyMetadataDict, emails: Mapping[str, str]) -> KeyMetadataDict:
    if meta.get("user_email"):
        return meta
    user_id: Final = meta.get("user_id")
    if not isinstance(user_id, str) or user_id not in emails:
        return meta
    updated: Final[KeyMetadataDict] = {**meta, "user_email": emails[user_id]}
    return updated


async def attach_user_emails(
    prisma_client: PrismaClient,
    recovered: Mapping[str, KeyMetadataDict],
) -> Mapping[str, KeyMetadataDict]:
    needing_email: Final = frozenset(
        user_id
        for meta in recovered.values()
        for user_id in (meta.get("user_id"),)
        if isinstance(user_id, str) and user_id and not meta.get("user_email")
    )
    emails: Final = await _emails_for_user_ids(prisma_client, needing_email)
    if not emails:
        return recovered
    return MappingProxyType({api_key: _meta_with_email(meta, emails) for api_key, meta in recovered.items()})


async def recover_double_hashed_key_metadata(
    prisma_client: PrismaClient,
    missing_keys: AbstractSet[str],
) -> Mapping[str, KeyMetadataDict]:
    """
    Recover key_alias/team_id/user_id for DailyUserSpend.api_key values that
    were double-hashed by the v1.99 spend-log provenance gate.

    Those rows store hash(VerificationToken.token) instead of the token, so the
    exact join misses. Postgres hashes the token column itself, one pass over
    active keys and one over deleted keys, so no key row crosses the wire.
    """
    sha_missing: Final = frozenset(key for key in missing_keys if is_valid_sha256_hash(key))
    if not sha_missing:
        return _EMPTY_KEY_METADATA

    from_active: Final = await _reverse_hash_key_metadata(
        prisma_client,
        _ACTIVE_TOKEN_DIGEST_SQL,
        sha_missing,
        warning="Failed reverse-hash recovery against active keys for %d missing keys: %s",
    )
    still_missing: Final = sha_missing - frozenset(from_active)
    if not still_missing:
        return from_active
    from_deleted: Final = await _reverse_hash_key_metadata(
        prisma_client,
        _DELETED_TOKEN_DIGEST_SQL,
        still_missing,
        warning="Failed reverse-hash recovery against deleted keys for %d missing keys: %s",
    )
    return MappingProxyType({**from_active, **from_deleted})


def _row_with_recovered_fields(
    row: Mapping[str, object],
    recovered: Mapping[str, KeyMetadataDict],
    *,
    api_key_field: str,
    alias_field: str,
    team_id_field: str,
    user_email_field: str,
) -> Mapping[str, object]:
    api_key: Final = row.get(api_key_field)
    if not isinstance(api_key, str) or api_key not in recovered:
        return row
    meta: Final = recovered[api_key]
    return MappingProxyType(
        {
            **row,
            alias_field: meta.get("key_alias") or row.get(alias_field),
            team_id_field: meta.get("team_id") or row.get(team_id_field),
            user_email_field: row.get(user_email_field) or meta.get("user_email"),
        }
    )


async def fill_missing_api_key_aliases(
    prisma_client: PrismaClient,
    rows: Sequence[Mapping[str, object]],
    *,
    api_key_field: str = "api_key",
    alias_field: str = "api_key_alias",
    team_id_field: str = "team_id",
    user_email_field: str = "user_email",
) -> tuple[Mapping[str, object], ...]:
    """
    Fill null api_key_alias / team_id / user_email on export rows whose api_key
    was double-hashed.

    Used by CloudZero and Focus, which join DailyUserSpend.api_key to
    VerificationToken.token and otherwise export null aliases for those rows.
    """
    missing_keys: Final = frozenset(
        key
        for row in rows
        for key in (row.get(api_key_field),)
        if isinstance(key, str) and key and row.get(alias_field) in (None, "")
    )
    if not missing_keys:
        return tuple(rows)

    recovered: Final = await attach_user_emails(
        prisma_client,
        await recover_double_hashed_key_metadata(prisma_client, missing_keys),
    )
    if not recovered:
        return tuple(rows)

    return tuple(
        _row_with_recovered_fields(
            row,
            recovered,
            api_key_field=api_key_field,
            alias_field=alias_field,
            team_id_field=team_id_field,
            user_email_field=user_email_field,
        )
        for row in rows
    )
