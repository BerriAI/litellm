from collections.abc import Awaitable, Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from types import MappingProxyType
from typing import Final, Protocol, TypeVar

from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import is_valid_sha256_hash
from litellm.proxy.utils import PrismaClient, hash_token
from litellm.repositories.table_repositories import DeletedVerificationTokenRepository
from litellm.repositories.user_repository import UserRepository
from litellm.repositories.verification_token_repository import (
    VerificationTokenRepository,
)

_T = TypeVar("_T")

_TOKEN_SCAN_PAGE: Final = 10_000

_SPEND_LOGS_KEY_METADATA_SQL: Final = """
SELECT DISTINCT ON (api_key)
    api_key,
    metadata->>'user_api_key_alias' AS key_alias,
    metadata->>'user_api_key_team_id' AS team_id,
    metadata->>'user_api_key_user_id' AS user_id,
    metadata->>'user_api_key_user_email' AS user_email
FROM "LiteLLM_SpendLogs"
WHERE api_key = ANY($1::text[])
  AND (
    NULLIF(metadata->>'user_api_key_alias', '') IS NOT NULL
    OR NULLIF(metadata->>'user_api_key_team_id', '') IS NOT NULL
    OR NULLIF(metadata->>'user_api_key_user_email', '') IS NOT NULL
  )
ORDER BY api_key, "startTime" DESC NULLS LAST
"""


class KeyMetadataDict(TypedDict, total=False):
    key_alias: ReadOnly[str | None]
    team_id: ReadOnly[str | None]
    user_id: ReadOnly[str | None]
    user_email: ReadOnly[str | None]


_EMPTY_KEY_METADATA: Final[Mapping[str, KeyMetadataDict]] = MappingProxyType({})
_EMPTY_EMAILS: Final[Mapping[str, str]] = MappingProxyType({})


class _TokenAliasRecord(Protocol):
    @property
    def token(self) -> str: ...

    @property
    def key_alias(self) -> str | None: ...

    @property
    def team_id(self) -> str | None: ...

    @property
    def user_id(self) -> str | None: ...


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


def _record_metadata(record: _TokenAliasRecord) -> KeyMetadataDict:
    meta: Final[KeyMetadataDict] = {
        "key_alias": record.key_alias,
        "team_id": record.team_id,
        "user_id": getattr(record, "user_id", None),
    }
    return meta


def _spend_log_row_metadata(row: Mapping[str, object]) -> KeyMetadataDict:
    meta: Final[KeyMetadataDict] = {
        "key_alias": row.get("key_alias") if isinstance(row.get("key_alias"), str) else None,
        "team_id": row.get("team_id") if isinstance(row.get("team_id"), str) else None,
        "user_id": row.get("user_id") if isinstance(row.get("user_id"), str) else None,
        "user_email": row.get("user_email") if isinstance(row.get("user_email"), str) else None,
    }
    return meta


def _token_digest_metadata(
    records: Sequence[_TokenAliasRecord],
    wanted: AbstractSet[str],
) -> Mapping[str, KeyMetadataDict]:
    return MappingProxyType(
        {
            digested: _record_metadata(record)
            for record in records
            for digested in (hash_token(record.token),)
            if digested in wanted
        }
    )


async def _paginate_token_digest_metadata(
    load_page: Callable[[int], Awaitable[Sequence[_TokenAliasRecord] | None]],
    wanted: AbstractSet[str],
    *,
    page_size: int,
    skip: int = 0,
    accumulated: Mapping[str, KeyMetadataDict] = _EMPTY_KEY_METADATA,
) -> Mapping[str, KeyMetadataDict]:
    if not wanted:
        return accumulated
    records: Final = await load_page(skip)
    if records is None:
        return accumulated
    page_hits: Final = _token_digest_metadata(records, wanted)
    combined: Final[Mapping[str, KeyMetadataDict]] = (
        MappingProxyType({**accumulated, **page_hits}) if page_hits else accumulated
    )
    still_wanted: Final = wanted - frozenset(page_hits)
    if not still_wanted or len(records) < page_size:
        return combined
    return await _paginate_token_digest_metadata(
        load_page,
        still_wanted,
        page_size=page_size,
        skip=skip + page_size,
        accumulated=combined,
    )


async def _reverse_hash_active_key_metadata(
    prisma_client: PrismaClient,
    wanted: AbstractSet[str],
    *,
    page_size: int,
) -> Mapping[str, KeyMetadataDict]:
    async def load_page(skip: int) -> Sequence[_TokenAliasRecord] | None:
        return await _db_or_empty(
            lambda: VerificationTokenRepository(prisma_client).table.find_many(
                take=page_size,
                skip=skip,
                order={"token": "asc"},  # mutable-ok: Prisma find_many order= is a dict
            ),
            "Failed reverse-hash recovery against active keys for %d missing keys: %s",
            len(wanted),
        )

    return await _paginate_token_digest_metadata(load_page, wanted, page_size=page_size)


async def _reverse_hash_deleted_key_metadata(
    prisma_client: PrismaClient,
    wanted: AbstractSet[str],
    *,
    page_size: int,
) -> Mapping[str, KeyMetadataDict]:
    async def load_page(skip: int) -> Sequence[_TokenAliasRecord] | None:
        return await _db_or_empty(
            lambda: DeletedVerificationTokenRepository(prisma_client).table.find_many(
                take=page_size,
                skip=skip,
                order=[{"deleted_at": "desc"}, {"id": "asc"}],  # mutable-ok: Prisma find_many order= is a dict
            ),
            "Failed reverse-hash recovery against deleted keys for %d missing keys: %s",
            len(wanted),
        )

    return await _paginate_token_digest_metadata(load_page, wanted, page_size=page_size)


async def _reverse_hash_key_metadata(
    prisma_client: PrismaClient,
    wanted: AbstractSet[str],
    *,
    page_size: int,
) -> Mapping[str, KeyMetadataDict]:
    from_active: Final = await _reverse_hash_active_key_metadata(prisma_client, wanted, page_size=page_size)
    still_wanted: Final = wanted - frozenset(from_active)
    if not still_wanted:
        return from_active
    from_deleted: Final = await _reverse_hash_deleted_key_metadata(prisma_client, still_wanted, page_size=page_size)
    return MappingProxyType({**from_active, **from_deleted})


async def _spend_logs_key_metadata(
    prisma_client: PrismaClient,
    wanted: AbstractSet[str],
) -> Mapping[str, KeyMetadataDict]:
    spend_log_rows: Final = await _db_or_empty(
        lambda: prisma_client.db.query_raw(
            _SPEND_LOGS_KEY_METADATA_SQL,
            tuple(wanted),
        ),
        "Failed SpendLogs metadata recovery for %d missing keys: %s",
        len(wanted),
    )
    if not isinstance(spend_log_rows, list):
        return _EMPTY_KEY_METADATA

    return MappingProxyType(
        {
            row["api_key"]: _spend_log_row_metadata(row)
            for row in spend_log_rows
            if isinstance(row, dict) and isinstance(row.get("api_key"), str) and row["api_key"] in wanted
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
            where={"user_id": {"in": tuple(user_ids)}},  # mutable-ok: Prisma find_many where= is a dict
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
    *,
    token_scan_page_size: int = _TOKEN_SCAN_PAGE,
) -> Mapping[str, KeyMetadataDict]:
    """
    Recover key_alias/team_id/user_email for DailyUserSpend.api_key values that
    were double-hashed by the v1.99 spend-log provenance gate.

    Those rows store hash(VerificationToken.token) instead of the token, so the
    exact join misses. Page through active then deleted tokens until every
    wanted digest is found or the table ends; fall back to SpendLogs metadata.
    Emails come from SpendLogs when present, otherwise from UserTable via the
    recovered key's user_id.
    """
    sha_missing: Final = frozenset(key for key in missing_keys if is_valid_sha256_hash(key))
    if not sha_missing:
        return _EMPTY_KEY_METADATA

    from_tokens: Final = await _reverse_hash_key_metadata(prisma_client, sha_missing, page_size=token_scan_page_size)
    still_missing: Final = sha_missing - frozenset(from_tokens)
    recovered: Final = (
        from_tokens
        if not still_missing
        else MappingProxyType({**from_tokens, **(await _spend_logs_key_metadata(prisma_client, still_missing))})
    )
    return await attach_user_emails(prisma_client, recovered)


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
            user_email_field: meta.get("user_email") or row.get(user_email_field),
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
        if isinstance(key, str)
        and key
        and (row.get(alias_field) in (None, "") or row.get(user_email_field) in (None, ""))
    )
    if not missing_keys:
        return tuple(rows)

    recovered: Final = await recover_double_hashed_key_metadata(prisma_client, missing_keys)
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
