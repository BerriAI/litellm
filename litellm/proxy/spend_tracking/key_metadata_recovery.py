from collections.abc import Awaitable, Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Final, Protocol, TypeVar

from prisma.errors import PrismaError
from typing_extensions import TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.litellm_logging import is_valid_sha256_hash
from litellm.proxy.utils import PrismaClient, hash_token
from litellm.repositories.table_repositories import DeletedVerificationTokenRepository
from litellm.repositories.user_repository import UserRepository
from litellm.repositories.verification_token_repository import (
    VerificationTokenRepository,
)

_T = TypeVar("_T")

# Cap reverse-hash scans so a Usage page with orphaned double-hashed api_key
# values cannot pull an unbounded VerificationToken table into memory.
_MAX_DOUBLE_HASH_TOKEN_SCAN: Final = 10_000

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
    key_alias: str | None
    team_id: str | None
    user_id: str | None
    user_email: str | None


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
    try:
        return await load()
    except PrismaError as e:
        verbose_proxy_logger.warning(warning, count, e)
        return None


def _token_digest_metadata(
    records: Sequence[_TokenAliasRecord],
    wanted: AbstractSet[str],
) -> dict[str, KeyMetadataDict]:
    return {
        digested: {
            "key_alias": record.key_alias,
            "team_id": record.team_id,
            "user_id": getattr(record, "user_id", None),
        }
        for record in records
        for digested in (hash_token(record.token),)
        if digested in wanted
    }


async def _reverse_hash_active_key_metadata(
    prisma_client: PrismaClient,
    wanted: AbstractSet[str],
) -> dict[str, KeyMetadataDict]:
    active_records: Final = await _db_or_empty(
        lambda: VerificationTokenRepository(prisma_client).table.find_many(take=_MAX_DOUBLE_HASH_TOKEN_SCAN),
        "Failed reverse-hash recovery against active keys for %d missing keys: %s",
        len(wanted),
    )
    if active_records is None:
        return {}
    return _token_digest_metadata(active_records, wanted)


async def _reverse_hash_deleted_key_metadata(
    prisma_client: PrismaClient,
    wanted: AbstractSet[str],
) -> dict[str, KeyMetadataDict]:
    deleted_records: Final = await _db_or_empty(
        lambda: DeletedVerificationTokenRepository(prisma_client).table.find_many(
            take=_MAX_DOUBLE_HASH_TOKEN_SCAN,
            order={"deleted_at": "desc"},
        ),
        "Failed reverse-hash recovery against deleted keys for %d missing keys: %s",
        len(wanted),
    )
    if deleted_records is None:
        return {}
    return _token_digest_metadata(deleted_records, wanted)


async def _reverse_hash_key_metadata(
    prisma_client: PrismaClient,
    wanted: AbstractSet[str],
) -> dict[str, KeyMetadataDict]:
    from_active: Final = await _reverse_hash_active_key_metadata(prisma_client, wanted)
    still_wanted: Final = wanted - frozenset(from_active)
    if not still_wanted:
        return from_active
    return {**from_active, **(await _reverse_hash_deleted_key_metadata(prisma_client, still_wanted))}


async def _spend_logs_key_metadata(
    prisma_client: PrismaClient,
    wanted: AbstractSet[str],
) -> dict[str, KeyMetadataDict]:
    spend_log_rows: Final = await _db_or_empty(
        lambda: prisma_client.db.query_raw(
            _SPEND_LOGS_KEY_METADATA_SQL,
            list(wanted),
        ),
        "Failed SpendLogs metadata recovery for %d missing keys: %s",
        len(wanted),
    )
    if not isinstance(spend_log_rows, list):
        return {}

    return {
        row["api_key"]: {
            "key_alias": row.get("key_alias"),
            "team_id": row.get("team_id"),
            "user_id": row.get("user_id"),
            "user_email": row.get("user_email"),
        }
        for row in spend_log_rows
        if isinstance(row, dict) and isinstance(row.get("api_key"), str) and row["api_key"] in wanted
    }


async def _emails_for_user_ids(
    prisma_client: PrismaClient,
    user_ids: AbstractSet[str],
) -> Mapping[str, str]:
    if not user_ids:
        return {}
    users: Final = await _db_or_empty(
        lambda: UserRepository(prisma_client).table.find_many(where={"user_id": {"in": list(user_ids)}}),
        "Failed user_email recovery for %d user ids: %s",
        len(user_ids),
    )
    if users is None:
        return {}
    return {
        user.user_id: user.user_email
        for user in users
        if getattr(user, "user_id", None) and getattr(user, "user_email", None)
    }


def _meta_with_email(meta: KeyMetadataDict, emails: Mapping[str, str]) -> KeyMetadataDict:
    if meta.get("user_email"):
        return meta
    user_id: Final = meta.get("user_id")
    if not isinstance(user_id, str) or user_id not in emails:
        return meta
    return {**meta, "user_email": emails[user_id]}


async def _with_user_emails(
    prisma_client: PrismaClient,
    recovered: Mapping[str, KeyMetadataDict],
) -> dict[str, KeyMetadataDict]:
    needing_email: Final = frozenset(
        user_id
        for meta in recovered.values()
        for user_id in (meta.get("user_id"),)
        if isinstance(user_id, str) and user_id and not meta.get("user_email")
    )
    emails: Final = await _emails_for_user_ids(prisma_client, needing_email)
    if not emails:
        return dict(recovered)
    return {api_key: _meta_with_email(meta, emails) for api_key, meta in recovered.items()}


async def recover_double_hashed_key_metadata(
    prisma_client: PrismaClient,
    missing_keys: AbstractSet[str],
) -> dict[str, KeyMetadataDict]:
    """
    Recover key_alias/team_id/user_email for DailyUserSpend.api_key values that
    were double-hashed by the v1.99 spend-log provenance gate.

    Those rows store hash(VerificationToken.token) instead of the token, so the
    exact join misses. Prefer a bounded reverse-hash against active/deleted
    tokens; fall back to SpendLogs metadata. Emails come from SpendLogs when
    present, otherwise from UserTable via the recovered key's user_id.
    """
    sha_missing: Final = frozenset(key for key in missing_keys if is_valid_sha256_hash(key))
    if not sha_missing:
        return {}

    from_tokens: Final = await _reverse_hash_key_metadata(prisma_client, sha_missing)
    still_missing: Final = sha_missing - frozenset(from_tokens)
    recovered: Final = (
        from_tokens
        if not still_missing
        else {**from_tokens, **(await _spend_logs_key_metadata(prisma_client, still_missing))}
    )
    return await _with_user_emails(prisma_client, recovered)


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
    return {
        **row,
        alias_field: meta.get("key_alias") or row.get(alias_field),
        team_id_field: meta.get("team_id") or row.get(team_id_field),
        user_email_field: meta.get("user_email") or row.get(user_email_field),
    }


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
