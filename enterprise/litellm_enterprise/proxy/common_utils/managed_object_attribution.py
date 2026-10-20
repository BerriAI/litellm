"""Creator attribution for the spend row a managed-object cost poll writes.

A batch or a background response is billed by a poll job that runs long after the creating
request returned, so the managed object row's creator columns are the only record of which
key, team, organization, user, end user, and tags carry the spend.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Final, Protocol, TypeGuard, cast

from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.constants import CLI_SESSION_KEY_PREFIX
from litellm.repositories.table_repositories import (
    DeletedVerificationTokenRepository,
    DeprecatedVerificationTokenRepository,
)
from litellm.repositories.team_repository import TeamRepository
from litellm.repositories.user_repository import UserRepository
from litellm.repositories.verification_token_repository import VerificationTokenRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models

    from litellm.proxy.utils import PrismaClient
    from litellm.repositories.prisma_protocols import TableActions


class ManagedObjectRow(Protocol):
    @property
    def created_at(self) -> datetime: ...

    @property
    def updated_at(self) -> datetime: ...

    @property
    def id(self) -> str: ...

    @property
    def unified_object_id(self) -> str: ...

    @property
    def model_object_id(self) -> str: ...

    @property
    def created_by(self) -> str | None: ...

    @property
    def status(self) -> str | None: ...

    @property
    def file_object(self) -> object: ...

    @property
    def org_id(self) -> str | None: ...

    @property
    def api_key(self) -> str | None: ...

    @property
    def team_id(self) -> str | None: ...

    @property
    def request_tags(self) -> object: ...

    @property
    def end_user_id(self) -> str | None: ...


_TAG_LIST_ADAPTER: Final[TypeAdapter[list[object]]] = TypeAdapter(list[object])
_KEY_OWNER_SCOPE_PREFIX: Final = "key:"
_MAX_ROTATION_HOPS: Final = 16


def _string_tags(request_tags: object) -> list[str] | None:
    if not isinstance(request_tags, list):
        return None
    return [tag for tag in _TAG_LIST_ADAPTER.validate_python(request_tags) if isinstance(tag, str)] or None


def _creator_user_id(created_by: str | None) -> str | None:
    if created_by is None or created_by.startswith(_KEY_OWNER_SCOPE_PREFIX):
        return None
    return created_by


def _user_table(prisma_client: PrismaClient) -> TableActions[prisma_models.LiteLLM_UserTable]:
    return UserRepository(prisma_client).table


def _token_table(prisma_client: PrismaClient) -> TableActions[prisma_models.LiteLLM_VerificationToken]:
    return VerificationTokenRepository(prisma_client).table


def _team_table(prisma_client: PrismaClient) -> TableActions[prisma_models.LiteLLM_TeamTable]:
    return TeamRepository(prisma_client).table


def _rotated_token_table(
    prisma_client: PrismaClient,
) -> TableActions[prisma_models.LiteLLM_DeprecatedVerificationToken]:
    return DeprecatedVerificationTokenRepository(prisma_client).table


def _deleted_token_table(prisma_client: PrismaClient) -> TableActions[prisma_models.LiteLLM_DeletedVerificationToken]:
    return DeletedVerificationTokenRepository(prisma_client).table


def _deleted_team_table(prisma_client: PrismaClient) -> TableActions[prisma_models.LiteLLM_DeletedTeamTable]:
    return TeamRepository(prisma_client).deleted_table


@dataclass(frozen=True, slots=True)
class _CreatorRows:
    key: prisma_models.LiteLLM_VerificationToken | None
    team: prisma_models.LiteLLM_TeamTable | None
    user: prisma_models.LiteLLM_UserTable | None


@dataclass(frozen=True, slots=True)
class ResponseCreator:
    metadata: Mapping[str, object]
    key_metadata: Mapping[str, object]
    team_id: str | None
    team_metadata: Mapping[str, object]
    settings_read: bool = True


class ManagedObjectCreatorAttribution:
    """Rebuilds the spend-tracking metadata of whoever created a managed object, so the poll
    job's spend row is attributed like the creating request: key hash, team, organization,
    user, aliases, and tags.

    Rows created before api_key and request_tags were persisted carry only created_by and
    team_id, and fall back to those. A named creating key owns user_api_key_alias; when it
    has no alias, or the key has since been rotated or deleted, the field keeps the creating
    user's alias, because a resolvable name is more useful on the spend row than a null.
    user_api_key_org_id is resolved too: the spend update writer reads it off this metadata
    to increment organization spend, so leaving it out drops the cost from org accounting.
    A background response also carries its end user, the creating key's project and its alias,
    and the key, team, and user budgets and per-model budgets, the way the read that billed it
    before did, and never names a key-scoped creator as its user. A key that sets no budget of
    its own takes its budget tier's, the same rule auth applies. Its creator's key and team
    metadata come along too, so the billed event reaches the key's and team's logging callbacks
    like that read did. A rotated key's metadata is read from the live key its rotations lead
    to, and a deleted key's or team's from its deleted row. A lookup that fails leaves the
    creator's settings unread and the attribution to the rows that still load, so the
    response still bills and still counts toward their budgets.
    """

    def __init__(self, prisma_client: PrismaClient, poller_name: str) -> None:
        self._prisma_client: Final = prisma_client
        self._poller_name: Final = poller_name

    async def batch_metadata(self, job: ManagedObjectRow, object_id: str) -> Mapping[str, object]:
        rows: Final = await self._rows(job, object_id)
        return {"user_api_key_user_id": job.created_by, **_shared_metadata(job, rows)}

    async def response_creator(self, job: ManagedObjectRow, object_id: str) -> ResponseCreator:
        try:
            return await self._read_response_creator(job, object_id)
        except Exception as e:
            verbose_proxy_logger.error(
                f"{self._poller_name}: could not read the creator of {object_id}, so it bills without their "
                f"settings: {e}"
            )
            return ResponseCreator(
                metadata=_response_creator_metadata(job, await self._rows(job, object_id)),
                key_metadata={},
                team_id=getattr(job, "team_id", None),
                team_metadata={},
                settings_read=False,
            )

    async def _read_response_creator(self, job: ManagedObjectRow, object_id: str) -> ResponseCreator:
        api_key: Final = getattr(job, "api_key", None)
        team_id: Final = getattr(job, "team_id", None)
        rows: Final = _CreatorRows(
            key=await self._live_key_row(api_key, job.created_by),
            team=await self._live_team_row(team_id),
            user=await self._user_row(object_id, _creator_user_id(job.created_by)),
        )
        return ResponseCreator(
            metadata=_response_creator_metadata(job, rows),
            key_metadata=row_metadata(rows.key) if rows.key is not None else await self._retired_key_metadata(api_key),
            team_id=team_id,
            team_metadata=(
                row_metadata(rows.team) if rows.team is not None else await self._deleted_team_metadata(team_id)
            ),
        )

    async def _retired_key_metadata(self, api_key: str | None) -> Mapping[str, object]:
        if not _names_a_key_row(api_key):
            return {}
        current_token: Final = await self._token_rotated_to(api_key, _MAX_ROTATION_HOPS)
        current_key: Final = (
            await _token_table(self._prisma_client).find_unique(where={"token": current_token})
            if current_token != api_key
            else None
        )
        if current_key is not None:
            return row_metadata(current_key)
        return row_metadata(
            await _deleted_token_table(self._prisma_client).find_first(
                where={"token": current_token}, order={"deleted_at": "desc"}
            )
        )

    async def _token_rotated_to(self, token: str, hops_left: int) -> str:
        rotation: Final = await _rotated_token_table(self._prisma_client).find_unique(where={"token": token})
        if rotation is None or hops_left == 0:
            return token
        return await self._token_rotated_to(rotation.active_token_id, hops_left - 1)

    async def _deleted_team_metadata(self, team_id: str | None) -> Mapping[str, object]:
        if not team_id:
            return {}
        return row_metadata(
            await _deleted_team_table(self._prisma_client).find_first(
                where={"team_id": team_id}, order={"deleted_at": "desc"}
            )
        )

    async def _rows(self, job: ManagedObjectRow, object_id: str) -> _CreatorRows:
        return _CreatorRows(
            key=await self._key_row(object_id, getattr(job, "api_key", None), job.created_by),
            team=await self._team_row(object_id, getattr(job, "team_id", None)),
            user=await self._user_row(object_id, _creator_user_id(job.created_by)),
        )

    async def _user_row(self, object_id: str, user_id: str | None) -> prisma_models.LiteLLM_UserTable | None:
        """Objects created by a team or service account key carry no user id, and
        find_unique(where={"user_id": None}) raises, so those resolve no row."""
        if not user_id:
            return None
        try:
            return await _user_table(self._prisma_client).find_unique(where={"user_id": user_id})
        except Exception as e:
            verbose_proxy_logger.error(f"{self._poller_name}: could not look up user {user_id} for {object_id}: {e}")
            return None

    async def _key_row(
        self, object_id: str, api_key: str | None, created_by: str | None
    ) -> prisma_models.LiteLLM_VerificationToken | None:
        try:
            return await self._live_key_row(api_key, created_by)
        except Exception as e:
            verbose_proxy_logger.error(f"{self._poller_name}: could not look up the creating key for {object_id}: {e}")
            return None

    async def _live_key_row(
        self, api_key: str | None, created_by: str | None
    ) -> prisma_models.LiteLLM_VerificationToken | None:
        if not api_key or _is_cli_session_key(api_key, created_by):
            return None
        return await _token_table(self._prisma_client).find_unique(
            where={"token": api_key}, include={"litellm_budget_table": True, "litellm_project_table": True}
        )

    async def _team_row(self, object_id: str, team_id: str | None) -> prisma_models.LiteLLM_TeamTable | None:
        try:
            return await self._live_team_row(team_id)
        except Exception as e:
            verbose_proxy_logger.error(f"{self._poller_name}: could not look up team {team_id} for {object_id}: {e}")
            return None

    async def _live_team_row(self, team_id: str | None) -> prisma_models.LiteLLM_TeamTable | None:
        if not team_id:
            return None
        return await _team_table(self._prisma_client).find_unique(where={"team_id": team_id})


def _response_creator_metadata(job: ManagedObjectRow, rows: _CreatorRows) -> Mapping[str, object]:
    end_user_id: Final = getattr(job, "end_user_id", None)
    return {
        "user_api_key_user_id": _creator_user_id(job.created_by),
        **_shared_metadata(job, rows),
        **({"user_api_key_end_user_id": end_user_id} if end_user_id else {}),
        "user_api_key_project_id": getattr(rows.key, "project_id", None),
        "user_api_key_project_alias": getattr(getattr(rows.key, "litellm_project_table", None), "project_alias", None),
        **_key_budget_fields(rows.key),
        **_budget_fields(
            rows.team,
            "user_api_key_team_spend",
            "user_api_key_team_max_budget",
            "user_api_key_team_model_max_budget",
        ),
        **_budget_fields(
            rows.user,
            "user_api_key_user_spend",
            "user_api_key_user_max_budget",
            "user_api_key_user_model_max_budget",
        ),
    }


def _shared_metadata(job: ManagedObjectRow, rows: _CreatorRows) -> Mapping[str, object]:
    api_key: Final = getattr(job, "api_key", None)
    team_id: Final = getattr(job, "team_id", None)
    key_alias: Final = _key_alias(api_key, job.created_by, rows.key)
    team_alias: Final = getattr(rows.team, "team_alias", None)
    org_id: Final = getattr(job, "org_id", None) or _organization_id(rows.key) or _organization_id(rows.team)
    tags: Final = _string_tags(getattr(job, "request_tags", None))
    return {
        "user_api_key": api_key,
        "user_api_key_hash": api_key,
        "user_api_key_team_id": team_id,
        **(
            {
                "user_api_key_user_email": getattr(rows.user, "user_email", None),
                "user_api_key_alias": getattr(rows.user, "user_alias", None),
            }
            if rows.user is not None
            else {}
        ),
        **({"user_api_key_alias": key_alias} if key_alias is not None else {}),
        **({"user_api_key_team_alias": team_alias} if team_alias is not None else {}),
        **({"user_api_key_org_id": org_id} if org_id is not None else {}),
        **({"tags": tags} if tags else {}),
    }


def row_metadata(row: object) -> Mapping[str, object]:
    metadata: Final = getattr(row, "metadata", None)
    if not isinstance(metadata, dict):
        return {}
    return cast(Mapping[str, object], metadata)  # cast-ok: prisma Json object column


def _names_a_key_row(api_key: object) -> TypeGuard[str]:
    from litellm.litellm_core_utils.litellm_logging import is_valid_sha256_hash

    return isinstance(api_key, str) and is_valid_sha256_hash(api_key)


def _is_cli_session_key(api_key: str, created_by: str | None) -> bool:
    return bool(created_by) and api_key == f"{CLI_SESSION_KEY_PREFIX}-{created_by}"


def _key_alias(
    api_key: str | None, created_by: str | None, key_row: prisma_models.LiteLLM_VerificationToken | None
) -> str | None:
    if api_key and _is_cli_session_key(api_key, created_by):
        return api_key
    return getattr(key_row, "key_alias", None) if key_row is not None else None


def _organization_id(row: object) -> str | None:
    return cast("str | None", getattr(row, "organization_id", None)) if row is not None else None


def _budget_fields(row: object, spend_key: str, max_budget_key: str, model_max_budget_key: str) -> Mapping[str, object]:
    if row is None:
        return {}
    return {
        spend_key: getattr(row, "spend", None),
        max_budget_key: getattr(row, "max_budget", None),
        model_max_budget_key: getattr(row, "model_max_budget", None),
    }


def _key_budget_fields(key: object) -> Mapping[str, object]:
    if key is None:
        return {}
    tier: Final = getattr(key, "litellm_budget_table", None)
    return {
        "user_api_key_spend": getattr(key, "spend", None),
        "user_api_key_max_budget": _own_or_tier_budget(key, tier, "max_budget"),
        "user_api_key_model_max_budget": _own_or_tier_budget(key, tier, "model_max_budget"),
    }


def _own_or_tier_budget(key: object, tier: object, field: str) -> object:
    own: Final = getattr(key, field, None)
    tier_value: Final = getattr(tier, field, None)
    return own if tier_value is None or own not in (None, {}) else tier_value
