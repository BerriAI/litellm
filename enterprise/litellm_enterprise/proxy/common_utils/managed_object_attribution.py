"""Creator attribution for the spend row a managed-object cost poll writes.

A batch or a background response is billed by a poll job that runs long after the creating
request returned, so the managed object row's creator columns are the only record of which
key, team, organization, user, and tags carry the spend.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, Protocol, cast

from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.constants import CLI_SESSION_KEY_PREFIX
from litellm.repositories.team_repository import TeamRepository
from litellm.repositories.user_repository import UserRepository
from litellm.repositories.verification_token_repository import VerificationTokenRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models

    from litellm.proxy.utils import PrismaClient
    from litellm.repositories.prisma_protocols import TableActions


class ManagedObjectRow(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def unified_object_id(self) -> str: ...

    @property
    def created_by(self) -> str | None: ...

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


_TAG_LIST_ADAPTER: Final[TypeAdapter[list[object]]] = TypeAdapter(list[object])


def _string_tags(request_tags: object) -> list[str] | None:
    if not isinstance(request_tags, list):
        return None
    return [tag for tag in _TAG_LIST_ADAPTER.validate_python(request_tags) if isinstance(tag, str)] or None


def _user_table(prisma_client: PrismaClient) -> TableActions[prisma_models.LiteLLM_UserTable]:
    return UserRepository(prisma_client).table


def _token_table(prisma_client: PrismaClient) -> TableActions[prisma_models.LiteLLM_VerificationToken]:
    return VerificationTokenRepository(prisma_client).table


def _team_table(prisma_client: PrismaClient) -> TableActions[prisma_models.LiteLLM_TeamTable]:
    return TeamRepository(prisma_client).table


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
    """

    def __init__(self, prisma_client: PrismaClient, poller_name: str) -> None:
        self._prisma_client: Final = prisma_client
        self._poller_name: Final = poller_name

    async def build(self, job: ManagedObjectRow, object_id: str) -> Mapping[str, object]:
        api_key: Final = getattr(job, "api_key", None)
        team_id: Final = getattr(job, "team_id", None)
        request_tags: Final = getattr(job, "request_tags", None)
        key_alias: Final = await self._key_alias(object_id, api_key, job.created_by)
        team_alias: Final = await self._team_alias(team_id)
        org_id: Final = await self._org_id(job, object_id)
        tags: Final = _string_tags(request_tags)
        return {
            "user_api_key_user_id": job.created_by,
            "user_api_key": api_key,
            "user_api_key_hash": api_key,
            "user_api_key_team_id": team_id,
            **(await self._user_info(object_id, job.created_by)),
            **({"user_api_key_alias": key_alias} if key_alias is not None else {}),
            **({"user_api_key_team_alias": team_alias} if team_alias is not None else {}),
            **({"user_api_key_org_id": org_id} if org_id is not None else {}),
            **({"tags": tags} if tags else {}),
        }

    async def _user_info(self, object_id: str, user_id: str | None) -> Mapping[str, str | None]:
        """Objects created by a team or service account key carry no user id, and
        find_unique(where={"user_id": None}) raises, so those return nothing."""
        if not user_id:
            return {}
        try:
            user_row: Final[prisma_models.LiteLLM_UserTable | None] = await _user_table(
                self._prisma_client
            ).find_unique(where={"user_id": user_id})
        except Exception as e:
            verbose_proxy_logger.error(f"{self._poller_name}: could not look up user {user_id} for {object_id}: {e}")
            return {}
        if user_row is None:
            return {}
        return {
            "user_api_key_user_email": getattr(user_row, "user_email", None),
            "user_api_key_alias": getattr(user_row, "user_alias", None),
        }

    async def _key_alias(self, object_id: str, api_key: str | None, created_by: str | None) -> str | None:
        if not api_key:
            return None
        if created_by and api_key == f"{CLI_SESSION_KEY_PREFIX}-{created_by}":
            return api_key
        try:
            key_row: Final[prisma_models.LiteLLM_VerificationToken | None] = await _token_table(
                self._prisma_client
            ).find_unique(where={"token": api_key})
        except Exception as e:
            verbose_proxy_logger.error(f"{self._poller_name}: could not look up key alias for {object_id}: {e}")
            return None
        return getattr(key_row, "key_alias", None) if key_row is not None else None

    async def _team_alias(self, team_id: str | None) -> str | None:
        if not team_id:
            return None
        try:
            team_row: Final[prisma_models.LiteLLM_TeamTable | None] = await _team_table(
                self._prisma_client
            ).find_unique(where={"team_id": team_id})
        except Exception as e:
            verbose_proxy_logger.error(f"{self._poller_name}: could not look up team alias for team {team_id}: {e}")
            return None
        return getattr(team_row, "team_alias", None) if team_row is not None else None

    async def _org_id(self, job: ManagedObjectRow, object_id: str) -> str | None:
        org_id: Final = getattr(job, "org_id", None)
        if org_id:
            return org_id
        key_org_id: Final = await self._key_org_id(getattr(job, "api_key", None), object_id)
        if key_org_id:
            return key_org_id
        return await self._team_org_id(getattr(job, "team_id", None), object_id)

    async def _key_org_id(self, api_key: str | None, object_id: str) -> str | None:
        if not api_key:
            return None
        try:
            key_row: Final[prisma_models.LiteLLM_VerificationToken | None] = await _token_table(
                self._prisma_client
            ).find_unique(where={"token": api_key})
        except Exception as e:
            verbose_proxy_logger.error(
                f"{self._poller_name}: could not resolve the key's org for {object_id}, still trying the team's: {e}"
            )
            return None
        return cast("str | None", getattr(key_row, "organization_id", None)) if key_row is not None else None

    async def _team_org_id(self, team_id: str | None, object_id: str) -> str | None:
        if not team_id:
            return None
        try:
            team_row: Final[prisma_models.LiteLLM_TeamTable | None] = await _team_table(
                self._prisma_client
            ).find_unique(where={"team_id": team_id})
        except Exception as e:
            verbose_proxy_logger.error(f"{self._poller_name}: could not resolve the team's org for {object_id}: {e}")
            return None
        return cast("str | None", getattr(team_row, "organization_id", None)) if team_row is not None else None
