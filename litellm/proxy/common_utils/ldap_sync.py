import asyncio
import math
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime, timedelta, timezone
from threading import Event
from typing import Final, Protocol

from fastapi import Request

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.common_utils.ldap_directory import DirectoryError, DirectoryGroup, LDAPConnection, read_directory
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager
from litellm.proxy.management_endpoints.team_endpoints import new_team
from litellm.proxy.management_helpers.team_roster_sync import (
    MembersMissing,
    RosterSync,
    RosterTarget,
    new_team_request_with_defaults,
    sync_team_roster,
)
from litellm.proxy.utils import PrismaClient, ProxyLogging
from litellm.repositories.team_repository import TeamRepository
from litellm.repositories.user_repository import UserRepository
from litellm.types.proxy.ldap_sync import LDAPSyncSettings

LDAP_SYNC_JOB_ID: Final = "ldap_team_sync"


class LDAPScheduler(Protocol):
    def add_job(
        self,
        func: Callable[[], Awaitable[None]],
        trigger: str,
        *,
        minutes: float,
        next_run_time: datetime,
        id: str,
        replace_existing: bool,
        max_instances: int,
        coalesce: bool,
    ) -> object: ...


async def resolve_user_ids(
    users: UserRepository, group: DirectoryGroup, settings: LDAPSyncSettings
) -> frozenset[str] | DirectoryError:
    rows: Final = (
        await users.find_by_emails(group.identities)
        if settings.litellm_match_field == "user_email"
        else await users.find_many_in(settings.litellm_match_field, group.identities)
    )
    identities: Final = tuple(
        row.user_email
        if settings.litellm_match_field == "user_email"
        else row.user_id
        if settings.litellm_match_field == "user_id"
        else row.sso_user_id
        for row in rows
    )
    normalized: Final = tuple(
        identity.casefold() if settings.litellm_match_field == "user_email" else identity
        for identity in identities
        if identity is not None
    )
    if len(normalized) != len(frozenset(normalized)):
        return DirectoryError("Multiple LiteLLM users match one LDAP identity")
    expected: Final = frozenset(
        identity.casefold() if settings.litellm_match_field == "user_email" else identity
        for identity in group.identities
    )
    verbose_proxy_logger.info(
        "LDAP sync team_id=%s matched_users=%d unmatched_users=%d",
        group.team_id,
        len(rows),
        len(expected - frozenset(normalized)),
    )
    return frozenset(row.user_id for row in rows)


class LDAPSync:
    def __init__(
        self,
        settings: LDAPSyncSettings,
        prisma_client: PrismaClient,
        pod_lock_manager: PodLockManager,
        user_api_key_cache: UserApiKeyCache,
        proxy_logging_obj: ProxyLogging | None,
        *,
        directory_connection: LDAPConnection | None = None,
    ) -> None:
        self.directory_connection = directory_connection
        self.settings = settings
        self.lease_seconds = math.ceil(settings.interval_minutes * 60) - 1
        self.time_budget_seconds = min(600, self.lease_seconds / 2)
        self.prisma_client = prisma_client
        self.pod_lock_manager = pod_lock_manager
        self.user_api_key_cache = user_api_key_cache
        self.proxy_logging_obj = proxy_logging_obj
        self.teams = TeamRepository(prisma_client, use_writer=True)
        self.users = UserRepository(prisma_client, use_writer=True)

    async def sync_group(self, group: DirectoryGroup) -> None:
        member_ids: Final = await resolve_user_ids(self.users, group, self.settings)
        if isinstance(member_ids, DirectoryError):
            verbose_proxy_logger.warning("LDAP sync skipped team_id=%s: %s", group.team_id, member_ids.reason)
            return
        team: Final = await self.teams.find_by_id(group.team_id, id_field="team_id")
        existing: Final[frozenset[str]] = (
            frozenset(member.user_id for member in team.members_with_roles if member.user_id is not None)
            if team is not None
            else frozenset()
        )
        if not member_ids and existing and not self.settings.allow_empty_groups:
            # An unreadable or misconfigured membership attribute looks exactly like an empty group.
            verbose_proxy_logger.warning(
                "LDAP sync skipped team_id=%s: group resolved to no LiteLLM users but the team has %d members; "
                "set allow_empty_groups to remove them",
                group.team_id,
                len(existing),
            )
            return
        if self.settings.dry_run:
            verbose_proxy_logger.info(
                "LDAP sync dry_run team_id=%s create_team=%s add=%s remove=%s",
                group.team_id,
                team is None,
                sorted(member_ids - existing),
                sorted(existing - member_ids),
            )
            return
        actor: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
        if team is None:
            await new_team(
                data=new_team_request_with_defaults(group.team_id, group.name, ()),
                http_request=Request(scope={"type": "http", "path": "/team/new"}),
                user_api_key_dict=actor,
            )
        outcome: Final = await sync_team_roster(
            self.prisma_client,
            group.team_id,
            RosterTarget(member_ids),
            actor,
            "ldap_sync",
            self.user_api_key_cache,
            self.proxy_logging_obj,
        )
        if isinstance(outcome, RosterSync):
            verbose_proxy_logger.info(
                "LDAP sync team_id=%s created=%s added=%d removed=%d",
                group.team_id,
                team is None,
                len(outcome.added),
                len(outcome.removed),
            )
            return
        verbose_proxy_logger.warning(
            "LDAP sync skipped team_id=%s: %s",
            group.team_id,
            "user disappeared before membership update" if isinstance(outcome, MembersMissing) else "team disappeared",
        )

    async def _sync(self) -> None:
        loop: Final = asyncio.get_running_loop()
        deadline: Final = loop.time() + self.time_budget_seconds
        cancellation: Final = Event()
        directory_settings: Final = self.settings.model_copy(
            update={"timeout_seconds": min(self.settings.timeout_seconds, max(1, self.lease_seconds // 4))}
        )
        try:
            # Only the read is cancelled on timeout; it performs no writes.
            snapshot: Final = await asyncio.wait_for(
                asyncio.to_thread(
                    read_directory, directory_settings, connection=self.directory_connection, stop=cancellation
                ),
                timeout=self.time_budget_seconds,
            )
        except asyncio.TimeoutError:
            verbose_proxy_logger.warning("LDAP sync timed out reading the directory; retrying next interval")
            return
        finally:
            cancellation.set()
        if isinstance(snapshot, DirectoryError):
            verbose_proxy_logger.warning("LDAP sync aborted: %s", snapshot.reason)
            return
        for group in snapshot:
            # Stop between teams rather than cancelling a team's write midway.
            if loop.time() >= deadline:
                verbose_proxy_logger.warning("LDAP sync reached its time budget; remaining groups retry next interval")
                return
            if isinstance(group, DirectoryError):
                verbose_proxy_logger.warning("LDAP sync skipped group_dn=%s: %s", group.group_dn, group.reason)
                continue
            try:
                await self.sync_group(group)
            except Exception as exc:  # noqa: BLE001  # isolate failed groups so later groups can still synchronize
                verbose_proxy_logger.warning(
                    "LDAP sync failed team_id=%s (%s); retrying next interval", group.team_id, type(exc).__name__
                )

    async def run(self) -> None:
        if (
            await self.pod_lock_manager.acquire_lock(
                cronjob_id=LDAP_SYNC_JOB_ID,
                ttl=self.lease_seconds,
                allow_reentrant=False,
            )
            is not True
        ):
            return
        try:
            await self._sync()
        finally:
            # Release on completion so a restart (e.g. dry_run -> apply) syncs immediately.
            await self.pod_lock_manager.release_lock(cronjob_id=LDAP_SYNC_JOB_ID)


def ldap_sync_settings(general_settings: Mapping[str, object]) -> LDAPSyncSettings | None:
    """The validated ``ldap_sync`` settings when the sync is enabled, otherwise ``None``."""
    raw: Final = general_settings.get("ldap_sync")
    settings: Final = None if raw is None else LDAPSyncSettings.model_validate(raw)
    return settings if settings is not None and settings.enabled else None


def initialize_ldap_sync(
    scheduler: LDAPScheduler,
    settings: LDAPSyncSettings,
    prisma_client: PrismaClient,
    pod_lock_manager: PodLockManager,
    user_api_key_cache: UserApiKeyCache,
    proxy_logging_obj: ProxyLogging,
) -> None:
    if pod_lock_manager.redis_cache is None:
        raise ValueError("ldap_sync requires shared Redis for cross-worker coordination")
    job: Final = LDAPSync(settings, prisma_client, pod_lock_manager, user_api_key_cache, proxy_logging_obj)
    scheduler.add_job(
        job.run,
        "interval",
        minutes=settings.interval_minutes,
        next_run_time=datetime.now(timezone.utc) + timedelta(seconds=1),
        id=LDAP_SYNC_JOB_ID,
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
