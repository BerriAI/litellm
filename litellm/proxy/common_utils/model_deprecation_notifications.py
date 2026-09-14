"""Email team admins about the deprecating models their team can reach"""

from __future__ import annotations

import html
import os
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol

from pydantic import BaseModel, Field, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.constants import EMAIL_MODEL_DEPRECATION_LOCK_ID
from litellm.integrations.email_alerting import (
    LITELLM_LOGO_URL,
    LITELLM_SUPPORT_CONTACT,
    get_team_admin_emails,
)
from litellm.integrations.email_templates.email_footer import EMAIL_FOOTER
from litellm.integrations.email_templates.model_deprecation_email import (
    MODEL_DEPRECATION_EMAIL_ROW_TEMPLATE,
    MODEL_DEPRECATION_EMAIL_TEMPLATE,
)
from litellm.models.team import LiteLLM_TeamTable
from litellm.proxy._types import ProxyException
from litellm.proxy.auth.auth_checks import can_team_access_model
from litellm.proxy.common_utils.model_deprecation import collect_model_deprecations
from litellm.repositories.team_repository import TeamRepository
from litellm.types.integrations.slack_alerting import SlackAlertingArgs, SlackAlertingCacheKeys
from litellm.types.proxy.model_deprecation import (
    DEFAULT_DEPRECATION_CHECK_INTERVAL_SECONDS,
    ModelDeprecationInfo,
)

if TYPE_CHECKING:
    from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager
    from litellm.proxy.utils import PrismaClient
    from litellm.router import Router


@dataclass(frozen=True, slots=True)
class AffectedModel:
    info: ModelDeprecationInfo
    display_name: str
    milestone: int


@dataclass(frozen=True, slots=True)
class _DeploymentOwner:
    model_name: str
    team_id: str
    public_name: str


@dataclass(frozen=True, slots=True)
class TeamNotification:
    team_id: str
    team_alias: str | None
    recipients: tuple[str, ...]
    models: tuple[AffectedModel, ...]


class DeprecationEmailCache(Protocol):
    async def async_get_cache(self, *, key: str) -> object: ...

    async def async_set_cache(self, *, key: str, value: float, ttl: float) -> None: ...


class EmailSender(Protocol):
    DEFAULT_LITELLM_EMAIL: str

    async def send_email(self, from_email: str, to_email: Sequence[str], subject: str, html_body: str) -> None: ...


class SmtpSend(Protocol):
    def __call__(self, *, receiver_email: str, subject: str, html: str) -> Awaitable[object]: ...


@dataclass(frozen=True, slots=True)
class DeprecationEmailContext:
    llm_router: Router
    prisma_client: PrismaClient
    cache: DeprecationEmailCache
    alerting_args: SlackAlertingArgs
    pod_lock_manager: PodLockManager | None
    deliver: Callable[[Sequence[str], str, str], Awaitable[None]]


def select_milestone(days_until: int, thresholds: Sequence[int]) -> int | None:
    """The most urgent threshold already reached, None while the first one is still ahead"""
    return min((threshold for threshold in thresholds if days_until <= threshold), default=None)


def _reached(info: ModelDeprecationInfo, thresholds: Sequence[int]) -> tuple[ModelDeprecationInfo, int] | None:
    milestone: Final = select_milestone(info.days_until_deprecation, thresholds)
    return None if milestone is None else (info, milestone)


class _TeamScope(BaseModel):
    team_id: str | None = None
    team_public_model_name: str | None = None


class _OwnedDeployment(BaseModel):
    model_name: str
    model_info: _TeamScope = Field(default_factory=_TeamScope)


def _owner_of(deployment: Mapping[str, object]) -> _DeploymentOwner | None:
    try:
        parsed: Final = _OwnedDeployment.model_validate(deployment)
    except ValidationError:
        return None
    if not parsed.model_info.team_id:
        return None
    return _DeploymentOwner(
        model_name=parsed.model_name,
        team_id=parsed.model_info.team_id,
        public_name=parsed.model_info.team_public_model_name or parsed.model_name,
    )


def _deployment_owners(llm_router: Router) -> Mapping[str, _DeploymentOwner]:
    """Team-scoped deployments keyed by their internal model name, with the name the team knows"""
    owners: Final = (_owner_of(deployment) for deployment in llm_router.get_model_list() or ())
    return MappingProxyType({owner.model_name: owner for owner in owners if owner is not None})


async def _team_can_access(model_name: str, team: LiteLLM_TeamTable, llm_router: Router) -> bool:
    """Reuse the auth check so wildcards, access groups and the empty-list rule match real requests"""
    try:
        await can_team_access_model(model=model_name, team_object=team, llm_router=llm_router)
    except ProxyException:
        return False
    except Exception as e:  # noqa: BLE001  # a malformed allowlist (e.g. a wildcard with regex metacharacters) must not abort the pass
        verbose_proxy_logger.debug("model_deprecation: access check failed for team %s: %s", team.team_id, e)
        return False
    return True


async def _display_name_for(
    info: ModelDeprecationInfo, team: LiteLLM_TeamTable, owner: _DeploymentOwner | None, llm_router: Router
) -> str | None:
    if owner is not None:
        return owner.public_name if owner.team_id == team.team_id else None
    if await _team_can_access(info.model_name, team, llm_router):
        return info.model_name
    return None


async def _affected_model(
    reached: tuple[ModelDeprecationInfo, int],
    team: LiteLLM_TeamTable,
    owners: Mapping[str, _DeploymentOwner],
    llm_router: Router,
) -> AffectedModel | None:
    info, milestone = reached
    display_name: Final = await _display_name_for(info, team, owners.get(info.model_name), llm_router)
    return None if display_name is None else AffectedModel(info=info, display_name=display_name, milestone=milestone)


async def _affected_for_team(
    team: LiteLLM_TeamTable,
    reached: Sequence[tuple[ModelDeprecationInfo, int]],
    owners: Mapping[str, _DeploymentOwner],
    llm_router: Router,
) -> tuple[AffectedModel, ...]:
    candidates: Final = tuple([await _affected_model(pair, team, owners, llm_router) for pair in reached])
    return tuple(model for model in candidates if model is not None)


async def resolve_affected_teams(
    infos: Sequence[ModelDeprecationInfo],
    llm_router: Router,
    teams: Sequence[LiteLLM_TeamTable],
    thresholds: Sequence[int],
) -> Mapping[str, tuple[AffectedModel, ...]]:
    """Per team id, the deprecating models it can reach that have crossed a threshold"""
    reached: Final = tuple(pair for pair in (_reached(info, thresholds) for info in infos) if pair is not None)
    owners: Final = _deployment_owners(llm_router)
    per_team: Final = MappingProxyType(
        {
            team.team_id: await _affected_for_team(team, reached, owners, llm_router)
            for team in teams
            if not team.blocked
        }
    )
    return MappingProxyType({team_id: models for team_id, models in per_team.items() if models})


def email_sent_key(team_id: str, model_name: str, milestone: int) -> str:
    return f"model_deprecation_email:{team_id}:{model_name}:{milestone}"


async def _unsent(
    team_id: str, models: Sequence[AffectedModel], cache: DeprecationEmailCache
) -> tuple[AffectedModel, ...]:
    flags: Final = tuple(
        [
            await cache.async_get_cache(key=email_sent_key(team_id, model.info.model_name, model.milestone)) is None
            for model in models
        ]
    )
    return tuple(model for model, unsent in zip(models, flags, strict=True) if unsent)


async def _notification_for(
    team: LiteLLM_TeamTable, models: Sequence[AffectedModel], cache: DeprecationEmailCache, prisma_client: PrismaClient
) -> TeamNotification | None:
    unsent: Final = await _unsent(team.team_id, models, cache)
    if not unsent:
        return None
    recipients: Final = await get_team_admin_emails(team, prisma_client)
    if not recipients:
        verbose_proxy_logger.debug("model_deprecation: team %s has no admin emails, skipping", team.team_id)
        return None
    return TeamNotification(team_id=team.team_id, team_alias=team.team_alias, recipients=recipients, models=unsent)


async def build_team_notifications(
    affected: Mapping[str, Sequence[AffectedModel]],
    teams: Sequence[LiteLLM_TeamTable],
    cache: DeprecationEmailCache,
    prisma_client: PrismaClient,
) -> tuple[TeamNotification, ...]:
    """One digest per team of the milestones not yet emailed, dropping teams with nobody to email"""
    teams_by_id: Final = MappingProxyType({team.team_id: team for team in teams})
    candidates: Final = tuple(
        [
            await _notification_for(teams_by_id[team_id], models, cache, prisma_client)
            for team_id, models in affected.items()
        ]
    )
    return tuple(notification for notification in candidates if notification is not None)


def _days_left_label(days_until: int) -> str:
    if days_until < 0:
        return f"deprecated {abs(days_until)}d ago"
    return "today" if days_until == 0 else f"{days_until}d"


def _render_row(model: AffectedModel) -> str:
    return MODEL_DEPRECATION_EMAIL_ROW_TEMPLATE.format(
        model_name=html.escape(model.display_name),
        provider=html.escape(model.info.litellm_provider or "unknown"),
        deprecation_date=model.info.deprecation_date.isoformat(),
        days_left=_days_left_label(model.info.days_until_deprecation),
        status=model.info.status,
    )


def render_model_deprecation_email(
    notification: TeamNotification, email_logo_url: str, email_support_contact: str
) -> tuple[str, str]:
    """(subject, html) for one team's digest, every model-sourced string escaped"""
    alias: Final = " ".join((notification.team_alias or "").split())
    team_name: Final = alias or notification.team_id
    verb: Final = (
        "deprecated" if any(model.info.days_until_deprecation < 0 for model in notification.models) else "deprecating"
    )
    subject: Final = f"[LiteLLM] {len(notification.models)} model(s) {verb} for team {team_name}"
    body: Final = MODEL_DEPRECATION_EMAIL_TEMPLATE.format(
        email_logo_url=html.escape(email_logo_url),
        team_name=html.escape(team_name),
        model_rows="\n".join(_render_row(model) for model in notification.models),
        email_support_contact=html.escape(email_support_contact),
    )
    return subject, body + EMAIL_FOOTER


def make_email_deliverer(
    email_logger: EmailSender | None, smtp_send: SmtpSend | None = None
) -> Callable[[Sequence[str], str, str], Awaitable[None]]:
    """The proxy's configured email provider when there is one, else the OSS SMTP helper per recipient"""
    if email_logger is not None:

        async def deliver_via_logger(recipients: Sequence[str], subject: str, html_body: str) -> None:
            await email_logger.send_email(
                from_email=email_logger.DEFAULT_LITELLM_EMAIL,
                to_email=tuple(recipients),
                subject=subject,
                html_body=html_body,
            )

        return deliver_via_logger

    async def deliver_over_smtp(recipients: Sequence[str], subject: str, html_body: str) -> None:
        from litellm.proxy.utils import send_email

        send: Final = smtp_send or send_email
        for recipient in recipients:
            await send(receiver_email=recipient, subject=subject, html=html_body)

    return deliver_over_smtp


async def _load_teams(prisma_client: PrismaClient) -> tuple[LiteLLM_TeamTable, ...]:
    return tuple(await TeamRepository(prisma_client).find_many())


async def _stamp_pass(cache: DeprecationEmailCache) -> None:
    await cache.async_set_cache(
        key=SlackAlertingCacheKeys.deprecation_email_pass_key.value,
        value=time.time(),
        ttl=DEFAULT_DEPRECATION_CHECK_INTERVAL_SECONDS,
    )


async def _claimed_email_window(pod_lock_manager: PodLockManager | None) -> bool:
    """Without a redis backed lock there is no fleet to coordinate, so a lone pod always sends"""
    if pod_lock_manager is None:
        return True
    return (
        await pod_lock_manager.acquire_lock(
            cronjob_id=EMAIL_MODEL_DEPRECATION_LOCK_ID,
            ttl=DEFAULT_DEPRECATION_CHECK_INTERVAL_SECONDS,
            allow_reentrant=False,
        )
    ) is not False


async def _send_team_notification(notification: TeamNotification, ctx: DeprecationEmailContext) -> bool:
    subject, html_body = render_model_deprecation_email(
        notification,
        email_logo_url=os.getenv("SMTP_SENDER_LOGO", os.getenv("EMAIL_LOGO_URL", LITELLM_LOGO_URL)),
        email_support_contact=os.getenv("EMAIL_SUPPORT_CONTACT", LITELLM_SUPPORT_CONTACT),
    )
    try:
        await ctx.deliver(notification.recipients, subject, html_body)
    except Exception as e:  # noqa: BLE001  # one team's mail failure must not block the rest; unstamped keys retry tomorrow
        verbose_proxy_logger.exception("model_deprecation: email to team %s failed: %s", notification.team_id, e)
        return False
    for model in notification.models:
        await ctx.cache.async_set_cache(
            key=email_sent_key(notification.team_id, model.info.model_name, model.milestone),
            value=time.time(),
            ttl=ctx.alerting_args.model_deprecation_email_ttl,
        )
    return True


async def send_model_deprecation_emails(ctx: DeprecationEmailContext) -> int:
    """One pass: resolve affected teams, drop what was already sent, deliver one digest per team

    The pass stamp is set whenever a resolution completed, sent or not, so the DB is consulted at most
    once a day; the fleet lock is only claimed once there is something to send
    """
    snapshot: Final = collect_model_deprecations(llm_router=ctx.llm_router)
    infos: Final = (*snapshot.deprecated, *snapshot.imminent, *snapshot.upcoming)
    if not infos:
        await _stamp_pass(ctx.cache)
        return 0
    teams: Final = await _load_teams(ctx.prisma_client)
    affected: Final = await resolve_affected_teams(
        infos, ctx.llm_router, teams, ctx.alerting_args.model_deprecation_email_thresholds
    )
    notifications: Final = await build_team_notifications(affected, teams, ctx.cache, ctx.prisma_client)
    await _stamp_pass(ctx.cache)
    if not notifications or not await _claimed_email_window(ctx.pod_lock_manager):
        return 0
    results: Final = tuple([await _send_team_notification(notification, ctx) for notification in notifications])
    return sum(results)
