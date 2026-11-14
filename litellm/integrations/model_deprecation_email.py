"""Two owner-scoped retirement reminders, with durable per-recipient delivery receipts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from pydantic import AliasPath, BaseModel, ConfigDict, Field, TypeAdapter

from litellm.integrations.email_templates.model_deprecation_email import (
    render_model_deprecation_email,
)
from litellm.models.team import LiteLLM_TeamTable
from litellm.proxy.common_utils.model_deprecation import collect_model_deprecations
from litellm.repositories.config_repository import ConfigRepository
from litellm.repositories.table_repositories import DailyUserSpendRepository
from litellm.repositories.team_repository import TeamRepository
from litellm.repositories.user_repository import UserRepository
from litellm.repositories.verification_token_repository import (
    VerificationTokenRepository,
)
from litellm.types.integrations.model_deprecation_email import (
    MODEL_DEPRECATION_EMAIL_EVENT,
    DeprecationEmailMilestone,
    ModelDeprecationEmail,
)
from litellm.types.proxy.model_deprecation import (
    DEPRECATION_IDLE_POLL_SECONDS,
    ModelDeprecationInfo,
)

if TYPE_CHECKING:
    from litellm import Router
    from litellm.proxy.utils import PrismaClient

_RECEIPT_PREFIX: Final = "model_deprecation_email:"
_PAGE_SIZE: Final = 100


class _OptIn(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    enabled: bool = Field(
        False,
        strict=True,
        validation_alias=AliasPath("email_settings", MODEL_DEPRECATION_EMAIL_EVENT),
    )


class _RequestCount(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    api_requests: int


class _UsedKey(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    api_key: str
    metrics: _RequestCount = Field(alias="_sum")


_USED_KEYS: Final = TypeAdapter(tuple[_UsedKey, ...])


class _DeliveryStatus(str, Enum):
    SENDING = "sending"
    SENT = "sent"


class _Receipt(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)
    revision: int = 0
    status: _DeliveryStatus = _DeliveryStatus.SENDING
    lease_until: datetime


@dataclass(frozen=True, slots=True)
class _Impact:
    key_count: int = 0
    requests: int = 0
    teams: frozenset[str] = frozenset()


def _admin_ids(team: LiteLLM_TeamTable | None) -> frozenset[str]:
    if team is None:
        return frozenset()
    return frozenset(
        member.user_id for member in (team.members_with_roles or ()) if member.role == "admin" and member.user_id
    )


def _recipient_emails(
    user_id: str | None, team: LiteLLM_TeamTable | None, addresses: Mapping[str, str]
) -> frozenset[str]:
    owner_ids: Final[frozenset[str]] = frozenset((user_id,)) if user_id else frozenset()
    resolved: Final = frozenset(addresses[owner] for owner in owner_ids | _admin_ids(team) if owner in addresses)
    if team is None:
        return resolved
    return resolved | frozenset(
        member.user_email.strip()
        for member in (team.members_with_roles or ())
        if member.role == "admin"
        and member.user_email
        and member.user_email.strip()
        and member.user_id not in addresses
    )


def _usage_model(model: ModelDeprecationInfo) -> str:
    name: Final = model.litellm_model or model.model_name
    provider: Final = model.litellm_provider
    if provider and not name.startswith(f"{provider}/"):
        return f"{provider}/{name}"
    return name


def _receipt_key(email: ModelDeprecationEmail) -> str:
    identity: Final = json.dumps(
        [
            email.usage_model,
            email.model.deprecation_date.isoformat(),
            int(email.milestone),
            email.recipient_email.casefold(),
        ]
    )
    return _RECEIPT_PREFIX + hashlib.sha256(identity.encode()).hexdigest()


class ModelDeprecationEmailSender:
    def __init__(self, prisma_client: PrismaClient) -> None:
        self.prisma: Final = prisma_client
        self.config: Final = ConfigRepository(prisma_client, use_writer=True)

    async def _impacts(self, model: str, start: date, end: date) -> Mapping[str, _Impact]:
        impacts: Final[dict[str, _Impact]] = {}  # mutable-ok: combines per-recipient counts across bounded usage pages
        offset: int = 0  # rebind-ok: each page advances the stable key grouping cursor
        while True:
            raw = await DailyUserSpendRepository(self.prisma).table.group_by(
                by=["api_key"],
                where={
                    "model": model,
                    "date": {"gte": start.isoformat(), "lte": end.isoformat()},
                    "api_requests": {"gt": 0},
                },
                sum={"api_requests": True},
                order={"api_key": "asc"},
                skip=offset,
                take=_PAGE_SIZE,
            )
            usage = _USED_KEYS.validate_python(raw)
            if not usage:
                return MappingProxyType(impacts)
            keys = await VerificationTokenRepository(self.prisma).find_many_in("token", (row.api_key for row in usage))
            teams = await TeamRepository(self.prisma).find_many_in(
                "team_id", frozenset(key.team_id for key in keys if key.team_id)
            )
            team_by_id = {team.team_id: team for team in teams}
            ids = frozenset(key.user_id for key in keys if key.user_id) | frozenset(
                chain.from_iterable(_admin_ids(team) for team in teams)
            )
            users = await UserRepository(self.prisma).find_many_in("user_id", ids)
            addresses = {
                user.user_id: user.user_email.strip() for user in users if user.user_email and user.user_email.strip()
            }
            counts = {row.api_key: row.metrics.api_requests for row in usage}
            for key in keys:
                team = team_by_id.get(key.team_id) if key.team_id else None
                if key.token is None:
                    continue
                recipients = _recipient_emails(key.user_id, team, addresses)
                for address in recipients:
                    previous = impacts.get(address, _Impact())
                    names: frozenset[str] = frozenset((team.team_alias or team.team_id,)) if team else frozenset()
                    impacts[address] = _Impact(
                        key_count=previous.key_count + 1,
                        requests=previous.requests + counts[key.token],
                        teams=previous.teams | names,
                    )
            if len(usage) < _PAGE_SIZE:
                return MappingProxyType(impacts)
            offset += _PAGE_SIZE

    async def _deliver(self, email: ModelDeprecationEmail, now: datetime, base_url: str) -> bool:
        from litellm.proxy.utils import send_email

        key: Final = _receipt_key(email)
        record: Final = await self.config.get_param(key)
        previous: Final = _Receipt.model_validate(record.param_value) if record else None
        if previous and (previous.status is _DeliveryStatus.SENT or previous.lease_until > now):
            return False
        revision: Final = previous.revision if previous else 0
        claim: Final = _Receipt(revision=revision + 1, lease_until=now + timedelta(minutes=5))
        if not await self.config.set_param_if_revision(key, claim.model_dump(mode="json"), revision):
            return False
        content: Final = render_model_deprecation_email(email, base_url=base_url)
        if not await send_email(
            receiver_email=email.recipient_email,
            subject=content.subject,
            html=content.html,
            text=content.text,
            message_id=f"<litellm-deprecation-{key.removeprefix(_RECEIPT_PREFIX)}@alerts.litellm.ai>",
        ):
            return False
        sent: Final = claim.model_copy(update={"revision": claim.revision + 1, "status": _DeliveryStatus.SENT})
        if not await self.config.set_param_if_revision(key, sent.model_dump(mode="json"), claim.revision):
            raise RuntimeError("Could not record the delivered model deprecation reminder")
        return True

    async def run(self, llm_router: Router, *, now: datetime | None = None) -> int:
        if not os.getenv("SMTP_HOST") or not os.getenv("SMTP_SENDER_EMAIL"):
            return 0
        if not await self.config.model_deprecations_enabled():
            return 0
        current: Final = now or datetime.now(timezone.utc)
        settings: Final = await self.config.get_param("general_settings")
        if not _OptIn.model_validate(settings.param_value if settings else {}).enabled:
            return 0
        base_url: Final = os.getenv("PROXY_BASE_URL", "")
        if not base_url:
            return 0
        await self.config.delete_params_older_than(_RECEIPT_PREFIX, current - timedelta(days=60))
        snapshot: Final = collect_model_deprecations(llm_router=llm_router, today=current.date())
        start: Final = current.date() - timedelta(days=29)
        seen: Final[set[tuple[str, date]]] = set()  # mutable-ok: deduplicates aliases in this pass
        sent: int = 0  # rebind-ok: counts successfully delivered reminders in this pass
        for model in snapshot.imminent:
            if model.days_until_deprecation not in tuple(DeprecationEmailMilestone):
                continue
            name = _usage_model(model)
            identity = (name, model.deprecation_date)
            if identity in seen:
                continue
            seen.add(identity)
            impacts = await self._impacts(name, start, current.date())
            for address, impact in sorted(impacts.items()):
                email = ModelDeprecationEmail(
                    model=model,
                    usage_model=name,
                    recipient_email=address,
                    milestone=DeprecationEmailMilestone(model.days_until_deprecation),
                    key_count=impact.key_count,
                    requests=impact.requests,
                    team_names=tuple(sorted(impact.teams)),
                    start_date=start,
                    end_date=current.date(),
                )
                if await self._deliver(email, now or datetime.now(timezone.utc), base_url):
                    sent += 1
        return sent


async def run_scheduled_model_deprecation_emails() -> None:
    """Poll the email settings independently; a large owner mailing never delays Slack checks."""
    from litellm._logging import verbose_proxy_logger

    while True:
        try:
            from litellm.proxy.proxy_server import llm_router, prisma_client

            if llm_router is not None and prisma_client is not None:
                await ModelDeprecationEmailSender(prisma_client).run(llm_router)
        except Exception:  # noqa: BLE001  # transient SMTP/DB failures must not terminate the reminder loop
            verbose_proxy_logger.exception("Error sending model deprecation email reminders")
        await asyncio.sleep(DEPRECATION_IDLE_POLL_SECONDS)
