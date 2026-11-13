from collections.abc import Mapping
from datetime import datetime
from itertools import chain
from typing import TYPE_CHECKING, Final, Literal

from pydantic import ConfigDict, TypeAdapter

import litellm
from litellm.litellm_core_utils.duration_parser import get_budget_window_start, get_next_standardized_reset_time
from litellm.models.budget import LiteLLM_BudgetTable
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.db.spend_counter_reseed import SpendCounterReseed
from litellm.proxy.management_helpers.customer_hierarchy import (
    PrismaRecord,
    effective_member_budget,
    related_budget,
    utc_datetime,
)
from litellm.proxy.utils import PrismaClient
from litellm.repositories.prisma_protocols import TableActions
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.proxy.carried_budget_state import PersonBudgetWindow
from litellm.types.proxy.management_endpoints.customer_hierarchy import Budget

if TYPE_CHECKING:
    from prisma import Prisma

_INCLUDE: Final = {"budget": True, "budget_template": True}
_DURATION: Final = TypeAdapter(Literal["1mo", "12mo"])


class PersonBudgetRecord(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    budget_id: str | None = None
    budget_template_id: str | None = None
    budget_override_expires_at: datetime | None = None
    budget: Mapping[str, object] | None = None
    budget_template: Mapping[str, object] | None = None

    def effective_budget(self, now: datetime) -> Budget:
        return effective_member_budget(
            related_budget(self.budget_template),
            related_budget(self.budget),
            self.budget_override_expires_at,
            now,
        )


async def load_person_budget(
    db: "Prisma", subject_type: Literal["user", "customer_member"], subject_id: str
) -> PersonBudgetRecord | None:
    table: Final[TableActions[PrismaRecord]] = (
        db.litellm_usertable if subject_type == "user" else db.litellm_customermembertable
    )
    id_column: Final = "user_id" if subject_type == "user" else "customer_member_id"
    try:
        row: Final = await table.find_unique(where={id_column: subject_id}, include=_INCLUDE)
    except Exception as error:  # noqa: BLE001  # a failed authoritative lookup must deny admission
        raise litellm.ServiceUnavailableError(
            message="Personal budget assignment is unavailable", model="", llm_provider=""
        ) from error
    return PersonBudgetRecord.model_validate(row.model_dump()) if row is not None else None


def person_budget_windows(
    budget: Budget, subject_type: Literal["user", "customer_member"], subject_id: str, now: datetime
) -> tuple[PersonBudgetWindow, ...]:
    increase: Final = LiteLLM_BudgetTable.model_validate(budget.model_dump()).active_temp_budget_increase(
        utc_datetime(now)
    )
    return tuple(
        _person_window(
            subject_type,
            subject_id,
            _DURATION.validate_python(limit.budget_duration),
            limit.max_budget + (increase if budget.temp_budget_duration == limit.budget_duration else 0.0),
            now,
            budget.cap_policy or "pause_until_reset",
        )
        for limit in budget.budget_limits or ()
    )


def _person_window(
    subject_type: Literal["user", "customer_member"],
    subject_id: str,
    duration: Literal["1mo", "12mo"],
    maximum: float,
    now: datetime,
    cap_policy: Literal["pause_allow_increase", "pause_until_reset"],
) -> PersonBudgetWindow:
    reset_at: Final = get_next_standardized_reset_time(duration, current_time=utc_datetime(now), timezone_str="UTC")
    return PersonBudgetWindow(
        entity_type=subject_type,
        entity_id=subject_id,
        budget_duration=duration,
        window_start=get_budget_window_start(duration, reset_at),
        reset_at=reset_at,
        max_budget=maximum,
        cap_policy=cap_policy,
    )


async def prepare_person_budget_windows(
    prisma_client: PrismaClient | None,
    user_id: str | None,
    customer_member_id: str | None,
    route: str,
    now: datetime,
) -> tuple[PersonBudgetWindow, ...]:
    if not RouteChecks.is_llm_api_route(route) or RouteChecks.is_unbilled_route(route):
        return ()
    if prisma_client is None:
        return ()
    subjects: Final[tuple[tuple[Literal["user", "customer_member"], str | None], ...]] = (
        ("user", user_id),
        ("customer_member", customer_member_id),
    )
    records: Final = tuple(
        [
            (subject_type, subject_id, await load_person_budget(prisma_client.db, subject_type, subject_id))
            for subject_type, subject_id in subjects
            if subject_id is not None
        ]
    )
    windows: Final = tuple(
        chain.from_iterable(
            person_budget_windows(record.effective_budget(now), subject_type, subject_id, now)
            for subject_type, subject_id, record in records
            if record is not None
        )
    )
    from litellm.proxy.proxy_server import spend_counter_store

    for window in windows:
        spend: Final = await SpendCounterReseed.coalesced_window(
            prisma_client,
            spend_counter_store,
            window.counter_key,
            window.counter_entity_type,
            window.entity_id,
            window.window_duration,
            window.window_start,
        )
        if spend is None:
            raise litellm.ServiceUnavailableError(
                message="Personal budget spend is unavailable", model="", llm_provider=""
            )
        if spend >= window.max_budget:
            action: Final = (
                "Contact your admin for a temporary budget increase."
                if window.cap_policy == "pause_allow_increase"
                else "Wait until the budget resets."
            )
            raise litellm.BudgetExceededError(
                current_cost=spend,
                max_budget=window.max_budget,
                message=f"{window.entity_type} budget exceeded for {window.budget_duration}. "
                f"Resets at {window.reset_at.isoformat()}. {action}",
                entity_type=window.entity_type,
                entity_id=window.entity_id,
            )
    return windows
