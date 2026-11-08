from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final

from pydantic import TypeAdapter

from litellm._auth.common_budgets import BudgetFacts, WindowFacts, window_start
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.proxy._types import Litellm_EntityType, LiteLLM_TeamTable, UserAPIKeyAuth
from litellm.proxy.auth.common_budget_adapter import CounterRead, raise_budget, read_budget


@dataclass(frozen=True, slots=True)
class BudgetWindow:
    budget_duration: str
    max_budget: float
    reset_at: object = None


_WINDOW: Final = TypeAdapter(BudgetWindow)


def _window(raw: object) -> BudgetWindow:
    return _WINDOW.validate_python(raw, from_attributes=True)


def _seconds(duration: str) -> int | None:
    try:
        return duration_in_seconds(duration)
    except Exception:  # noqa: BLE001  # preserve invalid-duration skip behavior
        return None


def _reset(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    if isinstance(value, str):
        try:
            parsed: Final = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return _reset(parsed)
    return None


async def check_windows(windows: Sequence[object], *, entity: str, identity: str | None) -> None:
    for raw in windows:
        await check_window(raw, entity=entity, identity=identity)


async def check_window(raw: object, *, entity: str, identity: str | None) -> None:
    window: Final = _window(raw)
    seconds: Final = _seconds(window.budget_duration)
    reset: Final = _reset(window.reset_at)
    start_timestamp: Final = window_start(
        WindowFacts(seconds, reset.timestamp() if reset is not None else None, datetime.now(timezone.utc).timestamp())
    )
    start: Final = datetime.fromtimestamp(start_timestamp, timezone.utc) if start_timestamp is not None else None
    facts, result = await read_budget(
        BudgetFacts("KeyWindow" if entity == "Key" else "TeamWindow", window.max_budget),
        CounterRead(
            f"spend:{entity.lower()}:{identity}:window:{window.budget_duration}",
            0.0,
            window.max_budget,
            window_entity_type=entity,
            window_entity_id=identity,
            window_duration=window.budget_duration,
            window_start=start,
        ),
    )
    if result.denied and facts.spend is not None:
        label: Final = "Key" if entity == "Key" else f"Team={identity}"
        raise_budget(
            facts,
            Litellm_EntityType.KEY if entity == "Key" else Litellm_EntityType.TEAM,
            identity,
            f"ExceededBudget: {label} over {window.budget_duration} budget. Spend=${facts.spend:.4f}, Limit=${window.max_budget:.2f}",
        )


async def virtual_key_multi_budget_check(valid_token: UserAPIKeyAuth) -> None:
    await check_windows(
        TypeAdapter(tuple[object, ...]).validate_python(valid_token.budget_limits or ()),  # pyright: ignore[reportUnknownMemberType]  # legacy input
        entity="Key",
        identity=valid_token.token,
    )


async def team_multi_budget_check(team_object: LiteLLM_TeamTable | None) -> None:
    if team_object is not None:
        await check_windows(
            TypeAdapter(tuple[object, ...]).validate_python(team_object.budget_limits or ()),  # pyright: ignore[reportUnknownMemberType]  # validate legacy window dictionaries
            entity="Team",
            identity=team_object.team_id,
        )
