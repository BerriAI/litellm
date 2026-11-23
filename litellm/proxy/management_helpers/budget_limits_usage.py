import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Final, Literal, TypeAlias

from litellm.proxy.spend_tracking.budget_reservation import get_budget_window_start

BudgetWindowEntityType: TypeAlias = Literal["Key", "Team", "User"]


def budget_limit_windows(budget_limits: Sequence[object] | str | None) -> tuple[Mapping[str, object], ...]:
    """Coerce stored `budget_limits` into plain window dicts.

    The DB Json column comes back as a parsed list off `find_unique`, as text off `json.dumps`'d
    values or raw SQL rows, and as `BudgetLimitEntry` objects once a pydantic model validated it.
    """
    if not budget_limits:
        return ()
    raw_windows: Final = json.loads(budget_limits) if isinstance(budget_limits, str) else budget_limits
    return tuple(raw_window if isinstance(raw_window, dict) else raw_window.model_dump() for raw_window in raw_windows)


def _window_max_budget(window: Mapping[str, object]) -> float | None:
    value: Final = window.get("max_budget")
    if not isinstance(value, (int, float, str)):
        return None
    try:
        return float(value)
    except ValueError:
        return None


async def _budget_window_usage(
    window: Mapping[str, object],
    counter_key: Callable[[str], str],
    entity_type: BudgetWindowEntityType,
    entity_id: str,
) -> tuple[str, Mapping[str, object]] | None:
    """Reads the same cross-pod counter, window entity and window start the budget check enforces against,
    so a stale-low counter is re-checked against the LiteLLM_BudgetWindowSpend row."""
    from litellm.proxy.proxy_server import get_current_spend

    duration: Final = window.get("budget_duration")
    if not isinstance(duration, str) or not duration:
        return None
    spend: Final = await get_current_spend(
        counter_key=counter_key(duration),
        fallback_spend=0.0,
        max_budget=_window_max_budget(window),
        window_entity_type=entity_type,
        window_entity_id=entity_id,
        window_duration=duration,
        window_start=get_budget_window_start(window),
    )
    return duration, {"current_spend": spend}


async def build_budget_limits_usage(
    budget_limits: Sequence[object] | str | None,
    counter_key: Callable[[str], str],
    entity_type: BudgetWindowEntityType,
    entity_id: str,
) -> Mapping[str, Mapping[str, object]] | None:
    """Current-window spend per budget window, keyed by budget_duration. None when there are no windows.

    Plain dicts, since /user/info and /team/info nest this inside dicts that pydantic serializes."""
    windows: Final = budget_limit_windows(budget_limits)
    if not windows:
        return None
    usages: Final = await asyncio.gather(
        *(
            _budget_window_usage(window=window, counter_key=counter_key, entity_type=entity_type, entity_id=entity_id)
            for window in windows
        )
    )
    return {duration: usage for duration, usage in (u for u in usages if u is not None)}
