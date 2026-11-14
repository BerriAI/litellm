from __future__ import annotations

from datetime import datetime, timezone
from typing import Final, Literal, TypeAlias
from urllib.parse import quote, unquote

from litellm.rust_bridge._native import NativeRedisStoreError

SPEND_COUNTERS_TARGET: Final = "spend_counters"
SPEND_COUNTER_PREFIX: Final = "spend:"
END_USER_COUNTER_PREFIX: Final = "spend:end_user:"
COMMITTED_COUNTER_PREFIX: Final = "spend:committed:"

SpendIncrement: TypeAlias = tuple[str, float]
SpendReading: TypeAlias = tuple[float | None, bool]
"""(value, authoritative): authoritative when Redis answered, a clean miss included."""

CustomerCounterScope: TypeAlias = Literal["customer_org", "customer_team", "customer_member"]


def customer_window_counter(
    scope: CustomerCounterScope, entity_id: str, budget_duration: str, window_start: datetime
) -> str:
    """Shared policies have independent balances for each scope, entity and current period."""
    return (
        f"spend:{scope}:{quote(entity_id, safe='')}:window:{budget_duration}"
        f"@{window_start.astimezone(timezone.utc).isoformat()}"
    )


def parse_customer_window_counter(counter_key: str) -> tuple[CustomerCounterScope, str, str, datetime] | None:
    prefix, separator, window = counter_key.rpartition(":window:")
    parts: Final = prefix.split(":", 2)
    if not separator or len(parts) != 3 or parts[0] != "spend":
        return None
    scope: Final = parts[1]
    if scope not in ("customer_org", "customer_team", "customer_member"):
        return None
    duration, at, start = window.rpartition("@")
    if not at or not duration:
        return None
    try:
        parsed: Final = datetime.fromisoformat(start)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return scope, unquote(parts[2]), duration, parsed


def key_counter(token: str) -> str:
    return f"spend:key:{token}"


def key_window_counter(token: str, budget_duration: str) -> str:
    return f"spend:key:{token}:window:{budget_duration}"


def team_counter(team_id: str) -> str:
    return f"spend:team:{team_id}"


def team_window_counter(team_id: str, budget_duration: str) -> str:
    return f"spend:team:{team_id}:window:{budget_duration}"


def team_member_counter(user_id: str, team_id: str) -> str:
    return f"spend:team_member:{user_id}:{team_id}"


def user_counter(user_id: str) -> str:
    return f"spend:user:{user_id}"


def end_user_counter(end_user_id: str) -> str:
    return f"{END_USER_COUNTER_PREFIX}{end_user_id}"


def org_counter(org_id: str) -> str:
    return f"spend:org:{org_id}"


def tag_counter(tag: str) -> str:
    return f"spend:tag:{tag}"


def project_counter(project_id: str) -> str:
    return f"spend:project:{project_id}"


def model_access_group_counter(access_group: str) -> str:
    return f"spend:model_access_group:{access_group}"


def shadow_eval_counter(job_id: str) -> str:
    return f"spend:shadow_eval:{job_id}"


def committed_counter(counter_key: str) -> str:
    """The charges-only twin of a key, team, user, org or end user lifetime counter."""
    return f"{COMMITTED_COUNTER_PREFIX}{counter_key.removeprefix(SPEND_COUNTER_PREFIX)}"


def is_unavailable(error: BaseException) -> bool:
    """Redis timed out or its breaker is open, the failures a post-call increment drops."""
    return isinstance(error, NativeRedisStoreError) and error.args[:1] in (("timeout",), ("breaker_open",))
