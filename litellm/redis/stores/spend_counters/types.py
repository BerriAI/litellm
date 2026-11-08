from __future__ import annotations

from typing import Final, TypeAlias

from litellm.rust_bridge._native import NativeRedisStoreError

SPEND_COUNTERS_TARGET: Final = "spend_counters"
SPEND_COUNTER_PREFIX: Final = "spend:"
END_USER_COUNTER_PREFIX: Final = "spend:end_user:"

SpendIncrement: TypeAlias = tuple[str, float]
SpendReading: TypeAlias = tuple[float | None, bool]
"""(value, authoritative): authoritative when Redis answered, a clean miss included."""


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


def is_unavailable(error: BaseException) -> bool:
    """Redis timed out or its breaker is open, the failures a post-call increment drops."""
    return isinstance(error, NativeRedisStoreError) and error.args[:1] in (("timeout",), ("breaker_open",))
