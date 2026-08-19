"""Canonical cross-pod spend counter keys.

Enforcement, budget reservation and budget introspection must read the exact same
string per scope or they silently observe different counters, so the format lives
here once instead of as an f-string per call site.
"""


def key_spend_counter(token: str | None) -> str:
    return f"spend:key:{token}"


def key_window_spend_counter(token: str | None, budget_duration: str) -> str:
    return f"{key_spend_counter(token)}:window:{budget_duration}"


def team_spend_counter(team_id: str) -> str:
    return f"spend:team:{team_id}"


def team_window_spend_counter(team_id: str, budget_duration: str) -> str:
    return f"{team_spend_counter(team_id)}:window:{budget_duration}"


def team_member_spend_counter(user_id: str, team_id: str) -> str:
    return f"spend:team_member:{user_id}:{team_id}"


def user_spend_counter(user_id: str) -> str:
    return f"spend:user:{user_id}"


def org_spend_counter(org_id: str) -> str:
    return f"spend:org:{org_id}"


def tag_spend_counter(tag_name: str) -> str:
    return f"spend:tag:{tag_name}"


def end_user_spend_counter(end_user_id: str) -> str:
    return f"spend:end_user:{end_user_id}"
