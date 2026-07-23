from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth


class SpendIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str | None = None
    team_id: str | None = None
    org_id: str | None = None


@dataclass(frozen=True, slots=True)
class FieldDivergence:
    field: str
    resolved_value: str | None
    consumed_value: str | None


def resolved_identity_from_key(key: UserAPIKeyAuth) -> SpendIdentity:
    return SpendIdentity(
        user_id=key.user_id,
        team_id=key.team_id,
        org_id=key.org_id,
    )


def spend_identity_divergence(
    resolved: SpendIdentity, consumed: SpendIdentity
) -> tuple[FieldDivergence, ...]:
    pairs = (
        ("user_id", resolved.user_id, consumed.user_id),
        ("team_id", resolved.team_id, consumed.team_id),
        ("org_id", resolved.org_id, consumed.org_id),
    )
    return tuple(
        FieldDivergence(
            field=field, resolved_value=resolved_value, consumed_value=consumed_value
        )
        for field, resolved_value, consumed_value in pairs
        if resolved_value != consumed_value
    )


def log_identity_divergence(
    resolved_key: UserAPIKeyAuth, consumed: SpendIdentity
) -> None:
    divergences = spend_identity_divergence(
        resolved_identity_from_key(resolved_key), consumed
    )
    if not divergences:
        return
    fields = ", ".join(
        f"{d.field} (resolved={d.resolved_value!r}, consumed={d.consumed_value!r})"
        for d in divergences
    )
    verbose_proxy_logger.warning(
        "Spend attribution metadata diverges from the resolved key identity "
        "[credential_ref=%s]: %s",
        resolved_key.token,
        fields,
    )
