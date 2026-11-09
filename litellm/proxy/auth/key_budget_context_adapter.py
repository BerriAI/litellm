from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from typing import Final, assert_never

import litellm
from litellm._auth import budget_context as core
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.budget_context_adapter import mapping_value


def numeric_facts(value: object) -> core.NumericFacts:
    return core.NumericFacts(float(value) if isinstance(value, (int, float)) else None, isinstance(value, bool))


def budget_throttle_percentage() -> float | None:
    return core.percentage(numeric_facts(litellm.budget_exceeded_throttle_percentage))


def read_key_throttle_facts(valid_token: UserAPIKeyAuth) -> core.ThrottleFacts:
    metadata: Final = mapping_value(getattr(valid_token, "metadata", None))
    facts: Final = core.ThrottleFacts(
        (metadata or {}).get("throttle_on_budget_exceeded") is True,
        valid_token.tpm_limit,
        valid_token.rpm_limit,
    )
    decision: Final = core.throttle(facts)
    return (
        replace(facts, percentage=numeric_facts(litellm.budget_exceeded_throttle_percentage))
        if decision.read_percentage
        else facts
    )


def should_throttle_budget_exceeded(valid_token: UserAPIKeyAuth) -> bool:
    return core.throttle(read_key_throttle_facts(valid_token)).percentage is not None


def inspect_key_budget_throttle(valid_token: UserAPIKeyAuth) -> float | None:
    decision: Final = core.throttle(read_key_throttle_facts(valid_token))
    return budget_throttle_percentage() if decision.percentage is not None else None


def _microseconds(value: datetime) -> int:
    delta: Final = value - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds


def _temporary(
    token: UserAPIKeyAuth,
    facts: core.TemporaryFacts,
    metadata: Mapping[str, object] | None,
) -> UserAPIKeyAuth:
    decision: Final = core.temporary(facts)
    if decision.invalid_increase_type is not None:
        raise TypeError(
            f"unsupported operand type(s) for +: '{type(token.max_budget).__name__}' and '{decision.invalid_increase_type}'"
        )
    if decision.copy:
        return token.model_copy(update={"max_budget": decision.max_budget})
    match decision.read:
        case None:
            return token
        case "Metadata":
            raw: Final = mapping_value(getattr(token, "metadata", None))
            return _temporary(
                token,
                replace(
                    facts,
                    metadata_read=True,
                    has_increase=raw is not None and "temp_budget_increase" in raw,
                    has_expiry=raw is not None and "temp_budget_expiry" in raw,
                ),
                raw,
            )
        case "Expiry":
            assert metadata is not None
            expiry: Final = datetime.fromisoformat(metadata["temp_budget_expiry"])  # pyright: ignore[reportArgumentType]  # preserve datetime's public TypeError for malformed metadata
            aware: Final = expiry.replace(tzinfo=timezone.utc) if expiry.tzinfo is None else expiry
            return _temporary(token, replace(facts, expires_at=_microseconds(aware)), metadata)
        case "Now":
            return _temporary(token, replace(facts, now=_microseconds(datetime.now(timezone.utc))), metadata)
        case "Increase":
            assert metadata is not None
            value: Final = metadata["temp_budget_increase"]
            raw_increase: Final = core.IncreaseFacts(
                float(value) if isinstance(value, (int, float)) else None, bool(value), type(value).__name__
            )
            return _temporary(token, replace(facts, increase=raw_increase), metadata)
        case _:
            assert_never(decision.read)


def update_key_budget_with_temp_budget_increase(valid_token: UserAPIKeyAuth) -> UserAPIKeyAuth:
    return _temporary(valid_token, core.TemporaryFacts(valid_token.max_budget), None)
