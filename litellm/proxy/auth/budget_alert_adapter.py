import asyncio
from collections.abc import Mapping
from dataclasses import replace
from typing import Final

from pydantic import TypeAdapter, ValidationError

import litellm
from litellm._auth import budget_alerts
from litellm._auth.budget_alerts import ConfigFacts, Decision, Facts, Failure, ThresholdFacts
from litellm._logging import verbose_proxy_logger
from litellm.constants import EMAIL_BUDGET_ALERT_MAX_SPEND_ALERT_PERCENTAGE
from litellm.proxy._types import CallInfo, Litellm_EntityType, LiteLLM_UserTable, UserAPIKeyAuth
from litellm.proxy.auth.common_budget_adapter import metadata
from litellm.proxy.utils import ProxyLogging

_CONFIG: Final = TypeAdapter(Mapping[str, object])
_OBJECT: Final = TypeAdapter(object)
_ITEMS: Final = TypeAdapter(tuple[object, ...])


def _threshold(key: str, raw: object) -> ThresholdFacts:
    text: Final = raw if isinstance(raw, str) else None
    items: Final = (
        tuple(item if isinstance(item, str) else None for item in _ITEMS.validate_python(raw))
        if isinstance(raw, list)
        else None
    )
    initial: Final = ThresholdFacts(key=key, digit=key.isdigit(), text=text, items=items)
    try:
        integer: Final = int(key)
    except ValueError as error:
        return replace(initial, integer_failure=Failure("ValueError", str(error)))
    parsed: Final = replace(initial, digits=str(integer))
    try:
        return replace(parsed, ratio=integer / 100.0)
    except OverflowError as error:
        return replace(parsed, ratio_failure=Failure("OverflowError", str(error)))


def _config(raw: object) -> tuple[tuple[ThresholdFacts, ...], Failure | None]:
    if not raw:
        return (), None
    try:
        value: Final = _CONFIG.validate_python(raw)
    except ValidationError:
        return (), Failure("AttributeError", f"'{type(raw).__name__}' object has no attribute 'items'")
    return tuple(_threshold(key, value) for key, value in value.items()), None


def read_alert_facts(facts: Facts, global_config: object, local_config: object) -> Facts:
    decision: Final = budget_alerts.plan(facts)
    if not decision.read_config:
        return facts
    global_entries, global_failure = _config(global_config)
    local_entries, local_failure = _config(local_config)
    loaded: Final = replace(facts, config=ConfigFacts(global_entries, local_entries, global_failure, local_failure))
    order: Final = budget_alerts.plan(loaded)
    if not order.read_order:
        return loaded
    assert loaded.config is not None
    return replace(loaded, config=replace(loaded.config, order=tuple(set(order.global_keys) | set(order.local_keys))))


def read_key_budget_alert_facts(valid_token: UserAPIKeyAuth) -> Facts:
    facts: Final = Facts(
        False, valid_token.spend, valid_token.max_budget, EMAIL_BUDGET_ALERT_MAX_SPEND_ALERT_PERCENTAGE
    )
    if not budget_alerts.plan(facts).read_config:
        return facts
    return read_alert_facts(
        facts,
        _OBJECT.validate_python(litellm.default_key_max_budget_alert_emails),  # pyright: ignore[reportUnknownMemberType]  # validate the legacy global config snapshot
        metadata(valid_token.metadata).get("max_budget_alert_emails"),  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]  # legacy metadata is validated at the boundary
    )


def read_member_budget_alert_facts(
    spend: float,
    max_budget: float,
    team_metadata: Mapping[str, object] | None,
) -> Facts:
    facts: Final = Facts(True, spend, max_budget, EMAIL_BUDGET_ALERT_MAX_SPEND_ALERT_PERCENTAGE)
    return read_alert_facts(
        facts, None, team_metadata.get("team_member_max_budget_alert_emails") if team_metadata else None
    )


def raise_alert_failure(decision: Decision) -> None:
    failure: Final = decision.failure
    if failure is None:
        return
    match failure.kind:
        case "AttributeError":
            raise AttributeError(failure.message)
        case "OverflowError":
            raise OverflowError(failure.message)
        case _:
            raise ValueError(failure.message)


def schedule_budget_alert(decision: Decision, call_info: CallInfo, proxy_logging_obj: ProxyLogging) -> None:
    raise_alert_failure(decision)
    if decision.emit:
        projected: Final = (
            call_info
            if decision.thresholds is None
            else call_info.model_copy(update={"max_budget_alert_emails": alert_emails(decision)})
        )
        asyncio.create_task(proxy_logging_obj.budget_alerts(type="max_budget_alert", user_info=projected))


def alert_emails(
    decision: Decision,
) -> dict[str, list[str]] | None:  # mutable-ok: CallInfo requires the legacy dict of lists projection
    return None if decision.thresholds is None else {entry.key: list(entry.recipients) for entry in decision.thresholds}


async def virtual_key_max_budget_alert_check(
    valid_token: UserAPIKeyAuth,
    proxy_logging_obj: ProxyLogging,
    user_obj: LiteLLM_UserTable | None = None,
) -> None:
    decision: Final = budget_alerts.plan(read_key_budget_alert_facts(valid_token))
    raise_alert_failure(decision)
    if not decision.emit:
        return
    if decision.thresholds is None:
        assert valid_token.max_budget is not None
        verbose_proxy_logger.debug(
            "Reached Max Budget Alert Threshold for token %s, spend %s, max_budget %s, alert_threshold %s",
            valid_token.token,
            valid_token.spend,
            valid_token.max_budget,
            valid_token.max_budget * EMAIL_BUDGET_ALERT_MAX_SPEND_ALERT_PERCENTAGE,
        )
    call_info: Final = CallInfo(
        token=valid_token.token,
        spend=valid_token.spend,
        max_budget=valid_token.max_budget,
        soft_budget=valid_token.soft_budget,
        user_id=valid_token.user_id,
        team_id=valid_token.team_id,
        team_alias=valid_token.team_alias,
        organization_id=valid_token.org_id,
        user_email=user_obj.user_email if user_obj else None,
        key_alias=valid_token.key_alias,
        event_group=Litellm_EntityType.KEY,
    )
    schedule_budget_alert(decision, call_info, proxy_logging_obj)


def team_member_max_budget_alert_check(
    team_id: str,
    team_alias: str | None,
    team_metadata: Mapping[str, object] | None,
    organization_id: str | None,
    user_id: str,
    user_email: str | None,
    proxy_logging_obj: ProxyLogging,
    spend: float,
    max_budget: float,
) -> None:
    decision: Final = budget_alerts.plan(read_member_budget_alert_facts(spend, max_budget, team_metadata))
    raise_alert_failure(decision)
    if not decision.emit:
        return
    call_info: Final = CallInfo(
        spend=spend,
        max_budget=max_budget,
        user_id=user_id,
        team_id=team_id,
        team_alias=team_alias,
        organization_id=organization_id,
        user_email=user_email,
        event_group=Litellm_EntityType.TEAM_MEMBER,
    )
    schedule_budget_alert(decision, call_info, proxy_logging_obj)
