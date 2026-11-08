import asyncio
from typing import Final
from unittest.mock import AsyncMock, Mock

import pytest

import litellm
from litellm._auth import budget_alerts
from litellm._auth.budget_alerts import Facts
from litellm._auth.failure import NativeAuthFailure, NativeAuthValueError
from litellm.constants import EMAIL_BUDGET_ALERT_MAX_SPEND_ALERT_PERCENTAGE
from litellm.proxy._types import CallInfo, Litellm_EntityType, UserAPIKeyAuth
from litellm.proxy.auth import budget_alert_adapter as native
from litellm.rust_bridge import _native
from tests.unit.proxy.auth.budget_alert_reference import reference_alert


def observed_reference(facts: Facts, global_config: object, local_config: object) -> object:
    try:
        return reference_alert(facts, global_config, local_config)
    except (ValueError, OverflowError, AttributeError) as error:
        return type(error).__name__, str(error)


def observed_native(facts: Facts, global_config: object, local_config: object) -> object:
    try:
        decision: Final = budget_alerts.plan(native.read_alert_facts(facts, global_config, local_config))
        native.raise_alert_failure(decision)
        config: Final = native.alert_emails(decision) if decision.emit else None
        return decision.emit, None if config is None else tuple((key, tuple(value)) for key, value in config.items())
    except (ValueError, OverflowError, AttributeError) as error:
        return type(error).__name__, str(error)


@pytest.mark.parametrize("member", (False, True))
@pytest.mark.parametrize("spend", (-1.0, 0.0, 7.9, 8.0, 9.0, 10.0, float("nan"), float("inf")))
@pytest.mark.parametrize("limit", (-1.0, 0.0, 10.0, float("nan"), float("inf")))
@pytest.mark.parametrize(
    "configs",
    (
        (None, None),
        ({"80": " a@x.test,b@x.test "}, {"80": [" b@x.test ", "c@x.test", "", None]}),
        (None, {"bad": "a@x.test"}),
        (None, {"0": [], "101": None, "080": [" a@x.test "]}),
        (None, {"80": ("a@x.test",)}),
        (None, {"٨٠": "a@x.test"}),
        (None, {"²": []}),
        (None, {"9" * 400: [], "80": "a@x.test"}),
        (None, {"9" * 400: []}),
        (None, "invalid"),
    ),
)
def test_native_threshold_merge_and_trigger_match_reference(
    member: bool, spend: float, limit: float, configs: tuple[object, object]
) -> None:
    facts: Final = Facts(member, spend, limit, EMAIL_BUDGET_ALERT_MAX_SPEND_ALERT_PERCENTAGE)
    global_config, local_config = configs
    global_value: Final = None if member else global_config
    assert observed_native(facts, global_value, local_config) == observed_reference(facts, global_value, local_config)


@pytest.mark.asyncio
@pytest.mark.parametrize("config", (None, {"80": " a@x.test, b@x.test "}))
async def test_key_alert_schedules_once_without_awaiting_and_preserves_projection(
    monkeypatch: pytest.MonkeyPatch, config: object
) -> None:
    monkeypatch.setattr(litellm, "default_key_max_budget_alert_emails", None)
    key: Final = UserAPIKeyAuth(
        token="token",
        spend=8,
        max_budget=10,
        metadata={"max_budget_alert_emails": config},
        user_id="owner",
        team_id="team",
    )
    logger: Final = Mock(budget_alerts=AsyncMock())
    await native.virtual_key_max_budget_alert_check(key, logger)
    assert logger.budget_alerts.call_count == 1
    assert logger.budget_alerts.await_count == 0
    call_info: Final = logger.budget_alerts.call_args.kwargs["user_info"]
    expected: Final = CallInfo(
        token="token",
        spend=8,
        max_budget=10,
        soft_budget=None,
        user_id="owner",
        team_id="team",
        team_alias=None,
        organization_id=None,
        user_email=None,
        key_alias=None,
        event_group=Litellm_EntityType.KEY,
    )
    projected: Final = (
        expected
        if config is None
        else expected.model_copy(update={"max_budget_alert_emails": {"80": ["a@x.test", "b@x.test"]}})
    )
    assert call_info.model_dump() == projected.model_dump()
    assert call_info.model_fields_set == projected.model_fields_set
    await asyncio.sleep(0)
    assert logger.budget_alerts.await_count == 1


def test_native_failure_is_terminal_before_alert_config_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    failure: Final = ValueError("native alert decoder failed")
    backend: Final = Mock(side_effect=failure)
    monkeypatch.setattr(_native, "request_auth_budget_alert_plan", backend)
    with pytest.raises(NativeAuthFailure, match="native alert decoder failed") as error:
        native.read_alert_facts(Facts(False, 8.0, 10.0, 0.8), None, {"80": []})
    assert isinstance(error.value, ValueError)
    assert backend.call_count == 1


def test_pricing_native_failure_never_becomes_lookup_failure() -> None:
    from litellm.proxy.auth.budget_pricing_adapter import read_model_price_facts

    failure: Final = NativeAuthValueError("native registry failed")
    registry: Final = Mock(group=Mock(side_effect=failure))
    router: Final = Mock(registry=Mock(return_value=registry))
    with pytest.raises(NativeAuthFailure) as error:
        read_model_price_facts("model", router)
    assert error.value is failure
    registry.group.assert_called_once()
    registry.deployments.assert_not_called()
