import math
from datetime import datetime, timezone
from typing import Final
from unittest.mock import Mock

import pytest

import litellm
from litellm._auth import budget_context as core
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth import key_budget_context_adapter as adapter
from tests.unit.proxy.auth import phase1_budget_context_reference as legacy
from tests.unit.proxy.auth import phase1_budget_throttle_reference as budget_throttle

_NOW: Final = datetime(2025, 1, 1, tzinfo=timezone.utc)


@pytest.mark.parametrize("native", (False, True), ids=("reference", "native"))
@pytest.mark.parametrize(
    "increase", (None, False, True, 0, -3, 3, float("nan"), float("inf"), "", "3", [], [3], {}, {"n": 3})
)
@pytest.mark.parametrize(
    "expiry", ("2024-12-31", "2025-01-01T00:00:00+00:00", "2025-01-01T00:00:00.000001Z", "2026-01-01T09:00:00+09:00")
)
@pytest.mark.parametrize("limit", (None, 0, 10))
def test_temporary_budget_reference_native(
    native: bool, monkeypatch: pytest.MonkeyPatch, increase: object, expiry: str, limit: float | None
) -> None:
    now: Final = Mock(return_value=_NOW)
    clock: Final = Mock(wraps=datetime, now=now)
    monkeypatch.setattr(adapter, "datetime", clock)
    monkeypatch.setattr(legacy, "datetime", clock)
    token: Final = UserAPIKeyAuth(
        max_budget=limit, metadata={"temp_budget_increase": increase, "temp_budget_expiry": expiry}
    )
    fn: Final = (
        adapter.update_key_budget_with_temp_budget_increase
        if native
        else legacy._update_key_budget_with_temp_budget_increase  # pyright: ignore[reportPrivateUsage]  # Phase 1 leaf is the unchanged reference implementation
    )
    active: Final = expiry.startswith("2026") or expiry == "2025-01-01T00:00:00.000001Z"
    changes: Final = limit is not None and active and bool(increase)
    if changes and not isinstance(increase, (int, float)):
        with pytest.raises(TypeError, match="unsupported operand type"):
            fn(token)
        return
    result: Final = fn(token)
    assert (result is token) is not changes
    if changes:
        assert limit is not None and isinstance(increase, (int, float))
        expected: Final = limit + increase
        assert result.max_budget is not None
        assert result.max_budget == expected or (math.isnan(result.max_budget) and math.isnan(expected))
    assert token.max_budget == limit
    assert now.call_count == (0 if limit is None else 1)


@pytest.mark.parametrize("native", (False, True), ids=("reference", "native"))
@pytest.mark.parametrize("expiry", (None, 5, "", "invalid"))
@pytest.mark.parametrize("increase_present", (False, True))
@pytest.mark.parametrize("limit", (None, 1))
def test_temporary_metadata_error_timing(
    native: bool, monkeypatch: pytest.MonkeyPatch, expiry: object, increase_present: bool, limit: float | None
) -> None:
    token: Final = UserAPIKeyAuth(
        max_budget=limit,
        metadata={"temp_budget_expiry": expiry, **({"temp_budget_increase": 0} if increase_present else {})},
    )
    now: Final = Mock(return_value=_NOW)
    clock: Final = Mock(wraps=datetime, now=now)
    monkeypatch.setattr(adapter, "datetime", clock)
    monkeypatch.setattr(legacy, "datetime", clock)
    fn: Final = (
        adapter.update_key_budget_with_temp_budget_increase
        if native
        else legacy._update_key_budget_with_temp_budget_increase  # pyright: ignore[reportPrivateUsage]  # Phase 1 leaf is the unchanged reference implementation
    )
    if limit is not None and increase_present:
        with pytest.raises(ValueError if isinstance(expiry, str) else TypeError):
            fn(token)
    else:
        assert fn(token) is token
    now.assert_not_called()


@pytest.mark.parametrize("pct", (None, False, True, "", "0.5", 0, -0.5, 0.25, 1, 2, float("nan"), float("inf")))
@pytest.mark.parametrize("opt_in", (None, False, True, 1, "true"))
@pytest.mark.parametrize("limit", (None, 0, 1))
def test_throttle_reference_native(
    monkeypatch: pytest.MonkeyPatch, pct: object, opt_in: object, limit: int | None
) -> None:
    monkeypatch.setattr(litellm, "budget_exceeded_throttle_percentage", pct)
    token: Final = UserAPIKeyAuth(metadata={"throttle_on_budget_exceeded": opt_in}, tpm_limit=limit)
    assert adapter.budget_throttle_percentage() == budget_throttle.budget_throttle_percentage()
    assert adapter.should_throttle_budget_exceeded(token) == budget_throttle.should_throttle_budget_exceeded(token)
    expected: Final = (
        budget_throttle.budget_throttle_percentage() if budget_throttle.should_throttle_budget_exceeded(token) else None
    )
    assert adapter.inspect_key_budget_throttle(token) == expected


@pytest.mark.parametrize("tpm", (None, 0, 1))
@pytest.mark.parametrize("rpm", (None, 0, 1))
@pytest.mark.parametrize("opt_in", (False, True))
def test_throttle_native_selects_config_read(tpm: int | None, rpm: int | None, opt_in: bool) -> None:
    assert core.throttle(core.ThrottleFacts(opt_in, tpm, rpm)).read_percentage is (
        opt_in and (tpm is not None or rpm is not None)
    )
