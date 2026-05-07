"""Unit tests for ``litellm/managed_agents/types.py`` Pydantic validators.

Covers contract §6.2 invariants on ``SandboxSpec``:
- ``timeout_minutes`` ∈ [1, 1440]
- ``idle_timeout_minutes`` ∈ [1, ``timeout_minutes``]
"""

import pytest
from pydantic import ValidationError

from litellm.managed_agents.types import SandboxSpec


class TestSandboxSpecIdleTimeoutBound:
    """``idle_timeout_minutes`` must not exceed ``timeout_minutes``."""

    def test_idle_below_timeout_is_valid(self) -> None:
        spec = SandboxSpec(type="opencode", timeout_minutes=60, idle_timeout_minutes=10)
        assert spec.idle_timeout_minutes == 10
        assert spec.timeout_minutes == 60

    def test_idle_equals_timeout_is_valid(self) -> None:
        """Boundary: idle == timeout is allowed (closed upper bound)."""
        spec = SandboxSpec(type="opencode", timeout_minutes=60, idle_timeout_minutes=60)
        assert spec.idle_timeout_minutes == 60
        assert spec.timeout_minutes == 60

    def test_idle_above_timeout_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            SandboxSpec(type="opencode", timeout_minutes=60, idle_timeout_minutes=120)
        # Surface the offending field in the error message
        assert "idle_timeout_minutes" in str(exc_info.value)

    def test_idle_and_timeout_minimum_boundary(self) -> None:
        """Boundary: both at lower bound ``1`` is valid."""
        spec = SandboxSpec(type="opencode", timeout_minutes=1, idle_timeout_minutes=1)
        assert spec.idle_timeout_minutes == 1
        assert spec.timeout_minutes == 1


class TestSandboxSpecTimeoutBounds:
    """``timeout_minutes`` is bounded to [1, 1440]."""

    def test_timeout_above_upper_bound_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            SandboxSpec(type="opencode", timeout_minutes=1500, idle_timeout_minutes=10)
        assert "timeout_minutes" in str(exc_info.value)

    def test_timeout_at_upper_bound_is_valid(self) -> None:
        spec = SandboxSpec(
            type="opencode",
            timeout_minutes=1440,
            idle_timeout_minutes=1440,
        )
        assert spec.timeout_minutes == 1440
        assert spec.idle_timeout_minutes == 1440

    def test_timeout_below_lower_bound_raises(self) -> None:
        with pytest.raises(ValidationError):
            SandboxSpec(type="opencode", timeout_minutes=0, idle_timeout_minutes=1)

    def test_idle_timeout_below_lower_bound_raises(self) -> None:
        with pytest.raises(ValidationError):
            SandboxSpec(type="opencode", timeout_minutes=10, idle_timeout_minutes=0)
