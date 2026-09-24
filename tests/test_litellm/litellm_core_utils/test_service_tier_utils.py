"""Requested/served service tier resolution for telemetry labels."""

from typing import cast

import pytest

from litellm.litellm_core_utils.service_tier_utils import get_requested_service_tier
from litellm.types.utils import StandardLoggingPayload


def _payload(tier: object) -> StandardLoggingPayload:
    return cast(StandardLoggingPayload, {"model_parameters": {"service_tier": tier}})


@pytest.mark.parametrize("tier", ["balanced", "flex", "priority", "auto", "default"])
def test_get_requested_service_tier_returns_known_tiers(tier: str):
    assert get_requested_service_tier(_payload(tier)) == tier


@pytest.mark.parametrize("tier", ["some_unknown_tier", 42, None])
def test_get_requested_service_tier_drops_unrecognized_values(tier: object):
    assert get_requested_service_tier(_payload(tier)) is None
