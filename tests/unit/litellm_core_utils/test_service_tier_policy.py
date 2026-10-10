from typing import Final

import pytest

from litellm.exceptions import PermissionDeniedError
from litellm.litellm_core_utils.service_tier_policy import ServiceTierPolicy, apply_service_tier_policy


@pytest.mark.parametrize(
    ("payload", "allowed", "effective"),
    (
        ({}, ("default",), "default"),
        ({"service_tier": "fast"}, ("priority",), "priority"),
        ({"service_tier": "priority"}, ("fast",), "priority"),
        ({"service_tier": "ultrafast", "extra_body": {"service_tier": "default"}}, ("default",), "default"),
    ),
)
def test_allowed_tier_matches_the_outbound_payload(
    payload: dict[str, object], allowed: tuple[str, ...], effective: str
) -> None:
    original: Final = dict(payload)
    result: Final = apply_service_tier_policy(payload, ServiceTierPolicy(allowed_service_tiers=allowed))
    assert result["service_tier"] == effective
    assert payload == original
    if "extra_body" in payload:
        assert result["extra_body"] == {"service_tier": effective}


@pytest.mark.parametrize(
    ("payload", "allowed"),
    (
        ({"service_tier": "ultrafast"}, ("default", "priority")),
        ({"service_tier": "default", "extra_body": {"service_tier": "ultrafast"}}, ("default",)),
        ({"service_tier": "auto"}, ("default",)),
        ({}, ()),
    ),
)
def test_disallowed_tier_returns_permission_denied(
    payload: dict[str, object], allowed: tuple[str, ...]
) -> None:
    with pytest.raises(PermissionDeniedError) as error:
        apply_service_tier_policy(payload, ServiceTierPolicy(allowed_service_tiers=allowed))
    assert error.value.status_code == 403
    assert "service_tier=" in error.value.message


def test_unrestricted_key_preserves_payload_identity() -> None:
    payload: Final[dict[str, object]] = {"service_tier": "future-tier"}
    assert apply_service_tier_policy(payload, ServiceTierPolicy()) is payload
