from typing import Final

import pytest

from litellm.integrations.lens.policy import (
    LENS_LOGGING_POLICY_PARAM,
    LensLoggingPolicy,
    lens_logging_enabled,
    lens_logging_enabled_for_call,
    lens_logging_metadata_error,
)
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth


@pytest.mark.parametrize(
    ("key_value", "team_value", "expected"),
    (
        (None, None, True),
        (None, False, False),
        (None, True, True),
        (False, None, False),
        (True, None, True),
        (False, True, False),
        (True, False, True),
    ),
)
def test_key_override_precedes_team_default(key_value: bool | None, team_value: bool | None, expected: bool) -> None:
    auth: Final = UserAPIKeyAuth(metadata={"lens_enabled": key_value}, team_metadata={"lens_enabled": team_value})
    assert lens_logging_enabled(auth) is expected


@pytest.mark.parametrize("metadata_name", ("metadata", "litellm_metadata"))
def test_caller_labels_cannot_enable_lens_for_a_disabled_authenticated_key(metadata_name: str) -> None:
    auth: Final = UserAPIKeyAuth(metadata={"lens_enabled": False})
    assert (
        lens_logging_enabled_for_call(
            {
                "litellm_params": {
                    metadata_name: {
                        "user_api_key_auth": auth,
                        "lens_enabled": True,
                        "user_api_key_metadata": {"lens_enabled": True},
                    }
                },
                "standard_logging_object": {"metadata": {"lens_enabled": True}},
            }
        )
        is False
    )


@pytest.mark.parametrize("enabled", (False, True))
def test_server_policy_carrier_controls_calls_without_exposing_auth(enabled: bool) -> None:
    assert (
        lens_logging_enabled_for_call(
            {"litellm_params": {LENS_LOGGING_POLICY_PARAM: LensLoggingPolicy(enabled=enabled)}}
        )
        is enabled
    )


@pytest.mark.parametrize("enabled", (False, True))
def test_json_cannot_impersonate_a_trusted_server_policy(enabled: bool) -> None:
    assert lens_logging_enabled_for_call({"litellm_params": {LENS_LOGGING_POLICY_PARAM: {"enabled": enabled}}}) is True


def test_authenticated_key_policy_precedes_the_effective_policy_carrier() -> None:
    assert (
        lens_logging_enabled_for_call(
            {
                "litellm_params": {
                    LENS_LOGGING_POLICY_PARAM: LensLoggingPolicy(enabled=True),
                    "metadata": {"user_api_key_auth": UserAPIKeyAuth(metadata={"lens_enabled": False})},
                }
            }
        )
        is False
    )


@pytest.mark.parametrize(
    "metadata", (None, {}, {"lens_enabled": None}, {"lens_enabled": False}, {"lens_enabled": True})
)
def test_only_boolean_or_inherited_lens_settings_are_accepted(metadata: dict[str, object] | None) -> None:
    assert lens_logging_metadata_error(metadata) is None


@pytest.mark.parametrize("value", ("false", "true", 0, 1, {}, []))
def test_non_boolean_lens_settings_are_rejected_without_coercion(value: object) -> None:
    assert lens_logging_metadata_error({"lens_enabled": value}) == "metadata.lens_enabled must be a boolean or null"
