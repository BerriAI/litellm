import pytest

from pydantic import ValidationError

from litellm.proxy.telemetry.settings import EnvPolicy, TelemetrySettings, env_policy, load_settings
from litellm.telemetry.consent import OFF, MissingRequirement, TelemetryConsent, UnknownGroup
from litellm.telemetry.records import TelemetryGroup

_HEARTBEAT = TelemetryConsent(frozenset({TelemetryGroup.HEARTBEAT}))
_STORED = TelemetryConsent(frozenset({TelemetryGroup.HEARTBEAT, TelemetryGroup.REQUEST_SUCCESS}))


def _policy(settings: TelemetrySettings) -> EnvPolicy:
    policy = env_policy(settings)
    assert isinstance(policy, EnvPolicy), policy
    return policy


def test_without_env_vars_the_stored_groups_apply_and_nothing_is_reported_as_set() -> None:
    policy = _policy(TelemetrySettings())
    assert policy.effective(_STORED) is _STORED
    assert policy.effective(None) == OFF
    assert policy.set_variables == ()
    assert policy.locked_off is False


def test_the_veto_wins_over_both_pinned_and_stored_groups() -> None:
    policy = _policy(TelemetrySettings(disabled=True, groups="heartbeat"))
    assert policy.effective(_STORED) == OFF
    assert policy.locked_off is True


def test_pinned_groups_override_stored_groups() -> None:
    policy = _policy(TelemetrySettings(groups="Heartbeat, "))
    assert policy.effective(_STORED) == _HEARTBEAT
    assert policy.locked_off is False


def test_an_empty_groups_variable_pins_telemetry_off() -> None:
    policy = _policy(TelemetrySettings(groups=""))
    assert policy.effective(_STORED) == OFF
    assert policy.locked_off is True


@pytest.mark.parametrize(
    ("groups", "error"),
    [
        ("heartbeat,full", UnknownGroup("full")),
        ("token_info", MissingRequirement(TelemetryGroup.TOKEN_INFO, TelemetryGroup.REQUEST_SUCCESS)),
    ],
)
def test_invalid_pinned_groups_are_an_error(groups: str, error: object) -> None:
    assert env_policy(TelemetrySettings(groups=groups)) == error


def test_set_variables_lists_every_telemetry_env_var_present_but_never_its_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_TELEMETRY_ENDPOINT", "https://secret-token@telemetry.example")
    monkeypatch.setenv("litellm_telemetry_disabled", "1")
    policy = _policy(TelemetrySettings())
    assert policy.set_variables == ("LITELLM_TELEMETRY_DISABLED", "LITELLM_TELEMETRY_ENDPOINT")
    assert policy.vetoed is True


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LITELLM_TELEMETRY_FLUSH_INTERVAL_SECONDS", "0"),
        ("LITELLM_TELEMETRY_FLUSH_INTERVAL_SECONDS", "-5"),
        ("LITELLM_TELEMETRY_SETTLE_TIMEOUT_SECONDS", "-1"),
    ],
)
def test_nonpositive_intervals_are_rejected_instead_of_flooding_the_receiver(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)
    assert isinstance(load_settings(), ValidationError)
