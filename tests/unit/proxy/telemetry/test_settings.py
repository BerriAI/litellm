import pytest

from pydantic import ValidationError

from litellm.proxy.telemetry.settings import EnvPolicy, TelemetrySettings, describe_errors, env_policy, load_settings
from litellm.telemetry.consent import OFF, TelemetryConsent
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
    ("groups", "message"),
    [
        ("heartbeat,full", "unknown telemetry group 'full'"),
        ("token_info", "telemetry group 'token_info' needs 'request_success' turned on too"),
    ],
)
def test_invalid_pinned_groups_fail_settings_validation(
    monkeypatch: pytest.MonkeyPatch, groups: str, message: str
) -> None:
    monkeypatch.setenv("LITELLM_TELEMETRY_GROUPS", groups)
    error = load_settings()
    assert isinstance(error, ValidationError)
    assert message in describe_errors(error)


@pytest.mark.parametrize("endpoint", ["not a url", "ftp://telemetry.example/v1", "telemetry.example/v1"])
def test_a_non_http_endpoint_fails_settings_validation(monkeypatch: pytest.MonkeyPatch, endpoint: str) -> None:
    monkeypatch.setenv("LITELLM_TELEMETRY_ENDPOINT", endpoint)
    assert isinstance(load_settings(), ValidationError)


def test_validation_errors_name_the_variable_but_never_echo_its_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_TELEMETRY_ENDPOINT", "ftp://secret-token@telemetry.example")
    error = load_settings()
    assert isinstance(error, ValidationError)
    described = describe_errors(error)
    assert "LITELLM_TELEMETRY_ENDPOINT" in described
    assert "secret-token" not in described


def test_the_flush_interval_is_not_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_TELEMETRY_FLUSH_INTERVAL_SECONDS", "1")
    settings = load_settings()
    assert isinstance(settings, TelemetrySettings)
    assert "flush_interval_seconds" not in TelemetrySettings.model_fields


def test_set_variables_lists_every_telemetry_env_var_present_but_never_its_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_TELEMETRY_ENDPOINT", "https://secret-token@telemetry.example")
    monkeypatch.setenv("litellm_telemetry_disabled", "1")
    policy = _policy(TelemetrySettings())
    assert policy.set_variables == ("LITELLM_TELEMETRY_DISABLED", "LITELLM_TELEMETRY_ENDPOINT")
    assert policy.vetoed is True


def test_a_negative_settle_timeout_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_TELEMETRY_SETTLE_TIMEOUT_SECONDS", "-1")
    assert isinstance(load_settings(), ValidationError)
