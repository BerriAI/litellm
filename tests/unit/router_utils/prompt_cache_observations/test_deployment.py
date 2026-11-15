import copy
from collections.abc import Mapping

import pytest

from litellm.router_utils.prompt_cache_observations.deployment import (
    DeploymentCacheSettings,
    SelectedDeployment,
    deployment_cache_identity,
    deployment_cache_settings,
    effective_destination,
    merged_cache_settings,
    selected_deployment,
    served_model,
)

_SELECTED = SelectedDeployment(deployment_id="deployment-1", base_model=None)
_DESTINATION = "https://api.anthropic.com/v1/messages"
_POINTS = [{"location": "message", "role": "system"}]
_ATTEMPT: dict[str, object] = {
    "model": "anthropic/claude-test",
    "model_info": {"id": "deployment-1", "prompt_cache_observation": True},
}


def _settings(attempt: Mapping[str, object] = _ATTEMPT) -> DeploymentCacheSettings:
    settings = deployment_cache_settings(attempt)
    assert settings is not None
    return settings


def _fingerprint(
    destination: str | None = _DESTINATION,
    model: str = "claude-test",
    selected: SelectedDeployment = _SELECTED,
    attempt: Mapping[str, object] = _ATTEMPT,
) -> str:
    return deployment_cache_identity(
        selected=selected,
        provider="anthropic",
        provider_model=model,
        destination=destination,
        settings=_settings(attempt),
    ).config_fingerprint


def test_fingerprint_needs_no_secret_so_every_worker_derives_the_same_identity() -> None:
    assert _fingerprint() == _fingerprint()


def test_attempt_settings_unrelated_to_caching_keep_the_same_fingerprint() -> None:
    noisy = {**_ATTEMPT, "api_key": "rotated", "timeout": 5, "rpm": 99, "metadata": {"tags": ["blue"]}}
    assert _fingerprint(attempt=noisy) == _fingerprint()


def test_gateway_cache_injection_changes_the_fingerprint() -> None:
    assert _fingerprint(attempt={**_ATTEMPT, "cache_control_injection_points": _POINTS}) != _fingerprint()


def test_settings_are_only_captured_for_an_opted_in_deployment() -> None:
    assert deployment_cache_settings({**_ATTEMPT, "model_info": {"id": "deployment-1"}}) is None
    assert _settings().selected == _SELECTED


def test_destination_drops_the_query_string_and_a_trailing_slash() -> None:
    assert effective_destination(f"{_DESTINATION}/?api-key=secret") == _DESTINATION
    assert effective_destination(None) is None


@pytest.mark.parametrize(
    ("destination", "model", "selected"),
    [
        ("https://gateway.example/anthropic/v1/messages", "claude-test", _SELECTED),
        (_DESTINATION, "claude-test-next", _SELECTED),
        (_DESTINATION, "claude-test", SelectedDeployment(deployment_id="deployment-1", base_model="claude-base")),
    ],
)
def test_destination_model_and_base_model_change_the_fingerprint(
    destination: str, model: str, selected: SelectedDeployment
) -> None:
    assert _fingerprint(destination, model, selected) != _fingerprint()


@pytest.mark.parametrize(
    ("model_info", "expected"),
    [
        ({"id": "d1", "prompt_cache_observation": True}, SelectedDeployment(deployment_id="d1", base_model=None)),
        (
            {"id": "d1", "prompt_cache_observation": True, "base_model": "claude-test", "rpm": 10},
            SelectedDeployment(deployment_id="d1", base_model="claude-test"),
        ),
        ({"id": "d1"}, None),
        ({"id": "d1", "prompt_cache_observation": False}, None),
        ({"prompt_cache_observation": True}, None),
        (None, None),
    ],
)
def test_only_deployments_that_opt_in_are_observed(model_info: object, expected: SelectedDeployment | None) -> None:
    assert selected_deployment(model_info) == expected


def test_settings_survive_the_deep_copy_logging_makes_of_call_details() -> None:
    settings = _settings({**_ATTEMPT, "cache_control_injection_points": _POINTS})
    assert copy.deepcopy(settings) == settings


def test_logged_model_names_resolve_to_the_served_model() -> None:
    assert served_model("anthropic", "anthropic/claude-test") == served_model("anthropic", "claude-test")
    assert _fingerprint(model="anthropic/claude-test") == _fingerprint(model="claude-test")


def test_a_second_hook_pass_keeps_settings_the_first_pass_saw() -> None:
    outer = _settings({**_ATTEMPT, "cache_control_injection_points": _POINTS})
    merged = dict(merged_cache_settings(outer, _settings()).settings)
    assert merged["cache_control_injection_points"] == _POINTS


def test_settings_for_another_deployment_are_replaced_not_merged() -> None:
    other = _settings({**_ATTEMPT, "model_info": {"id": "other", "prompt_cache_observation": True}})
    assert merged_cache_settings(other, _settings()) == _settings()


def test_a_setting_that_is_not_json_is_kept_as_unknown_rather_than_failing_the_capture() -> None:
    settings = _settings({**_ATTEMPT, "cache_control_injection_points": object()})
    assert dict(settings.settings)["cache_control_injection_points"] is None
