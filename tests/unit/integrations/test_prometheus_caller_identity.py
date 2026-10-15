from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Final, cast
from unittest.mock import patch

import pytest
import yaml
from prometheus_client import REGISTRY, generate_latest
from prometheus_client.parser import text_string_to_metric_families

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.integrations.prometheus import (
    DEFINED_PROMETHEUS_METRICS,
    PROMETHEUS_DEPLOYMENT_AND_LATENCY_CALLER_IDENTITY_METRICS,
    LabelValidationError,
    PrometheusMetricLabels,
    UserAPIKeyLabelNames,
    UserAPIKeyLabelValues,
    validate_caller_identity_settings,
    validate_prometheus_deployment_and_latency_caller_identity,
)
from litellm.types.utils import StandardLoggingPayload

TARGET_METRICS: Final[tuple[DEFINED_PROMETHEUS_METRICS, ...]] = cast(
    tuple[DEFINED_PROMETHEUS_METRICS, ...],
    tuple(sorted(PROMETHEUS_DEPLOYMENT_AND_LATENCY_CALLER_IDENTITY_METRICS)),
)
IDENTITY_MODES: Final = ("api_key_alias", "user_email", "both")


def _clear_prometheus_registry() -> None:
    for collector in list(REGISTRY._collector_to_names):  # pyright: ignore[reportPrivateUsage]
        REGISTRY.unregister(collector)


@pytest.fixture(autouse=True)
def reset_prometheus_settings(monkeypatch: pytest.MonkeyPatch):
    _clear_prometheus_registry()
    monkeypatch.setattr(litellm, "prometheus_deployment_and_latency_caller_identity", "api_key_alias")
    monkeypatch.setattr(litellm, "prometheus_metrics_config", None)
    monkeypatch.setattr(litellm, "prometheus_exclude_metrics", None)
    monkeypatch.setattr(litellm, "prometheus_exclude_labels", None)
    monkeypatch.setattr(litellm, "custom_prometheus_metadata_labels", [])
    monkeypatch.setattr(litellm, "custom_prometheus_tags", [])
    yield
    _clear_prometheus_registry()


def _expected_identity_labels(baseline: list[str], mode: str) -> list[str]:
    expected = list(baseline)
    alias_index = expected.index(UserAPIKeyLabelNames.API_KEY_ALIAS.value)
    if mode == "user_email":
        expected[alias_index] = UserAPIKeyLabelNames.USER_EMAIL.value
    elif mode == "both":
        expected.insert(alias_index + 1, UserAPIKeyLabelNames.USER_EMAIL.value)
    return expected


def _set_caller_identity(monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    monkeypatch.setattr(litellm, "prometheus_deployment_and_latency_caller_identity", mode)


@pytest.mark.parametrize("metric_name", TARGET_METRICS)
@pytest.mark.parametrize("mode", IDENTITY_MODES)
def test_target_metric_label_schema_for_each_caller_identity_mode(
    monkeypatch: pytest.MonkeyPatch,
    metric_name: DEFINED_PROMETHEUS_METRICS,
    mode: str,
):
    _set_caller_identity(monkeypatch, "api_key_alias")
    baseline = PrometheusMetricLabels.get_labels(metric_name)

    _set_caller_identity(monkeypatch, mode)
    actual = PrometheusMetricLabels.get_labels(metric_name)

    assert actual == _expected_identity_labels(baseline, mode)


def test_repeated_label_resolution_does_not_mutate_class_level_or_shared_lists(
    monkeypatch: pytest.MonkeyPatch,
):
    total_request_labels = PrometheusMetricLabels.litellm_deployment_total_requests
    success_labels = PrometheusMetricLabels.litellm_deployment_success_responses
    original = tuple(total_request_labels)

    assert success_labels is total_request_labels
    for mode in (*IDENTITY_MODES, *reversed(IDENTITY_MODES)):
        _set_caller_identity(monkeypatch, mode)
        for metric_name in TARGET_METRICS:
            resolved = PrometheusMetricLabels.get_labels(metric_name)
            assert resolved is not getattr(PrometheusMetricLabels, metric_name)

    assert PrometheusMetricLabels.litellm_deployment_total_requests is total_request_labels
    assert PrometheusMetricLabels.litellm_deployment_success_responses is success_labels
    assert success_labels is total_request_labels
    assert tuple(total_request_labels) == original


def test_invalid_caller_identity_mode_fails_during_prometheus_initialization(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_caller_identity(monkeypatch, "invalid")

    with pytest.raises(
        ValueError,
        match="prometheus_deployment_and_latency_caller_identity",
    ) as exc_info:
        PrometheusLogger()

    message = str(exc_info.value)
    assert "prometheus_deployment_and_latency_caller_identity" in message
    for accepted_value in IDENTITY_MODES:
        assert accepted_value in message


def test_label_resolution_rejects_non_string_class_labels(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        PrometheusMetricLabels,
        "litellm_deployment_total_requests",
        ["api_key_alias", 1],
    )

    with pytest.raises(TypeError, match=r"Prometheus labels .* must be strings"):
        PrometheusMetricLabels.get_labels("litellm_deployment_total_requests")


@pytest.mark.parametrize(
    ("mode", "include_labels", "is_valid"),
    (
        ("api_key_alias", ["api_key_alias"], True),
        ("api_key_alias", ["user_email"], False),
        ("user_email", ["user_email"], True),
        ("user_email", ["api_key_alias"], False),
        ("both", ["api_key_alias"], True),
        ("both", ["user_email"], True),
        ("both", ["api_key_alias", "user_email"], True),
    ),
)
def test_include_labels_validation_matches_caller_identity_mode(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    include_labels: list[str],
    is_valid: bool,
):
    _set_caller_identity(monkeypatch, mode)
    monkeypatch.setattr(
        litellm,
        "prometheus_metrics_config",
        [
            {
                "group": "caller_identity",
                "metrics": ["litellm_deployment_total_requests"],
                "include_labels": include_labels,
            }
        ],
    )

    if not is_valid:
        with pytest.raises(ValueError, match="Configuration validation failed"):
            PrometheusLogger()
        return

    logger = PrometheusLogger()
    assert logger.get_labels_for_metric("litellm_deployment_total_requests") == include_labels


@pytest.mark.parametrize(
    ("mode", "exclude_labels", "remaining_identity_labels"),
    (
        ("api_key_alias", ["api_key_alias"], set[str]()),
        ("api_key_alias", ["user_email"], {"api_key_alias"}),
        ("user_email", ["user_email"], set[str]()),
        ("user_email", ["api_key_alias"], {"user_email"}),
        ("both", ["api_key_alias"], {"user_email"}),
        ("both", ["user_email"], {"api_key_alias"}),
        ("both", ["api_key_alias", "user_email"], set[str]()),
    ),
)
def test_exclude_labels_can_remove_supported_identity_labels(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    exclude_labels: list[str],
    remaining_identity_labels: set[str],
):
    _set_caller_identity(monkeypatch, mode)
    monkeypatch.setattr(litellm, "prometheus_exclude_labels", exclude_labels)

    logger = PrometheusLogger()
    labels = logger.get_labels_for_metric("litellm_deployment_total_requests")

    assert set(labels) & {"api_key_alias", "user_email"} == remaining_identity_labels


@pytest.mark.parametrize("mode", IDENTITY_MODES)
def test_non_target_metric_label_schema_is_unchanged(monkeypatch: pytest.MonkeyPatch, mode: str):
    baseline = list(PrometheusMetricLabels.litellm_overhead_with_guardrails_latency_metric)
    _set_caller_identity(monkeypatch, mode)

    actual = PrometheusMetricLabels.get_labels("litellm_overhead_with_guardrails_latency_metric")

    assert actual == baseline
    assert "api_key_alias" in actual
    assert "user_email" not in actual


def _standard_logging_payload(user_email: str | None = "alice@example.com") -> StandardLoggingPayload:
    return cast(
        StandardLoggingPayload,
        {
            "api_base": "https://api.example.com",
            "model_group": "requested-model",
            "model_id": "deployment-id",
            "request_tags": [],
            "metadata": {
                "user_api_key_hash": "hashed-key",
                "user_api_key_alias": "alias-a",
                "user_api_key_user_email": user_email,
                "user_api_key_team_id": "team-id",
                "user_api_key_team_alias": "team-alias",
                "requester_ip_address": "192.0.2.10",
                "user_agent": "caller-identity-test",
            },
            "hidden_params": {
                "additional_headers": None,
                "litellm_overhead_time_ms": 125,
            },
        },
    )


def _sample_labels(scrape: str, sample_name: str) -> list[dict[str, str]]:
    return [
        sample.labels
        for family in text_string_to_metric_families(scrape)
        for sample in family.samples
        if sample.name == sample_name
    ]


@pytest.mark.parametrize("mode", IDENTITY_MODES)
def test_successful_request_emits_configured_identity_on_real_counter_and_histogram_samples(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
):
    _set_caller_identity(monkeypatch, mode)
    logger = PrometheusLogger()
    payload = _standard_logging_payload()
    enum_values = UserAPIKeyLabelValues(
        end_user="end-user",
        user="user-id",
        user_email="alice@example.com",
        hashed_api_key="hashed-key",
        api_key_alias="alias-a",
        requested_model="requested-model",
        model_group="requested-model",
        team="team-id",
        team_alias="team-alias",
        model="provider-model",
        litellm_model_name="deployment-model",
        model_id="deployment-id",
        api_base="https://api.example.com",
        api_provider="openai",
        client_ip="192.0.2.10",
        user_agent="caller-identity-test",
    )
    start_time = datetime.now()
    api_call_start_time = start_time + timedelta(milliseconds=100)
    completion_start_time = api_call_start_time + timedelta(milliseconds=200)
    end_time = start_time + timedelta(seconds=1)
    request_kwargs = {
        "model": "deployment-model",
        "stream": True,
        "start_time": start_time,
        "api_call_start_time": api_call_start_time,
        "completion_start_time": completion_start_time,
        "end_time": end_time,
        "litellm_params": {
            "custom_llm_provider": "openai",
            "metadata": {
                "model_info": {"id": "deployment-id"},
                "queue_time_seconds": 0.05,
            },
        },
        "standard_logging_object": payload,
    }

    logger._set_latency_metrics(  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
        kwargs=request_kwargs,
        model="deployment-model",
        user_api_key="hashed-key",
        user_api_key_alias="alias-a",
        user_api_team="team-id",
        user_api_team_alias="team-alias",
        enum_values=enum_values,
    )
    logger.set_llm_deployment_success_metrics(  # pyright: ignore[reportUnknownMemberType]
        request_kwargs=request_kwargs,
        start_time=start_time,
        end_time=end_time,
        enum_values=enum_values,
        output_tokens=10,
    )

    scrape = generate_latest(REGISTRY).decode()
    sample_names = (
        "litellm_deployment_total_requests_total",
        "litellm_deployment_success_responses_total",
        "litellm_request_total_latency_metric_count",
        "litellm_llm_api_latency_metric_count",
        "litellm_llm_api_time_to_first_token_metric_count",
        "litellm_request_queue_time_seconds_count",
        "litellm_overhead_latency_metric_count",
        "litellm_deployment_latency_per_output_token_count",
    )
    for sample_name in sample_names:
        samples = _sample_labels(scrape, sample_name)
        assert len(samples) == 1, sample_name
        labels = samples[0]
        if mode == "api_key_alias":
            assert labels["api_key_alias"] == "alias-a"
            assert "user_email" not in labels
        elif mode == "user_email":
            assert labels["user_email"] == "alice@example.com"
            assert "api_key_alias" not in labels
        else:
            assert labels["api_key_alias"] == "alias-a"
            assert labels["user_email"] == "alice@example.com"


@pytest.mark.parametrize(
    ("standard_email", "metadata_email", "auth_email", "expected_email"),
    (
        ("standard@example.com", "metadata@example.com", "auth@example.com", "standard@example.com"),
        (None, "metadata@example.com", "auth@example.com", "metadata@example.com"),
        (None, None, "auth@example.com", "auth@example.com"),
        (None, None, None, "None"),
    ),
)
def test_deployment_failure_email_fallbacks_reach_both_real_counters(
    monkeypatch: pytest.MonkeyPatch,
    standard_email: str | None,
    metadata_email: str | None,
    auth_email: str | None,
    expected_email: str,
):
    _set_caller_identity(monkeypatch, "both")
    logger = PrometheusLogger()
    payload = _standard_logging_payload(user_email=standard_email)
    metadata = {
        "model_info": {"id": "deployment-id"},
        "user_api_key_user_email": metadata_email,
        "user_api_key_auth": UserAPIKeyAuth(user_email=auth_email),
    }
    request_kwargs = {
        "model": "deployment-model",
        "litellm_params": {
            "custom_llm_provider": "openai",
            "metadata": metadata,
        },
        "standard_logging_object": payload,
        "exception": RuntimeError("provider failed"),
    }

    logger.set_llm_deployment_failure_metrics(request_kwargs)  # pyright: ignore[reportUnknownMemberType]

    scrape = generate_latest(REGISTRY).decode()
    for sample_name in (
        "litellm_deployment_failure_responses_total",
        "litellm_deployment_total_requests_total",
    ):
        samples = _sample_labels(scrape, sample_name)
        assert len(samples) == 1, sample_name
        assert samples[0]["api_key_alias"] == "alias-a"
        assert samples[0]["user_email"] == expected_email


@pytest.mark.asyncio
async def test_proxy_config_loads_caller_identity_before_initializing_callbacks(tmp_path: Path):
    from litellm.proxy.proxy_server import ProxyConfig

    config_path = _write_proxy_config(
        tmp_path,
        {
            "callbacks": ["prometheus"],
            "prometheus_deployment_and_latency_caller_identity": "both",
        },
    )
    observed_modes: list[str] = []

    def capture_mode(*args: object, **kwargs: object) -> None:
        observed_modes.append(litellm.prometheus_deployment_and_latency_caller_identity)

    with patch(  # test-quality-ok: callback interception verifies schema selection before construction
        "litellm.proxy.proxy_server.initialize_callbacks_on_proxy", side_effect=capture_mode
    ):
        await ProxyConfig().load_config(router=None, config_file_path=str(config_path))

    assert observed_modes == ["both"]
    assert litellm.prometheus_deployment_and_latency_caller_identity == "both"


def _identity_settings(mode: object, metrics_config: object = None) -> dict[str, object]:
    settings: dict[str, object] = {"prometheus_deployment_and_latency_caller_identity": mode}
    if metrics_config is not None:
        settings["prometheus_metrics_config"] = metrics_config
    return settings


def test_validate_mode_returns_each_accepted_value_and_defaults_to_api_key_alias(
    monkeypatch: pytest.MonkeyPatch,
):
    for mode in IDENTITY_MODES:
        _set_caller_identity(monkeypatch, mode)
        assert validate_prometheus_deployment_and_latency_caller_identity() == mode

    monkeypatch.delattr(litellm, "prometheus_deployment_and_latency_caller_identity")
    assert validate_prometheus_deployment_and_latency_caller_identity() == "api_key_alias"


def test_accepted_values_constant_matches_parametrized_modes():
    from litellm.types.integrations.prometheus import (
        PROMETHEUS_DEPLOYMENT_AND_LATENCY_CALLER_IDENTITY_VALUES,
    )

    assert PROMETHEUS_DEPLOYMENT_AND_LATENCY_CALLER_IDENTITY_VALUES == IDENTITY_MODES
    assert len(TARGET_METRICS) == 9


@pytest.mark.parametrize(
    "invalid_mode",
    ("user-email", "USER_EMAIL", "", None, True, 1, ["user_email"], {"mode": "user_email"}),
)
def test_validate_mode_rejects_invalid_values_and_names_accepted_ones(
    monkeypatch: pytest.MonkeyPatch,
    invalid_mode: object,
):
    monkeypatch.setattr(litellm, "prometheus_deployment_and_latency_caller_identity", invalid_mode)

    with pytest.raises(ValueError, match="prometheus_deployment_and_latency_caller_identity") as exc_info:
        validate_prometheus_deployment_and_latency_caller_identity()

    message = str(exc_info.value)
    assert repr(invalid_mode) in message
    for accepted_value in IDENTITY_MODES:
        assert accepted_value in message


def test_validate_caller_identity_settings_without_key_leaves_mode_untouched(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_caller_identity(monkeypatch, "both")

    validate_caller_identity_settings({"prometheus_metrics_config": []})

    assert litellm.prometheus_deployment_and_latency_caller_identity == "both"


@pytest.mark.parametrize("mode", IDENTITY_MODES)
def test_validate_caller_identity_settings_stores_each_valid_mode(mode: str):
    validate_caller_identity_settings(_identity_settings(mode))

    assert litellm.prometheus_deployment_and_latency_caller_identity == mode


@pytest.mark.parametrize("invalid_mode", ("user-email", None))
def test_validate_caller_identity_settings_rejects_invalid_and_null_modes(invalid_mode: object):
    with pytest.raises(ValueError, match="prometheus_deployment_and_latency_caller_identity"):
        validate_caller_identity_settings(_identity_settings(invalid_mode))


def test_user_email_mode_conflict_error_names_every_conflicting_metric_and_only_those():
    metrics_config = [
        {
            "group": "non_target",
            "metrics": ["litellm_overhead_with_guardrails_latency_metric"],
            "include_labels": ["api_key_alias"],
        },
        {
            "group": "target_pair",
            "metrics": ["litellm_deployment_total_requests", "litellm_llm_api_latency_metric"],
            "include_labels": ["api_key_alias"],
        },
        {
            "group": "target_single",
            "metrics": ["litellm_request_queue_time_seconds"],
            "include_labels": ["api_key_alias"],
        },
    ]

    with pytest.raises(ValueError, match="prometheus_deployment_and_latency_caller_identity") as exc_info:
        validate_caller_identity_settings(_identity_settings("user_email", metrics_config))

    message = str(exc_info.value)
    for conflicting_metric in (
        "litellm_deployment_total_requests",
        "litellm_llm_api_latency_metric",
        "litellm_request_queue_time_seconds",
    ):
        assert conflicting_metric in message
    assert "litellm_overhead_with_guardrails_latency_metric" not in message
    assert "prometheus_deployment_and_latency_caller_identity" in message
    assert "user_email" in message


@pytest.mark.parametrize(
    ("mode", "metrics_config"),
    (
        (
            "user_email",
            [
                {
                    "group": "g",
                    "metrics": ["litellm_deployment_total_requests"],
                    "include_labels": ["user_email"],
                }
            ],
        ),
        (
            "user_email",
            [
                {
                    "group": "g",
                    "metrics": ["litellm_overhead_with_guardrails_latency_metric"],
                    "include_labels": ["api_key_alias"],
                }
            ],
        ),
        (
            "api_key_alias",
            [
                {
                    "group": "g",
                    "metrics": ["litellm_deployment_total_requests"],
                    "include_labels": ["api_key_alias"],
                }
            ],
        ),
        (
            "both",
            [
                {
                    "group": "g",
                    "metrics": ["litellm_deployment_total_requests"],
                    "include_labels": ["api_key_alias"],
                }
            ],
        ),
        ("user_email", None),
        ("user_email", ["not-a-dict"]),
        (
            "user_email",
            [{"group": "g", "metrics": ["litellm_deployment_total_requests"], "include_labels": None}],
        ),
        ("user_email", [{"group": "g", "metrics": None, "include_labels": ["api_key_alias"]}]),
    ),
)
def test_validate_caller_identity_settings_accepts_non_conflicting_configs(
    mode: str,
    metrics_config: object,
):
    settings = _identity_settings(mode)
    settings["prometheus_metrics_config"] = metrics_config

    validate_caller_identity_settings(settings)

    assert litellm.prometheus_deployment_and_latency_caller_identity == mode


def _write_proxy_config(tmp_path: Path, litellm_settings: dict[str, object]) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model_list": [
                    {
                        "model_name": "test-model",
                        "litellm_params": {"model": "openai/gpt-4", "api_key": "test-key"},
                    }
                ],
                "litellm_settings": litellm_settings,
            },
            sort_keys=False,
        )
    )
    return config_path


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "litellm_settings",
    (
        {
            "callbacks": ["prometheus"],
            "prometheus_deployment_and_latency_caller_identity": "user-email",
        },
        {
            "callbacks": ["prometheus"],
            "prometheus_deployment_and_latency_caller_identity": None,
        },
        {
            "callbacks": ["prometheus"],
            "prometheus_deployment_and_latency_caller_identity": "user_email",
            "prometheus_metrics_config": [
                {
                    "group": "g",
                    "metrics": ["litellm_deployment_total_requests"],
                    "include_labels": ["api_key_alias"],
                }
            ],
        },
    ),
    ids=("typo-mode", "null-mode", "include-labels-conflict"),
)
async def test_proxy_config_fails_boot_before_callbacks_on_invalid_caller_identity_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    litellm_settings: dict[str, object],
):
    from litellm.proxy.proxy_server import ProxyConfig

    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", False)
    monkeypatch.delenv("LITELLM_CONFIG_BUCKET_NAME", raising=False)
    config_path = _write_proxy_config(tmp_path, litellm_settings)

    with patch(  # test-quality-ok: asserts boot fails before any callback initialization
        "litellm.proxy.proxy_server.initialize_callbacks_on_proxy"
    ) as callback_init:
        with pytest.raises(ValueError, match="prometheus_deployment_and_latency_caller_identity"):
            await ProxyConfig().load_config(router=None, config_file_path=str(config_path))

    callback_init.assert_not_called()


def test_failed_init_leaves_registry_clean_so_a_corrected_retry_succeeds(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_caller_identity(monkeypatch, "user-email")
    with pytest.raises(ValueError, match="prometheus_deployment_and_latency_caller_identity"):
        PrometheusLogger()

    assert list(REGISTRY._collector_to_names) == []  # pyright: ignore[reportPrivateUsage]

    _set_caller_identity(monkeypatch, "user_email")
    logger = PrometheusLogger()
    assert "user_email" in logger.get_labels_for_metric("litellm_deployment_total_requests")


@pytest.mark.parametrize("invalid_label", ("api_key_alias", "user_email"))
def test_label_validation_error_names_mode_setting_for_identity_labels_on_target_metric(
    monkeypatch: pytest.MonkeyPatch,
    invalid_label: str,
):
    _set_caller_identity(monkeypatch, "user_email")

    error = LabelValidationError(
        metric_name="litellm_deployment_total_requests",
        invalid_labels=[invalid_label],
        valid_labels=["user_email"],
    )

    assert "prometheus_deployment_and_latency_caller_identity='user_email'" in error.message
    assert invalid_label in error.message


def test_label_validation_error_keeps_base_message_for_non_identity_cases(
    monkeypatch: pytest.MonkeyPatch,
):
    _set_caller_identity(monkeypatch, "user_email")
    non_target_metric = LabelValidationError(
        metric_name="litellm_overhead_with_guardrails_latency_metric",
        invalid_labels=["api_key_alias"],
        valid_labels=[],
    )
    non_identity_label = LabelValidationError(
        metric_name="litellm_deployment_total_requests",
        invalid_labels=["bogus_label"],
        valid_labels=[],
    )

    for error in (non_target_metric, non_identity_label):
        assert "caller-identity" not in error.message
        assert error.message.startswith("Invalid labels for metric")
