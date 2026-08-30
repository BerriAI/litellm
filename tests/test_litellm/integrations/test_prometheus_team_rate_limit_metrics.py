"""
Tests for the team-scoped rate limit Prometheus gauges.

LiteLLM exposed configured/remaining rate limits at virtual key scope
(``litellm_remaining_api_key_*_for_model``) and deployment scope
(``litellm_deployment_{tpm,rpm}_limit``) but not at team scope, so there was
no way to alert on a team approaching the ``model_tpm_limit`` /
``model_rpm_limit`` configured on its team object.

The v3 rate limiter already computes those numbers for its ``model_per_team``
descriptor and ships them to clients as
``x-ratelimit-model_per_team-{remaining,limit}-{requests,tokens}``. These
tests cover routing those already-computed values to Prometheus.
"""

from typing import get_args
from unittest.mock import MagicMock, patch

import pytest

from prometheus_client import CollectorRegistry, Gauge

from litellm.integrations.prometheus import PrometheusLogger, _ExcludedLabelMetric
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _PROXY_MaxParallelRequestsHandler_v3,
)
from litellm.types.integrations.prometheus import (
    DEFINED_PROMETHEUS_METRICS,
    NoOpMetric,
    PrometheusMetricLabels,
    UserAPIKeyLabelNames,
)

TEAM_LABELS = {"team": "team-abc", "team_alias": "research", "model": "gpt-4o-mini"}
ORIGINAL_LABELNAMES = ("team", "team_alias", "model")

TEAM_RATE_LIMIT_METRICS = (
    "litellm_remaining_team_requests_for_model",
    "litellm_remaining_team_tokens_for_model",
    "litellm_team_rpm_limit",
    "litellm_team_tpm_limit",
)


@pytest.fixture(autouse=True)
def _single_process_collection(monkeypatch):
    """
    Series retirement is only possible outside multiprocess collection, so pin
    the mode rather than depending on whatever the ambient environment has set.
    """
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    monkeypatch.delenv("prometheus_multiproc_dir", raising=False)


def _logger_with_mock_team_gauges() -> PrometheusLogger:
    with patch("litellm.integrations.prometheus.PrometheusLogger.__init__", return_value=None):
        logger = PrometheusLogger()
    for metric_name in TEAM_RATE_LIMIT_METRICS:
        setattr(logger, metric_name, MagicMock())
    logger.get_labels_for_metric = MagicMock(side_effect=PrometheusMetricLabels.get_labels)
    logger._team_series_label_values = {}
    return logger


def _payload_with_headers(additional_headers: dict) -> dict:
    return {
        "metadata": {},
        "hidden_params": {"additional_headers": additional_headers},
    }


def _set_team_metrics(logger: PrometheusLogger, standard_logging_payload: dict) -> None:
    logger._set_team_rate_limit_metrics(
        user_api_team="team-abc",
        user_api_team_alias="research",
        model_group="gpt-4o-mini",
        standard_logging_payload=standard_logging_payload,
    )


def _assert_set_once(logger: PrometheusLogger, metric_name: str, value: int) -> None:
    getattr(logger, metric_name).labels.return_value.set.assert_called_once_with(value)


ALL_TEAM_HEADERS = {
    "x-ratelimit-model_per_team-remaining-requests": 42,
    "x-ratelimit-model_per_team-remaining-tokens": 900,
    "x-ratelimit-model_per_team-limit-requests": 100,
    "x-ratelimit-model_per_team-limit-tokens": 1000,
}


def test_team_metrics_are_defined_with_team_and_model_labels():
    defined_metrics = get_args(DEFINED_PROMETHEUS_METRICS)
    expected_labels = [
        UserAPIKeyLabelNames.TEAM.value,
        UserAPIKeyLabelNames.TEAM_ALIAS.value,
        UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value,
    ]

    for metric_name in TEAM_RATE_LIMIT_METRICS:
        assert metric_name in defined_metrics
        labels = PrometheusMetricLabels.get_labels(metric_name)
        for expected_label in expected_labels:
            assert expected_label in labels


def test_every_logger_owned_metric_resolves_labels():
    """
    ``PrometheusMetricLabels.get_labels`` resolves a metric name to a label
    list via ``getattr``, so a metric added to the literal without a matching
    label attribute fails at logger construction time in production rather
    than at lint time.

    ``litellm_in_flight_requests`` is excluded because it is a label-free
    gauge registered by the in-flight middleware, not by ``PrometheusLogger``;
    it appears in the literal only so ``prometheus_metrics_config`` can name it.
    """
    for metric_name in get_args(DEFINED_PROMETHEUS_METRICS):
        if metric_name == "litellm_in_flight_requests":
            continue
        assert isinstance(PrometheusMetricLabels.get_labels(metric_name), list)


def test_sets_every_team_gauge_from_v3_headers():
    logger = _logger_with_mock_team_gauges()

    _set_team_metrics(logger, _payload_with_headers(dict(ALL_TEAM_HEADERS)))

    _assert_set_once(logger, "litellm_remaining_team_requests_for_model", 42)
    _assert_set_once(logger, "litellm_remaining_team_tokens_for_model", 900)
    _assert_set_once(logger, "litellm_team_rpm_limit", 100)
    _assert_set_once(logger, "litellm_team_tpm_limit", 1000)


def test_labels_carry_team_and_requested_model():
    logger = _logger_with_mock_team_gauges()

    _set_team_metrics(logger, _payload_with_headers(dict(ALL_TEAM_HEADERS)))

    for metric_name in TEAM_RATE_LIMIT_METRICS:
        labelnames = PrometheusMetricLabels.get_labels(metric_name)
        label_values = getattr(logger, metric_name).labels.call_args.args
        assert dict(zip(labelnames, label_values, strict=True)) == TEAM_LABELS


def test_emits_nothing_when_team_has_no_configured_limits():
    """A team without per-model limits gets no descriptor, so no header, so no series."""
    logger = _logger_with_mock_team_gauges()

    _set_team_metrics(
        logger,
        _payload_with_headers(
            {
                "x-ratelimit-model_per_key-remaining-requests": 42,
                "x-ratelimit-model_per_key-limit-requests": 100,
            }
        ),
    )

    for metric_name in TEAM_RATE_LIMIT_METRICS:
        getattr(logger, metric_name).labels.assert_not_called()


def test_drops_stale_series_when_a_team_limit_is_removed():
    """
    Prometheus keeps a child series for the life of the process once emitted,
    so a team whose limit is removed would otherwise keep publishing the last
    values it saw and alerts would fire on a limit nobody enforces.
    """
    logger = _logger_with_mock_team_gauges()

    _set_team_metrics(logger, _payload_with_headers(dict(ALL_TEAM_HEADERS)))
    _assert_set_once(logger, "litellm_remaining_team_requests_for_model", 42)

    _set_team_metrics(logger, _payload_with_headers({}))

    for metric_name in TEAM_RATE_LIMIT_METRICS:
        gauge = getattr(logger, metric_name)
        gauge.remove.assert_called_once_with("team-abc", "research", "gpt-4o-mini")


def test_survives_removing_a_series_that_was_never_emitted():
    """The common case: a team that never had a limit for this model."""
    logger = _logger_with_mock_team_gauges()
    for metric_name in TEAM_RATE_LIMIT_METRICS:
        getattr(logger, metric_name).remove.side_effect = KeyError("not present")

    _set_team_metrics(logger, _payload_with_headers({}))

    for metric_name in TEAM_RATE_LIMIT_METRICS:
        getattr(logger, metric_name).labels.assert_not_called()


def test_emits_only_the_dimension_the_team_configured():
    """A team with only an RPM limit must not get a fabricated TPM series."""
    logger = _logger_with_mock_team_gauges()

    _set_team_metrics(
        logger,
        _payload_with_headers(
            {
                "x-ratelimit-model_per_team-remaining-requests": 7,
                "x-ratelimit-model_per_team-limit-requests": 60,
            }
        ),
    )

    _assert_set_once(logger, "litellm_remaining_team_requests_for_model", 7)
    _assert_set_once(logger, "litellm_team_rpm_limit", 60)
    logger.litellm_remaining_team_tokens_for_model.labels.assert_not_called()
    logger.litellm_team_tpm_limit.labels.assert_not_called()


def test_emits_zero_remaining_rather_than_skipping_it():
    """An exhausted team is the case operators alert on, so 0 must be a real sample."""
    logger = _logger_with_mock_team_gauges()

    _set_team_metrics(
        logger,
        _payload_with_headers(
            {
                "x-ratelimit-model_per_team-remaining-requests": 0,
                "x-ratelimit-model_per_team-remaining-tokens": 0,
            }
        ),
    )

    _assert_set_once(logger, "litellm_remaining_team_requests_for_model", 0)
    _assert_set_once(logger, "litellm_remaining_team_tokens_for_model", 0)


def test_emits_nothing_for_a_request_with_no_team():
    logger = _logger_with_mock_team_gauges()

    logger._set_team_rate_limit_metrics(
        user_api_team=None,
        user_api_team_alias=None,
        model_group="gpt-4o-mini",
        standard_logging_payload=_payload_with_headers(dict(ALL_TEAM_HEADERS)),
    )

    for metric_name in TEAM_RATE_LIMIT_METRICS:
        getattr(logger, metric_name).labels.assert_not_called()


@pytest.mark.parametrize("bad_value", ["100", None, True, 12.5])
def test_ignores_non_int_header_values(bad_value):
    logger = _logger_with_mock_team_gauges()

    _set_team_metrics(
        logger,
        _payload_with_headers({"x-ratelimit-model_per_team-remaining-requests": bad_value}),
    )

    logger.litellm_remaining_team_requests_for_model.labels.assert_not_called()


def test_raises_nothing_when_payload_has_no_hidden_params():
    logger = _logger_with_mock_team_gauges()

    _set_team_metrics(logger, {"metadata": {}})

    for metric_name in TEAM_RATE_LIMIT_METRICS:
        getattr(logger, metric_name).labels.assert_not_called()


def test_limiter_publishes_team_headers_in_the_shape_the_gauges_read():
    """
    Pins the producer/consumer contract: the gauges read header names the v3
    limiter builds from ``descriptor_key`` + ``rate_limit_type``, so a change
    to that format would otherwise silently stop the team series.
    """
    headers = _PROXY_MaxParallelRequestsHandler_v3._merge_ratelimit_statuses_into_additional_headers(
        additional_headers={},
        statuses=[
            {
                "code": "OK",
                "current_limit": 100,
                "limit_remaining": 42,
                "rate_limit_type": "requests",
                "descriptor_key": "model_per_team",
            },
            {
                "code": "OK",
                "current_limit": 1000,
                "limit_remaining": 900,
                "rate_limit_type": "tokens",
                "descriptor_key": "model_per_team",
            },
        ],
    )

    assert headers == {
        "x-ratelimit-model_per_team-remaining-requests": 42,
        "x-ratelimit-model_per_team-limit-requests": 100,
        "x-ratelimit-model_per_team-remaining-tokens": 900,
        "x-ratelimit-model_per_team-limit-tokens": 1000,
    }


def _logger_with_real_gauge(metric_name: str, gauge: Gauge) -> PrometheusLogger:
    logger = _logger_with_mock_team_gauges()
    setattr(logger, metric_name, gauge)
    return logger


def test_removes_a_real_prometheus_child_series_when_the_limit_disappears():
    """
    Mock gauges cannot prove the drop works, since prometheus_client owns the
    child-series bookkeeping. This drives the real Gauge end to end.
    """
    registry = CollectorRegistry()
    gauge = Gauge("litellm_team_rpm_limit", "doc", labelnames=list(ORIGINAL_LABELNAMES), registry=registry)
    logger = _logger_with_real_gauge("litellm_team_rpm_limit", gauge)

    _set_team_metrics(logger, _payload_with_headers({"x-ratelimit-model_per_team-limit-requests": 60}))
    assert registry.get_sample_value("litellm_team_rpm_limit", TEAM_LABELS) == 60

    _set_team_metrics(logger, _payload_with_headers({}))
    assert registry.get_sample_value("litellm_team_rpm_limit", TEAM_LABELS) is None


def test_removing_a_never_emitted_real_series_raises_nothing():
    registry = CollectorRegistry()
    gauge = Gauge("litellm_team_tpm_limit", "doc", labelnames=list(ORIGINAL_LABELNAMES), registry=registry)
    logger = _logger_with_real_gauge("litellm_team_tpm_limit", gauge)

    _set_team_metrics(logger, _payload_with_headers({}))

    assert registry.get_sample_value("litellm_team_tpm_limit", TEAM_LABELS) is None


def test_excluded_label_wrapper_sets_and_removes_using_the_kept_labels():
    registry = CollectorRegistry()
    real = Gauge("litellm_team_rpm_limit", "doc", labelnames=["team", "model"], registry=registry)
    wrapper = _ExcludedLabelMetric(real, ORIGINAL_LABELNAMES, frozenset({"team_alias"}))
    kept = {"team": "team-abc", "model": "gpt-4o-mini"}

    wrapper.labels(**TEAM_LABELS).set(60)
    assert registry.get_sample_value("litellm_team_rpm_limit", kept) == 60

    wrapper.remove(*(TEAM_LABELS[name] for name in ORIGINAL_LABELNAMES))
    assert registry.get_sample_value("litellm_team_rpm_limit", kept) is None


def test_excluded_label_wrapper_cannot_remove_when_every_label_is_excluded():
    """
    With every label excluded the gauge collapses to a single unlabeled sample,
    which has no child series for prometheus_client to remove. Such a metric
    cannot represent per-team state at all, so there is no correct value to
    drop it to; this pins the behaviour rather than papering over it.
    """
    registry = CollectorRegistry()
    real = Gauge("litellm_team_rpm_limit", "doc", registry=registry)
    wrapper = _ExcludedLabelMetric(real, ORIGINAL_LABELNAMES, frozenset(ORIGINAL_LABELNAMES))

    wrapper.labels(**TEAM_LABELS).set(60)
    assert registry.get_sample_value("litellm_team_rpm_limit", {}) == 60

    wrapper.remove(*(TEAM_LABELS[name] for name in ORIGINAL_LABELNAMES))
    assert registry.get_sample_value("litellm_team_rpm_limit", {}) == 60


def test_noop_metric_remove_is_inert():
    """A disabled metric answers every call without recording or raising."""
    metric = NoOpMetric()

    child = metric.labels(*TEAM_LABELS.values())

    assert child is metric
    assert child.set(60) is None
    assert metric.remove(*TEAM_LABELS.values()) is None


def test_retires_the_old_series_when_a_team_is_renamed():
    """
    A rename changes team_alias, which starts a new series. The old one would
    otherwise keep publishing the values it held at rename time, double
    counting the team on any sum over `team`.
    """
    registry = CollectorRegistry()
    gauge = Gauge("litellm_team_rpm_limit", "doc", labelnames=list(ORIGINAL_LABELNAMES), registry=registry)
    logger = _logger_with_real_gauge("litellm_team_rpm_limit", gauge)
    headers = {"x-ratelimit-model_per_team-limit-requests": 60}

    _set_team_metrics(logger, _payload_with_headers(headers))
    assert registry.get_sample_value("litellm_team_rpm_limit", TEAM_LABELS) == 60

    logger._set_team_rate_limit_metrics(
        user_api_team="team-abc",
        user_api_team_alias="ml-research",
        model_group="gpt-4o-mini",
        standard_logging_payload=_payload_with_headers(headers),
    )

    renamed = {**TEAM_LABELS, "team_alias": "ml-research"}
    assert registry.get_sample_value("litellm_team_rpm_limit", renamed) == 60
    assert registry.get_sample_value("litellm_team_rpm_limit", TEAM_LABELS) is None


def test_keeps_other_teams_when_one_team_is_renamed():
    registry = CollectorRegistry()
    gauge = Gauge("litellm_team_rpm_limit", "doc", labelnames=list(ORIGINAL_LABELNAMES), registry=registry)
    logger = _logger_with_real_gauge("litellm_team_rpm_limit", gauge)
    headers = {"x-ratelimit-model_per_team-limit-requests": 60}

    logger._set_team_rate_limit_metrics(
        user_api_team="team-other",
        user_api_team_alias="platform",
        model_group="gpt-4o-mini",
        standard_logging_payload=_payload_with_headers(headers),
    )
    _set_team_metrics(logger, _payload_with_headers(headers))
    logger._set_team_rate_limit_metrics(
        user_api_team="team-abc",
        user_api_team_alias="ml-research",
        model_group="gpt-4o-mini",
        standard_logging_payload=_payload_with_headers(headers),
    )

    other = {"team": "team-other", "team_alias": "platform", "model": "gpt-4o-mini"}
    assert registry.get_sample_value("litellm_team_rpm_limit", other) == 60


def test_emits_nothing_when_the_team_label_is_excluded():
    """
    Without a team label the gauge collapses to one sample shared by every
    team, which attributes a limit to nobody and cannot be retired.
    """
    logger = _logger_with_mock_team_gauges()
    logger.get_labels_for_metric = MagicMock(return_value=["model"])

    _set_team_metrics(logger, _payload_with_headers(dict(ALL_TEAM_HEADERS)))

    for metric_name in TEAM_RATE_LIMIT_METRICS:
        getattr(logger, metric_name).labels.assert_not_called()
        getattr(logger, metric_name).remove.assert_not_called()


def test_excluded_labels_never_reach_team_gauge_labelnames():
    """
    `exclude_labels` is applied inside `get_labels_for_metric`, so the
    labelnames a team gauge is constructed with never contain an excluded
    label. The factory only wraps a metric when its labelnames still intersect
    `exclude_labels`, so these gauges are always real prometheus_client
    Gauges and always expose `collect` for alias cleanup.
    """
    with patch("litellm.integrations.prometheus.PrometheusLogger.__init__", return_value=None):
        logger = PrometheusLogger()
    logger.exclude_labels = frozenset({"model", "team_alias"})
    logger.label_filters = {}
    logger._cached_metric_labels = {}

    for metric_name in TEAM_RATE_LIMIT_METRICS:
        labelnames = logger.get_labels_for_metric(metric_name)
        assert not frozenset(labelnames) & logger.exclude_labels
        assert "team" in labelnames


def test_retires_the_old_alias_when_the_limit_is_removed_after_a_rename():
    """
    A team can be renamed and then have its limit removed before it sends
    another limited request. The removal path only knows the current
    labelset, so without sweeping superseded aliases on that path too, the
    pre-rename series would stay published forever.
    """
    registry = CollectorRegistry()
    gauge = Gauge("litellm_team_rpm_limit", "doc", labelnames=list(ORIGINAL_LABELNAMES), registry=registry)
    logger = _logger_with_real_gauge("litellm_team_rpm_limit", gauge)

    _set_team_metrics(logger, _payload_with_headers({"x-ratelimit-model_per_team-limit-requests": 60}))
    assert registry.get_sample_value("litellm_team_rpm_limit", TEAM_LABELS) == 60

    logger._set_team_rate_limit_metrics(
        user_api_team="team-abc",
        user_api_team_alias="ml-research",
        model_group="gpt-4o-mini",
        standard_logging_payload=_payload_with_headers({}),
    )

    assert registry.get_sample_value("litellm_team_rpm_limit", TEAM_LABELS) is None
    renamed = {**TEAM_LABELS, "team_alias": "ml-research"}
    assert registry.get_sample_value("litellm_team_rpm_limit", renamed) is None


def test_rename_survives_a_tracked_series_that_is_already_gone():
    """
    The tracked labelset can outlive the child series it names, for instance
    when a cardinality cap evicts it. Retiring it must not break the emission
    that triggered the retirement.
    """
    registry = CollectorRegistry()
    gauge = Gauge("litellm_team_rpm_limit", "doc", labelnames=list(ORIGINAL_LABELNAMES), registry=registry)
    logger = _logger_with_real_gauge("litellm_team_rpm_limit", gauge)
    headers = {"x-ratelimit-model_per_team-limit-requests": 60}

    _set_team_metrics(logger, _payload_with_headers(headers))
    gauge.remove(*(TEAM_LABELS[name] for name in ORIGINAL_LABELNAMES))
    assert registry.get_sample_value("litellm_team_rpm_limit", TEAM_LABELS) is None

    logger._set_team_rate_limit_metrics(
        user_api_team="team-abc",
        user_api_team_alias="ml-research",
        model_group="gpt-4o-mini",
        standard_logging_payload=_payload_with_headers(headers),
    )

    renamed = {**TEAM_LABELS, "team_alias": "ml-research"}
    assert registry.get_sample_value("litellm_team_rpm_limit", renamed) == 60


def test_does_not_attempt_retirement_under_multiprocess_collection(monkeypatch):
    """
    prometheus_client refuses to remove a labelset when PROMETHEUS_MULTIPROC_DIR
    is set, warning instead, because a worker cannot retire a series another
    worker wrote. Attempting it on every team request would emit warnings while
    leaving the sample in place, so the gauges are set and nothing is retired.
    """
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", "/tmp/does-not-need-to-exist")
    logger = _logger_with_mock_team_gauges()

    _set_team_metrics(logger, _payload_with_headers(dict(ALL_TEAM_HEADERS)))
    _set_team_metrics(logger, _payload_with_headers({}))

    _assert_set_once(logger, "litellm_remaining_team_requests_for_model", 42)
    for metric_name in TEAM_RATE_LIMIT_METRICS:
        getattr(logger, metric_name).remove.assert_not_called()


def test_retires_series_when_collection_is_single_process(monkeypatch):
    monkeypatch.delenv("PROMETHEUS_MULTIPROC_DIR", raising=False)
    monkeypatch.delenv("prometheus_multiproc_dir", raising=False)
    logger = _logger_with_mock_team_gauges()

    _set_team_metrics(logger, _payload_with_headers(dict(ALL_TEAM_HEADERS)))
    _set_team_metrics(logger, _payload_with_headers({}))

    for metric_name in TEAM_RATE_LIMIT_METRICS:
        getattr(logger, metric_name).remove.assert_called_once_with("team-abc", "research", "gpt-4o-mini")
