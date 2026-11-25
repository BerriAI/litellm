import json
from collections.abc import Iterator, Mapping
from datetime import date, datetime, timezone
from typing import Final

import pytest
from prometheus_client import REGISTRY
from prometheus_client.samples import Sample

import litellm
from litellm import Router
from litellm.integrations.prometheus import PrometheusLogger, _deployment_deprecation_date, _deprecated_label
from litellm.litellm_core_utils import get_model_cost_map
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.proxy import proxy_server
from litellm.proxy.common_utils.model_deprecation import is_model_deprecated
from litellm.router_api import ChangeSet
from litellm.rust_bridge._native import NativeInstalled, NativeModelCostMap
from litellm.types.integrations.prometheus import DEFINED_PROMETHEUS_METRICS, PrometheusMetricLabels
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.router import Deployment

_TODAY: Final = date(2026, 6, 15)
_NOW: Final = datetime(2026, 6, 15, 12, tzinfo=timezone.utc)
_METRICS: Final[tuple[DEFINED_PROMETHEUS_METRICS, ...]] = (
    "litellm_proxy_total_requests_metric",
    "litellm_proxy_failed_requests_metric",
    "litellm_requests_metric",
)


@pytest.fixture(autouse=True)
def isolated_metrics(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    previous: Final = tuple(REGISTRY._collector_to_names)
    for collector in previous:
        REGISTRY.unregister(collector)
    monkeypatch.setattr(litellm, "prometheus_emit_deprecated_label", False)
    yield
    for collector in tuple(REGISTRY._collector_to_names):
        REGISTRY.unregister(collector)
    for collector in previous:
        REGISTRY.register(collector)


def _deployment(identity: str, deprecation_date: str | None, group: str = "group") -> Deployment:
    return Deployment(
        model_name=group,
        litellm_params={"model": "openai/deprecated-label-synthetic", "api_key": "synthetic-key"},
        model_info={"id": identity, "deprecation_date": deprecation_date},
    )


def _router(*deployments: Deployment) -> Router:
    return Router(model_list=[deployment.to_json(exclude_none=True) for deployment in deployments])


@pytest.mark.parametrize("metric", _METRICS)
def test_deprecated_label_is_opt_in(monkeypatch: pytest.MonkeyPatch, metric: DEFINED_PROMETHEUS_METRICS) -> None:
    original: Final = PrometheusMetricLabels.get_labels(metric)
    assert "deprecated" not in original
    monkeypatch.setattr(litellm, "prometheus_emit_deprecated_label", True)
    enabled: Final = PrometheusMetricLabels.get_labels(metric)
    assert enabled.count("deprecated") == 1
    assert [label for label in enabled if label != "deprecated"] == original


def test_flag_does_not_change_unrelated_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    original: Final = PrometheusMetricLabels.get_labels("litellm_spend_metric")
    monkeypatch.setattr(litellm, "prometheus_emit_deprecated_label", True)
    assert PrometheusMetricLabels.get_labels("litellm_spend_metric") == original


@pytest.mark.parametrize("metric", _METRICS)
def test_existing_custom_deprecated_label_is_not_duplicated(
    monkeypatch: pytest.MonkeyPatch, metric: DEFINED_PROMETHEUS_METRICS
) -> None:
    monkeypatch.setattr(litellm, "custom_prometheus_metadata_labels", ["deprecated"])
    monkeypatch.setattr(litellm, "prometheus_emit_deprecated_label", True)
    assert PrometheusMetricLabels.get_labels(metric).count("deprecated") == 1


@pytest.mark.parametrize(
    "deprecation_date, expected",
    ((None, False), (date(2026, 6, 14), True), (_TODAY, False), (date(2026, 6, 16), False)),
)
def test_date_boundary_matches_deprecations_endpoint(deprecation_date: date | None, expected: bool) -> None:
    assert is_model_deprecated(deprecation_date, _TODAY) is expected


@pytest.mark.parametrize("override, expected", (("2026-06-14", "true"), ("2026-06-15", "false"), (None, "false")))
def test_deployment_date_controls_label(override: str | None, expected: str) -> None:
    router: Final = _router(_deployment("dated", override))
    assert _deprecated_label("dated", "group", router, _TODAY) == expected


@pytest.mark.parametrize(
    "override, catalog_date, expected",
    (("2000-01-01", "2999-01-01", "true"), ("2999-01-01", "2000-01-01", "false")),
)
def test_override_beats_catalog(
    monkeypatch: pytest.MonkeyPatch, override: str, catalog_date: str, expected: str
) -> None:
    catalog: Final = NativeModelCostMap.bundled(min_model_count=0, max_shrink_ratio=0)
    monkeypatch.setattr(get_model_cost_map, "model_cost_inputs", catalog)
    installed: Final = catalog.install(
        json.dumps({"openai/deprecated-label-synthetic": {"deprecation_date": catalog_date}}).encode()
    )
    assert isinstance(installed, NativeInstalled)
    assert (
        GetModelCostMap.loaded_model_cost_map()["openai/deprecated-label-synthetic"]["deprecation_date"] == catalog_date
    )
    monkeypatch.setattr(litellm, "model_cost", installed.rows)
    deployment: Final = _deployment("override", override).to_json(exclude_none=True)
    assert is_model_deprecated(_deployment_deprecation_date(deployment), _TODAY) is (expected == "true")


def test_cached_missing_date_tracks_catalog_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog: Final = NativeModelCostMap.bundled(min_model_count=0, max_shrink_ratio=0)
    monkeypatch.setattr(get_model_cost_map, "model_cost_inputs", catalog)
    deployment: Final = _deployment("catalog-reload", None).to_json(exclude_none=True)
    assert _deployment_deprecation_date(deployment) is None
    installed: Final = catalog.install(
        json.dumps({"openai/deprecated-label-synthetic": {"deprecation_date": "2000-01-01"}}).encode()
    )
    assert isinstance(installed, NativeInstalled)
    monkeypatch.setattr(litellm, "model_cost", installed.rows)
    assert _deployment_deprecation_date(deployment) == date(2000, 1, 1)


def test_cached_date_flips_at_midnight_and_tracks_deployment_edits() -> None:
    router: Final = _router(_deployment("edited", "2026-06-15"))
    assert _deprecated_label("edited", "group", router, _TODAY) == "false"
    assert _deprecated_label("edited", "group", router, date(2026, 6, 16)) == "true"
    router.apply(ChangeSet(upsert=(_deployment("edited", "2999-01-01"),)))
    assert _deprecated_label("edited", "group", router, date(2026, 6, 16)) == "false"


@pytest.mark.parametrize("second_date, expected", (("2000-01-01", "true"), (None, "false")))
def test_pre_routing_failure_requires_every_deployment_deprecated(second_date: str | None, expected: str) -> None:
    router: Final = _router(_deployment("first", "2000-01-01"), _deployment("second", second_date))
    assert _deprecated_label(None, "group", router, _TODAY) == expected


@pytest.mark.parametrize("identity, group", ((None, "unknown"), ("missing-id", "group"), (None, "")))
def test_unknown_deployments_are_not_deprecated(identity: str | None, group: str) -> None:
    router: Final = _router(_deployment("known", "2000-01-01"))
    assert _deprecated_label(identity, group, router, _TODAY) == "false"


def test_missing_router_is_not_deprecated() -> None:
    assert _deprecated_label("known", "group", None, _TODAY) == "false"


def _success_kwargs(deprecated_metadata: str | None = None) -> dict[str, object]:
    return {
        "model": "deprecated-label-synthetic",
        "litellm_params": {"metadata": {}},
        "start_time": _NOW,
        "end_time": _NOW,
        "standard_logging_object": {
            "id": "deprecation-success",
            "call_type": "completion",
            "status": "success",
            "response_cost": 0,
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
            "startTime": 1.0,
            "endTime": 2.0,
            "completionStartTime": 1.5,
            "model": "deprecated-label-synthetic",
            "model_id": "hook-deployment",
            "model_group": "group",
            "api_base": "https://unused.invalid",
            "custom_llm_provider": "openai",
            "request_tags": [],
            "stream": False,
            "model_parameters": {},
            "hidden_params": {"litellm_overhead_time_ms": None, "additional_headers": None},
            "metadata": {
                "deprecated": deprecated_metadata,
                "user_api_key_hash": "synthetic-hash",
                "user_api_key_alias": "synthetic-alias",
                "user_api_key_team_id": None,
                "user_api_key_team_alias": None,
                "user_api_key_user_id": "synthetic-user",
                "user_api_key_user_email": None,
            },
        },
    }


def _samples(metric: str) -> tuple[Sample, ...]:
    return tuple(
        sample for family in REGISTRY.collect() for sample in family.samples if sample.name == metric + "_total"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", (False, True))
@pytest.mark.parametrize("custom_deprecated", (False, True))
@pytest.mark.parametrize("override, expected", (("2000-01-01", "true"), ("2999-01-01", "false")))
async def test_success_hooks_emit_deprecated_counter_values(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, custom_deprecated: bool, override: str, expected: str
) -> None:
    monkeypatch.setattr(litellm, "prometheus_emit_deprecated_label", enabled)
    monkeypatch.setattr(litellm, "custom_prometheus_metadata_labels", ["deprecated"] if custom_deprecated else [])
    monkeypatch.setattr(proxy_server, "llm_router", _router(_deployment("hook-deployment", override)))
    logger: Final = PrometheusLogger()
    metadata_value: Final = "caller-value" if custom_deprecated else None
    await logger.async_log_success_event(_success_kwargs(metadata_value), None, _NOW, _NOW)
    for metric in ("litellm_proxy_total_requests_metric", "litellm_requests_metric"):
        samples: Final = _samples(metric)
        assert len(samples) == 1
        assert samples[0].value == 1
        assert samples[0].labels["requested_model"] == "group"
        assert samples[0].labels.get("deprecated") == (expected if enabled else metadata_value)
    spend_samples: Final = _samples("litellm_spend_metric")
    assert len(spend_samples) == 1
    assert spend_samples[0].labels.get("deprecated") == metadata_value


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", (False, True))
@pytest.mark.parametrize("routed", (False, True))
@pytest.mark.parametrize("override, expected", (("2000-01-01", "true"), ("2999-01-01", "false")))
async def test_failure_hooks_emit_deprecated_counter_values(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, routed: bool, override: str, expected: str
) -> None:
    monkeypatch.setattr(litellm, "prometheus_emit_deprecated_label", enabled)
    monkeypatch.setattr(proxy_server, "llm_router", _router(_deployment("hook-deployment", override)))
    logger: Final = PrometheusLogger()
    request_data: Final[Mapping[str, object]] = {
        "model": "group",
        "metadata": {"model_info": {"id": "hook-deployment"}} if routed else {},
    }
    await logger.async_post_call_failure_hook(
        request_data=dict(request_data),
        original_exception=Exception("synthetic failure"),
        user_api_key_dict=UserAPIKeyAuth(api_key="synthetic-hash"),
    )
    for metric in ("litellm_proxy_failed_requests_metric", "litellm_proxy_total_requests_metric"):
        samples: Final = _samples(metric)
        assert len(samples) == 1
        assert samples[0].value == 1
        assert samples[0].labels["requested_model"] == "group"
        assert samples[0].labels.get("deprecated") == (expected if enabled else None)
