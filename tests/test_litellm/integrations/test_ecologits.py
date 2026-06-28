import datetime
import os
import sys
from typing import Any, Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock

import pytest
from prometheus_client import REGISTRY

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.ecologits import (
    ECOLOGITS_DEFAULT_API_BASE,
    ECOLOGITS_ESTIMATIONS_PATH,
    ECOLOGITS_KWARGS_RESULT_KEY,
    ECOLOGITS_ZONE_METADATA_KEY,
    EcoLogitsLogger,
)


@pytest.fixture(autouse=True)
def _clear_ecologits_env(monkeypatch):
    """Keep tests deterministic regardless of the developer's shell env."""
    monkeypatch.delenv("ECOLOGITS_ELECTRICITY_MIX_ZONE", raising=False)
    monkeypatch.delenv("ECOLOGITS_API_BASE", raising=False)


@pytest.fixture(autouse=True)
def _ensure_ecologits_counters_registered():
    """Re-register the module-global ecologits Counters if a sibling test
    wiped them from the global REGISTRY.

    The ``ecologits`` module registers its Counters once, at import time,
    against the default ``prometheus_client.REGISTRY``. Several sibling
    ``test_prometheus_*.py`` files have an ``autouse`` fixture that clears
    the *entire* REGISTRY (``REGISTRY.unregister(<every collector>)``).
    Under ``make test-unit-integrations`` (``pytest -n 4``) those tests
    interleave with these in the same worker, so by the time a counter
    assertion runs the ecologits Counters may no longer be in REGISTRY —
    ``REGISTRY.get_sample_value`` then returns ``None`` and every measured
    delta collapses to ``0.0``. The Counter objects themselves survive, so
    re-registering the ones that went missing restores observability without
    resetting their accumulated values (the tests measure deltas)."""

    from litellm.integrations.ecologits import _ECOLOGITS_COUNTERS

    for counter in (_ECOLOGITS_COUNTERS or {}).values():
        try:
            REGISTRY.register(counter)
        except ValueError:
            pass
    yield


def _make_kwargs(
    *,
    model: str = "gpt-4o",
    provider: str = "openai",
    metadata: Dict[str, Any] | None = None,
    model_info: Dict[str, Any] | None = None,
    top_level_model_info: Dict[str, Any] | None = None,
    standard_logging_object: Dict[str, Any] | None = None,
    start_time: datetime.datetime | None = None,
    end_time: datetime.datetime | None = None,
) -> Dict[str, Any]:
    """Build a kwargs dict shaped like what the proxy hands to async_logging_hook.

    ``model_info`` is placed under ``litellm_params.metadata.model_info`` —
    the location the proxy router actually populates at logging time.
    ``top_level_model_info`` populates the alternate ``kwargs["model_info"]``
    slot for tests that exercise both lookup paths.
    """
    if start_time is None:
        start_time = datetime.datetime(2026, 5, 5, 12, 0, 0)
    if end_time is None:
        end_time = start_time + datetime.timedelta(milliseconds=750)
    md: Dict[str, Any] = dict(metadata or {})
    if model_info is not None:
        md["model_info"] = model_info
    kwargs: Dict[str, Any] = {
        "model": model,
        "custom_llm_provider": provider,
        "start_time": start_time,
        "end_time": end_time,
        "litellm_params": {"metadata": md},
    }
    if top_level_model_info is not None:
        kwargs["model_info"] = top_level_model_info
    if standard_logging_object is not None:
        kwargs["standard_logging_object"] = standard_logging_object
    return kwargs


def _make_response(completion_tokens: int = 42) -> litellm.ModelResponse:
    response = litellm.ModelResponse()
    response.usage = litellm.Usage(
        prompt_tokens=10,
        completion_tokens=completion_tokens,
        total_tokens=10 + completion_tokens,
    )
    return response


def _impacts_response_payload() -> Dict[str, Any]:
    return {
        "impacts": {
            "energy": {"value": {"min": 0.0001, "max": 0.0005}, "unit": "kWh"},
            "gwp": {"value": {"min": 0.00005, "max": 0.0002}, "unit": "kgCO2eq"},
        },
        "warnings": [],
    }


def _stub_http_response(payload: Dict[str, Any], status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    return resp


def _counter_value(metric_name: str, **labels: str) -> float:
    """Read the current Counter sample value for one specific labelset.

    Counters are global singletons across the test session, so individual
    tests must measure the delta (after - before) rather than the absolute
    value.
    """
    return REGISTRY.get_sample_value(metric_name, labels) or 0.0


# ---------------------------------------------------------------------------
# Payload building
# ---------------------------------------------------------------------------


def test_build_payload_extracts_required_fields():
    logger = EcoLogitsLogger()
    kwargs = _make_kwargs(model="claude-3-5-sonnet", provider="anthropic")
    response = _make_response(completion_tokens=128)

    payload = logger._build_payload(kwargs=kwargs, result=response)

    assert payload == {
        "provider": "anthropic",
        "model_name": "claude-3-5-sonnet",
        "output_token_count": 128,
        "request_latency": 0.75,
    }


def test_build_payload_returns_none_when_provider_missing():
    logger = EcoLogitsLogger()
    kwargs = _make_kwargs(provider="")
    kwargs["custom_llm_provider"] = None

    payload = logger._build_payload(kwargs=kwargs, result=_make_response())

    assert payload is None


def test_build_payload_returns_none_when_response_lacks_usage():
    logger = EcoLogitsLogger()
    response = litellm.ModelResponse()  # no usage assigned

    payload = logger._build_payload(kwargs=_make_kwargs(), result=response)

    assert payload is None


def test_build_payload_returns_none_when_timing_missing():
    logger = EcoLogitsLogger()
    kwargs = _make_kwargs()
    kwargs.pop("end_time")

    payload = logger._build_payload(kwargs=kwargs, result=_make_response())

    assert payload is None


# ---------------------------------------------------------------------------
# Zone resolution
# ---------------------------------------------------------------------------


def test_zone_resolution_from_nested_model_info_wins_over_default():
    """The canonical proxy path: model_list[].model_info in YAML lands in
    kwargs["litellm_params"]["metadata"]["model_info"] at hook time."""
    logger = EcoLogitsLogger(default_electricity_mix_zone="WOR")
    kwargs = _make_kwargs(model_info={ECOLOGITS_ZONE_METADATA_KEY: "FRA"})

    payload = logger._build_payload(kwargs=kwargs, result=_make_response())

    assert payload is not None
    assert payload["electricity_mix_zone"] == "FRA"


def test_zone_resolution_from_top_level_model_info_also_works():
    """Some call paths populate kwargs["model_info"] directly — both paths
    must resolve."""
    logger = EcoLogitsLogger(default_electricity_mix_zone="WOR")
    kwargs = _make_kwargs(top_level_model_info={ECOLOGITS_ZONE_METADATA_KEY: "DEU"})

    payload = logger._build_payload(kwargs=kwargs, result=_make_response())

    assert payload is not None
    assert payload["electricity_mix_zone"] == "DEU"


def test_zone_resolution_top_level_wins_when_both_populated():
    """If both paths carry the key, the top-level one takes precedence."""
    logger = EcoLogitsLogger(default_electricity_mix_zone="WOR")
    kwargs = _make_kwargs(
        model_info={ECOLOGITS_ZONE_METADATA_KEY: "ESP"},
        top_level_model_info={ECOLOGITS_ZONE_METADATA_KEY: "FRA"},
    )

    payload = logger._build_payload(kwargs=kwargs, result=_make_response())

    assert payload["electricity_mix_zone"] == "FRA"


def test_zone_resolution_falls_back_to_env_var(monkeypatch):
    monkeypatch.setenv("ECOLOGITS_ELECTRICITY_MIX_ZONE", "DEU")
    logger = EcoLogitsLogger()

    payload = logger._build_payload(kwargs=_make_kwargs(), result=_make_response())

    assert payload is not None
    assert payload["electricity_mix_zone"] == "DEU"


def test_zone_omitted_when_neither_model_info_nor_env_set(monkeypatch):
    monkeypatch.delenv("ECOLOGITS_ELECTRICITY_MIX_ZONE", raising=False)
    logger = EcoLogitsLogger()

    payload = logger._build_payload(kwargs=_make_kwargs(), result=_make_response())

    assert payload is not None
    assert "electricity_mix_zone" not in payload


def test_zone_ignored_when_set_only_under_litellm_params_metadata():
    """Setting the zone directly in litellm_params.metadata (not under
    model_info) is intentionally NOT recognized — zone is a deployment
    property, not a caller-controlled override, for now."""
    logger = EcoLogitsLogger(default_electricity_mix_zone="WOR")
    kwargs = _make_kwargs(metadata={ECOLOGITS_ZONE_METADATA_KEY: "ITA"})

    payload = logger._build_payload(kwargs=kwargs, result=_make_response())

    assert payload["electricity_mix_zone"] == "WOR"


# ---------------------------------------------------------------------------
# Enrichment placement (Langfuse + SLO consumers)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_logging_hook_enriches_litellm_params_metadata_on_success():
    """Langfuse reads from litellm_params.metadata, so that's where the
    impacts payload must land. The top-level kwargs slot is intentionally
    NOT used (only OTEL ever saw it there)."""
    logger = EcoLogitsLogger()
    impacts = _impacts_response_payload()
    logger.async_http_handler.post = AsyncMock(
        return_value=_stub_http_response(impacts)
    )

    kwargs = _make_kwargs(model_info={ECOLOGITS_ZONE_METADATA_KEY: "FRA"})
    result = _make_response(completion_tokens=64)

    new_kwargs, new_result = await logger.async_logging_hook(
        kwargs=kwargs, result=result, call_type="completion"
    )

    assert new_result is result
    # Top-level slot must be empty — that was the pre-fix bug for Langfuse.
    assert ECOLOGITS_KWARGS_RESULT_KEY not in new_kwargs
    # Nested location is the one Langfuse forwards.
    assert (
        new_kwargs["litellm_params"]["metadata"][ECOLOGITS_KWARGS_RESULT_KEY] == impacts
    )

    logger.async_http_handler.post.assert_awaited_once()
    posted_url = logger.async_http_handler.post.await_args.kwargs["url"]
    posted_json = logger.async_http_handler.post.await_args.kwargs["json"]
    assert posted_url == f"{ECOLOGITS_DEFAULT_API_BASE}{ECOLOGITS_ESTIMATIONS_PATH}"
    assert posted_json["provider"] == "openai"
    assert posted_json["model_name"] == "gpt-4o"
    assert posted_json["output_token_count"] == 64
    assert posted_json["electricity_mix_zone"] == "FRA"
    assert posted_json["request_latency"] == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_async_logging_hook_mirrors_into_standard_logging_object_metadata():
    """Datadog and SpendLogs read from standard_logging_object.metadata, so
    the enrichment must mirror into that dict as well."""
    logger = EcoLogitsLogger()
    impacts = _impacts_response_payload()
    logger.async_http_handler.post = AsyncMock(
        return_value=_stub_http_response(impacts)
    )

    slo = {"metadata": {"team_id": "team-foo"}}
    kwargs = _make_kwargs(standard_logging_object=slo)

    new_kwargs, _ = await logger.async_logging_hook(
        kwargs=kwargs, result=_make_response(), call_type="completion"
    )

    slo_metadata = new_kwargs["standard_logging_object"]["metadata"]
    assert slo_metadata[ECOLOGITS_KWARGS_RESULT_KEY] == impacts
    # Pre-existing entries are not clobbered.
    assert slo_metadata["team_id"] == "team-foo"


@pytest.mark.asyncio
async def test_async_logging_hook_works_when_standard_logging_object_missing():
    """When standard_logging_object is absent (e.g. call_type that skips
    SLO building), the hook must not crash — litellm_params path still
    works."""
    logger = EcoLogitsLogger()
    impacts = _impacts_response_payload()
    logger.async_http_handler.post = AsyncMock(
        return_value=_stub_http_response(impacts)
    )

    kwargs = _make_kwargs()
    assert "standard_logging_object" not in kwargs

    new_kwargs, _ = await logger.async_logging_hook(
        kwargs=kwargs, result=_make_response(), call_type="completion"
    )

    assert (
        new_kwargs["litellm_params"]["metadata"][ECOLOGITS_KWARGS_RESULT_KEY] == impacts
    )
    assert "standard_logging_object" not in new_kwargs


@pytest.mark.asyncio
async def test_async_logging_hook_preserves_existing_litellm_params_metadata():
    """Enrichment must not overwrite unrelated metadata entries already in
    litellm_params.metadata (e.g. user_api_key_alias, requester_ip)."""
    logger = EcoLogitsLogger()
    impacts = _impacts_response_payload()
    logger.async_http_handler.post = AsyncMock(
        return_value=_stub_http_response(impacts)
    )

    kwargs = _make_kwargs(
        metadata={"user_api_key_alias": "alias-1", "requester_ip_address": "1.2.3.4"}
    )

    new_kwargs, _ = await logger.async_logging_hook(
        kwargs=kwargs, result=_make_response(), call_type="completion"
    )

    md = new_kwargs["litellm_params"]["metadata"]
    assert md["user_api_key_alias"] == "alias-1"
    assert md["requester_ip_address"] == "1.2.3.4"
    assert md[ECOLOGITS_KWARGS_RESULT_KEY] == impacts


# ---------------------------------------------------------------------------
# Prometheus counter emission
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_logging_hook_increments_prometheus_for_min_max_shape():
    """Range-shape impacts (value:{min,max}) produce one Counter series per
    bound. Verified by measuring the delta on the global Counter."""
    logger = EcoLogitsLogger()
    impacts = {
        "impacts": {
            "energy": {"value": {"min": 0.001, "max": 0.003}, "unit": "kWh"},
            "gwp": {"value": {"min": 0.0001, "max": 0.0002}, "unit": "kgCO2eq"},
        }
    }
    logger.async_http_handler.post = AsyncMock(
        return_value=_stub_http_response(impacts)
    )

    # Use a zone unique to this test to avoid label-set collisions with
    # other tests that share model+provider.
    zone = "TEST-MINMAX"
    common = dict(
        model="gpt-4o",
        custom_llm_provider="openai",
        electricity_mix_zone=zone,
    )
    energy_min_before = _counter_value(
        "litellm_ecologits_energy_kwh_total", bound="min", **common
    )
    energy_max_before = _counter_value(
        "litellm_ecologits_energy_kwh_total", bound="max", **common
    )
    gwp_min_before = _counter_value(
        "litellm_ecologits_gwp_kgco2eq_total", bound="min", **common
    )

    kwargs = _make_kwargs(model_info={ECOLOGITS_ZONE_METADATA_KEY: zone})
    await logger.async_logging_hook(
        kwargs=kwargs,
        result=_make_response(completion_tokens=64),
        call_type="completion",
    )

    energy_min_after = _counter_value(
        "litellm_ecologits_energy_kwh_total", bound="min", **common
    )
    energy_max_after = _counter_value(
        "litellm_ecologits_energy_kwh_total", bound="max", **common
    )
    gwp_min_after = _counter_value(
        "litellm_ecologits_gwp_kgco2eq_total", bound="min", **common
    )

    assert energy_min_after - energy_min_before == pytest.approx(0.001)
    assert energy_max_after - energy_max_before == pytest.approx(0.003)
    assert gwp_min_after - gwp_min_before == pytest.approx(0.0001)


@pytest.mark.asyncio
async def test_async_logging_hook_increments_prometheus_for_deterministic_shape():
    """Deterministic-shape impacts (flat value:<float>) produce one series
    with bound='value'."""
    logger = EcoLogitsLogger()
    impacts = {
        "impacts": {
            "energy": {"value": 0.005, "unit": "kWh"},
        }
    }
    logger.async_http_handler.post = AsyncMock(
        return_value=_stub_http_response(impacts)
    )

    zone = "TEST-DETERM"
    labels = dict(
        model="gpt-4o",
        custom_llm_provider="openai",
        electricity_mix_zone=zone,
        bound="value",
    )
    before = _counter_value("litellm_ecologits_energy_kwh_total", **labels)

    kwargs = _make_kwargs(model_info={ECOLOGITS_ZONE_METADATA_KEY: zone})
    await logger.async_logging_hook(
        kwargs=kwargs, result=_make_response(), call_type="completion"
    )

    after = _counter_value("litellm_ecologits_energy_kwh_total", **labels)
    assert after - before == pytest.approx(0.005)


@pytest.mark.asyncio
async def test_async_logging_hook_skips_prometheus_when_impact_value_missing():
    """An impact entry that has neither 'value' nor min/max must not emit
    any series — guards against bogus 0-valued samples."""
    logger = EcoLogitsLogger()
    impacts = {
        "impacts": {
            # 'unit' only, no numeric value — should be skipped silently.
            "energy": {"unit": "kWh"},
        }
    }
    logger.async_http_handler.post = AsyncMock(
        return_value=_stub_http_response(impacts)
    )

    zone = "TEST-NOVALUE"
    common = dict(
        model="gpt-4o",
        custom_llm_provider="openai",
        electricity_mix_zone=zone,
    )
    # Snapshot every potential bound to prove none of them get touched.
    before = {
        bound: _counter_value(
            "litellm_ecologits_energy_kwh_total", bound=bound, **common
        )
        for bound in ("value", "min", "max")
    }

    kwargs = _make_kwargs(model_info={ECOLOGITS_ZONE_METADATA_KEY: zone})
    await logger.async_logging_hook(
        kwargs=kwargs, result=_make_response(), call_type="completion"
    )

    for bound, baseline in before.items():
        current = _counter_value(
            "litellm_ecologits_energy_kwh_total", bound=bound, **common
        )
        assert current == baseline, f"bound={bound} should not have changed"


# ---------------------------------------------------------------------------
# Error paths — enrichment must never propagate failures
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_logging_hook_does_not_raise_on_non_200():
    logger = EcoLogitsLogger()
    logger.async_http_handler.post = AsyncMock(
        return_value=_stub_http_response({"detail": "bad request"}, status_code=400)
    )

    kwargs = _make_kwargs()
    result = _make_response()

    new_kwargs, new_result = await logger.async_logging_hook(
        kwargs=kwargs, result=result, call_type="completion"
    )

    assert ECOLOGITS_KWARGS_RESULT_KEY not in new_kwargs["litellm_params"]["metadata"]
    assert new_result is result


@pytest.mark.asyncio
async def test_async_logging_hook_swallows_http_exceptions():
    logger = EcoLogitsLogger()
    logger.async_http_handler.post = AsyncMock(side_effect=RuntimeError("boom"))

    new_kwargs, new_result = await logger.async_logging_hook(
        kwargs=_make_kwargs(),
        result=_make_response(),
        call_type="completion",
    )

    assert ECOLOGITS_KWARGS_RESULT_KEY not in new_kwargs["litellm_params"]["metadata"]
    assert new_result is not None


@pytest.mark.asyncio
async def test_async_logging_hook_skips_api_call_when_payload_incomplete():
    logger = EcoLogitsLogger()
    logger.async_http_handler.post = AsyncMock()

    kwargs = _make_kwargs()
    kwargs.pop("end_time")  # makes latency unresolvable

    await logger.async_logging_hook(
        kwargs=kwargs, result=_make_response(), call_type="completion"
    )

    logger.async_http_handler.post.assert_not_awaited()


# ---------------------------------------------------------------------------
# no-log gate — a no-log request must not reach the external EcoLogits API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_logging_hook_respects_no_log_flag():
    """A request marked ``litellm_params["no-log"] = True`` must not POST any
    metadata to the EcoLogits API.

    ``async_logging_hook`` runs in the first loop of
    ``async_log_success_event`` — before ``should_run_callback()`` applies the
    no-log gate to the regular success handlers — so the gate has to be
    re-applied inside the hook itself.
    """
    logger = EcoLogitsLogger()
    logger.async_http_handler.post = AsyncMock()

    kwargs = _make_kwargs()
    kwargs["litellm_params"]["no-log"] = True
    result = _make_response()

    new_kwargs, new_result = await logger.async_logging_hook(
        kwargs=kwargs, result=result, call_type="completion"
    )

    logger.async_http_handler.post.assert_not_awaited()
    # kwargs/result pass through untouched — no enrichment either.
    assert ECOLOGITS_KWARGS_RESULT_KEY not in new_kwargs["litellm_params"]["metadata"]
    assert new_result is result


@pytest.mark.asyncio
async def test_async_logging_hook_ignores_no_log_when_globally_disabled(monkeypatch):
    """``litellm.global_disable_no_log_param`` is the admin override that makes
    LiteLLM ignore per-request no-log flags. When it's set, a no-log request
    must still be enriched and posted — mirroring ``should_run_callback``.
    """
    monkeypatch.setattr(litellm, "global_disable_no_log_param", True)
    logger = EcoLogitsLogger()
    impacts = _impacts_response_payload()
    logger.async_http_handler.post = AsyncMock(
        return_value=_stub_http_response(impacts)
    )

    kwargs = _make_kwargs()
    kwargs["litellm_params"]["no-log"] = True

    new_kwargs, _ = await logger.async_logging_hook(
        kwargs=kwargs, result=_make_response(), call_type="completion"
    )

    logger.async_http_handler.post.assert_awaited_once()
    assert (
        new_kwargs["litellm_params"]["metadata"][ECOLOGITS_KWARGS_RESULT_KEY] == impacts
    )


# ---------------------------------------------------------------------------
# Pipeline integration — enrichment is visible to downstream callbacks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runs_before_downstream_callbacks_in_pipeline():
    """End-to-end: when EcoLogits is registered first in litellm.callbacks, a
    downstream CustomLogger's async_logging_hook receives the enriched
    metadata. This is the user-facing contract — "enrichment happens before
    all other gateway callbacks" — verified through litellm's actual
    callback pipeline.
    """
    seen_by_downstream: List[Tuple[Dict[str, Any], Any]] = []

    class RecordingDownstreamLogger(CustomLogger):
        async def async_logging_hook(self, kwargs, result, call_type):
            seen_by_downstream.append((dict(kwargs), result))
            return kwargs, result

    impacts = _impacts_response_payload()
    ecologits_logger = EcoLogitsLogger()
    ecologits_logger.async_http_handler.post = AsyncMock(
        return_value=_stub_http_response(impacts)
    )

    original_callbacks = litellm.callbacks
    original_async_success = list(litellm._async_success_callback)
    try:
        downstream = RecordingDownstreamLogger()
        litellm.callbacks = [ecologits_logger, downstream]
        litellm._async_success_callback = [ecologits_logger, downstream]

        from litellm.litellm_core_utils.litellm_logging import Logging

        logging_obj = Logging(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            call_type="completion",
            start_time=datetime.datetime(2026, 5, 5, 12, 0, 0),
            litellm_call_id="test-call-id",
            function_id="fn-id",
        )
        logging_obj.model_call_details = _make_kwargs()

        await logging_obj.async_success_handler(
            result=_make_response(completion_tokens=64),
            start_time=datetime.datetime(2026, 5, 5, 12, 0, 0),
            end_time=datetime.datetime(2026, 5, 5, 12, 0, 0, 750000),
        )
    finally:
        litellm.callbacks = original_callbacks
        litellm._async_success_callback = original_async_success

    assert seen_by_downstream, "downstream logger never received the call"
    downstream_kwargs, _ = seen_by_downstream[-1]
    # Enrichment lives under litellm_params.metadata, not at the top level.
    enriched = downstream_kwargs["litellm_params"]["metadata"].get(
        ECOLOGITS_KWARGS_RESULT_KEY
    )
    assert enriched == impacts


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_string_callback_registered_in_known_list():
    """callbacks: ['ecologits'] in config.yaml must resolve to a CustomLogger."""
    assert "ecologits" in litellm._known_custom_logger_compatible_callbacks

    from litellm.litellm_core_utils.litellm_logging import (
        _init_custom_logger_compatible_class,
    )

    instance = _init_custom_logger_compatible_class(
        logging_integration="ecologits",
        internal_usage_cache=None,
        llm_router=None,
    )

    assert isinstance(instance, EcoLogitsLogger)
