import datetime
from typing import Any, Dict, Final, List, Tuple

import pytest
from prometheus_client import REGISTRY

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.ecologits import (
    ECOLOGITS_IMPACT_FIELDS,
    ECOLOGITS_KWARGS_RESULT_KEY,
    ECOLOGITS_ZONE_METADATA_KEY,
    EcoLogitsLogger,
    EcoLogitsSdk,
)


@pytest.fixture(autouse=True)
def _clear_ecologits_env(monkeypatch):
    """Keep tests deterministic regardless of the developer's shell env."""
    monkeypatch.delenv("ECOLOGITS_ELECTRICITY_MIX_ZONE", raising=False)


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


def _make_response(completion_tokens: int = 42, model: str | None = None) -> litellm.ModelResponse:
    response = litellm.ModelResponse()
    if model is not None:
        response.model = model
    response.usage = litellm.Usage(
        prompt_tokens=10,
        completion_tokens=completion_tokens,
        total_tokens=10 + completion_tokens,
    )
    return response


def _counter_value(metric_name: str, **labels: str) -> float:
    """Read the current Counter sample value for one specific labelset.

    Counters are global singletons across the test session, so individual
    tests must measure the delta (after - before) rather than the absolute
    value.
    """
    return REGISTRY.get_sample_value(metric_name, labels) or 0.0


def _stub_sdk(llm_impacts) -> EcoLogitsSdk:
    """A stubbed SDK bundle around a caller-provided llm_impacts function."""
    from ecologits.model_repository import models
    from ecologits.tracers.utils import PROVIDER_CONFIG_MAP

    return EcoLogitsSdk(llm_impacts=llm_impacts, models=models, provider_config=PROVIDER_CONFIG_MAP)


# ---------------------------------------------------------------------------
# Payload building
# ---------------------------------------------------------------------------


def test_build_payload_extracts_required_fields():
    logger = EcoLogitsLogger()
    kwargs = _make_kwargs(model="claude-sonnet-4-5", provider="anthropic")
    response = _make_response(completion_tokens=128)

    payload = logger._build_payload(kwargs=kwargs, result=response, sdk=logger._sdk)

    assert payload is not None
    assert payload.provider == "anthropic"
    assert payload.model_name == "claude-sonnet-4-5"
    assert payload.output_token_count == 128
    assert payload.request_latency == pytest.approx(0.75)
    assert payload.to_dict() == {
        "provider": "anthropic",
        "model_name": "claude-sonnet-4-5",
        "output_token_count": 128,
        "request_latency": 0.75,
    }


def test_build_payload_maps_litellm_provider_to_ecologits_provider():
    logger = EcoLogitsLogger()
    kwargs = _make_kwargs(model="codestral-latest", provider="mistral")

    payload = logger._build_payload(kwargs=kwargs, result=_make_response(), sdk=logger._sdk)

    assert payload is not None
    assert payload.provider == "mistralai"
    assert payload.model_name == "codestral-latest"


def test_build_payload_strips_registered_provider_prefix():
    """litellm model strings carry a provider prefix; only that exact segment
    is stripped. Hub ids with their own slashes keep their first segment."""
    logger = EcoLogitsLogger()
    prefixed = logger._build_payload(
        kwargs=_make_kwargs(model="openai/gpt-4o"), result=_make_response(), sdk=logger._sdk
    )
    hub_prefixed = logger._build_payload(
        kwargs=_make_kwargs(model="huggingface/databricks/dbrx-instruct", provider="huggingface"),
        result=_make_response(),
        sdk=logger._sdk,
    )
    bare_hub_id = logger._build_payload(
        kwargs=_make_kwargs(model="databricks/dbrx-instruct", provider="huggingface"),
        result=_make_response(),
        sdk=logger._sdk,
    )

    assert prefixed is not None and prefixed.model_name == "gpt-4o"
    assert hub_prefixed is not None and hub_prefixed.model_name == "databricks/dbrx-instruct"
    assert bare_hub_id is not None and bare_hub_id.model_name == "databricks/dbrx-instruct"


def test_build_payload_falls_back_to_response_model_id():
    """When kwargs["model"] carries a router alias the model id echoed on the
    response can still resolve."""
    logger = EcoLogitsLogger()
    kwargs = _make_kwargs(model="team-finetune-alias")

    payload = logger._build_payload(kwargs=kwargs, result=_make_response(model="gpt-4o"), sdk=logger._sdk)

    assert payload is not None
    assert payload.model_name == "gpt-4o"


def test_build_payload_returns_none_for_unmapped_provider():
    """Marketplace/reseller providers are skipped: the SDK registry only
    carries first-party vendor architectures."""
    logger = EcoLogitsLogger()

    payload = logger._build_payload(kwargs=_make_kwargs(provider="bedrock"), result=_make_response(), sdk=logger._sdk)

    assert payload is None


def test_build_payload_returns_none_when_provider_missing():
    logger = EcoLogitsLogger()
    kwargs = _make_kwargs(provider="")
    kwargs["custom_llm_provider"] = None

    payload = logger._build_payload(kwargs=kwargs, result=_make_response(), sdk=logger._sdk)

    assert payload is None


def test_build_payload_returns_none_when_response_lacks_usage():
    logger = EcoLogitsLogger()
    response = litellm.ModelResponse()  # no usage assigned

    payload = logger._build_payload(kwargs=_make_kwargs(), result=response, sdk=logger._sdk)

    assert payload is None


def test_build_payload_returns_none_when_timing_missing():
    logger = EcoLogitsLogger()
    kwargs = _make_kwargs()
    kwargs.pop("end_time")

    payload = logger._build_payload(kwargs=kwargs, result=_make_response(), sdk=logger._sdk)

    assert payload is None


def test_build_payload_returns_none_when_model_not_in_registry():
    logger = EcoLogitsLogger()

    payload = logger._build_payload(
        kwargs=_make_kwargs(model="gpt-not-real"), result=_make_response(), sdk=logger._sdk
    )

    assert payload is None


# ---------------------------------------------------------------------------
# Zone resolution
# ---------------------------------------------------------------------------


def test_zone_resolution_from_nested_model_info_wins_over_default():
    """The canonical proxy path: model_list[].model_info in YAML lands in
    kwargs["litellm_params"]["metadata"]["model_info"] at hook time."""
    logger = EcoLogitsLogger(default_electricity_mix_zone="WOR")
    kwargs = _make_kwargs(model_info={ECOLOGITS_ZONE_METADATA_KEY: "FRA"})

    payload = logger._build_payload(kwargs=kwargs, result=_make_response(), sdk=logger._sdk)

    assert payload is not None
    assert payload.electricity_mix_zone == "FRA"


def test_zone_resolution_from_top_level_model_info_also_works():
    """Some call paths populate kwargs["model_info"] directly — both paths
    must resolve."""
    logger = EcoLogitsLogger(default_electricity_mix_zone="WOR")
    kwargs = _make_kwargs(top_level_model_info={ECOLOGITS_ZONE_METADATA_KEY: "DEU"})

    payload = logger._build_payload(kwargs=kwargs, result=_make_response(), sdk=logger._sdk)

    assert payload is not None
    assert payload.electricity_mix_zone == "DEU"


def test_zone_resolution_top_level_wins_when_both_populated():
    """If both paths carry the key, the top-level one takes precedence."""
    logger = EcoLogitsLogger(default_electricity_mix_zone="WOR")
    kwargs = _make_kwargs(
        model_info={ECOLOGITS_ZONE_METADATA_KEY: "ESP"},
        top_level_model_info={ECOLOGITS_ZONE_METADATA_KEY: "FRA"},
    )

    payload = logger._build_payload(kwargs=kwargs, result=_make_response(), sdk=logger._sdk)

    assert payload is not None
    assert payload.electricity_mix_zone == "FRA"


def test_zone_resolution_falls_back_to_env_var(monkeypatch):
    monkeypatch.setenv("ECOLOGITS_ELECTRICITY_MIX_ZONE", "DEU")
    logger = EcoLogitsLogger()

    payload = logger._build_payload(kwargs=_make_kwargs(), result=_make_response(), sdk=logger._sdk)

    assert payload is not None
    assert payload.electricity_mix_zone == "DEU"


def test_zone_omitted_when_neither_model_info_nor_env_set(monkeypatch):
    monkeypatch.delenv("ECOLOGITS_ELECTRICITY_MIX_ZONE", raising=False)
    logger = EcoLogitsLogger()

    payload = logger._build_payload(kwargs=_make_kwargs(), result=_make_response(), sdk=logger._sdk)

    assert payload is not None
    assert payload.electricity_mix_zone is None
    assert "electricity_mix_zone" not in payload.to_dict()


def test_zone_ignored_when_set_only_under_litellm_params_metadata():
    """Setting the zone directly in litellm_params.metadata (not under
    model_info) is intentionally NOT recognized — zone is a deployment
    property, not a caller-controlled override, for now."""
    logger = EcoLogitsLogger(default_electricity_mix_zone="WOR")
    kwargs = _make_kwargs(metadata={ECOLOGITS_ZONE_METADATA_KEY: "ITA"})

    payload = logger._build_payload(kwargs=kwargs, result=_make_response(), sdk=logger._sdk)

    assert payload is not None
    assert payload.electricity_mix_zone == "WOR"


# ---------------------------------------------------------------------------
# End-to-end enrichment (real SDK engine, bundled data)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_logging_hook_enriches_litellm_params_metadata_on_success():
    """Langfuse reads from litellm_params.metadata, so that's where the
    impacts payload must land. The top-level kwargs slot is intentionally
    NOT used (only OTEL ever saw it there)."""
    logger = EcoLogitsLogger()

    kwargs = _make_kwargs(model_info={ECOLOGITS_ZONE_METADATA_KEY: "FRA"})
    result = _make_response(completion_tokens=64)

    new_kwargs, new_result = await logger.async_logging_hook(
        kwargs=kwargs, result=result, call_type="completion"
    )

    assert new_result is result
    # Top-level slot must be empty — that was the pre-fix bug for Langfuse.
    assert ECOLOGITS_KWARGS_RESULT_KEY not in new_kwargs
    enriched = new_kwargs["litellm_params"]["metadata"][ECOLOGITS_KWARGS_RESULT_KEY]
    assert enriched["ecologits_payload"] == {
        "provider": "openai",
        "model_name": "gpt-4o",
        "output_token_count": 64,
        "request_latency": 0.75,
        "electricity_mix_zone": "FRA",
    }
    impacts = enriched["impacts"]
    for field in ECOLOGITS_IMPACT_FIELDS:
        assert field in impacts
    assert isinstance(impacts["energy"]["value"], dict)
    assert set(impacts["energy"]["value"]) == {"min", "max"}
    assert impacts["energy"]["unit"] == "kWh"


@pytest.mark.asyncio
async def test_async_logging_hook_mirrors_into_standard_logging_object_metadata():
    """Datadog and SpendLogs read from standard_logging_object.metadata, so
    the enrichment must mirror into that dict as well."""
    logger = EcoLogitsLogger()

    slo = {"metadata": {"team_id": "team-foo"}}
    kwargs = _make_kwargs(standard_logging_object=slo)

    new_kwargs, _ = await logger.async_logging_hook(
        kwargs=kwargs, result=_make_response(), call_type="completion"
    )

    slo_metadata = new_kwargs["standard_logging_object"]["metadata"]
    assert ECOLOGITS_KWARGS_RESULT_KEY in slo_metadata
    # Pre-existing entries are not clobbered.
    assert slo_metadata["team_id"] == "team-foo"


@pytest.mark.asyncio
async def test_async_logging_hook_works_when_standard_logging_object_missing():
    """When standard_logging_object is absent (e.g. call_type that skips
    SLO building), the hook must not crash — litellm_params path still
    works."""
    logger = EcoLogitsLogger()

    kwargs = _make_kwargs()
    assert "standard_logging_object" not in kwargs

    new_kwargs, _ = await logger.async_logging_hook(
        kwargs=kwargs, result=_make_response(), call_type="completion"
    )

    assert (
        ECOLOGITS_KWARGS_RESULT_KEY in new_kwargs["litellm_params"]["metadata"]
    )
    assert "standard_logging_object" not in new_kwargs


@pytest.mark.asyncio
async def test_async_logging_hook_preserves_existing_litellm_params_metadata():
    """Enrichment must not overwrite unrelated metadata entries already in
    litellm_params.metadata (e.g. user_api_key_alias, requester_ip)."""
    logger = EcoLogitsLogger()

    kwargs = _make_kwargs(
        metadata={"user_api_key_alias": "alias-1", "requester_ip_address": "1.2.3.4"}
    )

    new_kwargs, _ = await logger.async_logging_hook(
        kwargs=kwargs, result=_make_response(), call_type="completion"
    )

    md = new_kwargs["litellm_params"]["metadata"]
    assert md["user_api_key_alias"] == "alias-1"
    assert md["requester_ip_address"] == "1.2.3.4"
    assert ECOLOGITS_KWARGS_RESULT_KEY in md


# ---------------------------------------------------------------------------
# Prometheus counter emission
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_logging_hook_increments_prometheus_for_range_shape():
    """Range-valued impacts (gpt-4o has undisclosed MoE parameters) produce
    one Counter series per bound — 'min' and 'max' — and no 'value' series."""
    logger = EcoLogitsLogger()

    zone = "FRA"
    common = dict(
        model="gpt-4o",
        custom_llm_provider="openai",
        electricity_mix_zone=zone,
    )
    before = {
        bound: _counter_value("litellm_ecologits_energy_kwh_total", bound=bound, **common)
        for bound in ("value", "min", "max")
    }
    gwp_min_before = _counter_value("litellm_ecologits_gwp_kgco2eq_total", bound="min", **common)

    kwargs = _make_kwargs(model_info={ECOLOGITS_ZONE_METADATA_KEY: zone})
    await logger.async_logging_hook(
        kwargs=kwargs,
        result=_make_response(completion_tokens=64),
        call_type="completion",
    )

    deltas = {
        bound: _counter_value("litellm_ecologits_energy_kwh_total", bound=bound, **common) - before[bound]
        for bound in ("value", "min", "max")
    }
    assert deltas["min"] > 0
    assert deltas["max"] > 0
    assert deltas["min"] != deltas["max"]
    assert deltas["value"] == 0
    gwp_min_after = _counter_value("litellm_ecologits_gwp_kgco2eq_total", bound="min", **common)
    assert gwp_min_after - gwp_min_before > 0


@pytest.mark.asyncio
async def test_async_logging_hook_increments_prometheus_for_deterministic_shape():
    """Deterministic-shape impacts (cohere command-a-03-2025 has disclosed
    dense parameters) produce one series with bound='value' only."""
    logger = EcoLogitsLogger()

    zone = "DEU"
    common = dict(
        model="command-a-03-2025",
        custom_llm_provider="cohere",
        electricity_mix_zone=zone,
    )
    before = {
        bound: _counter_value("litellm_ecologits_energy_kwh_total", bound=bound, **common)
        for bound in ("value", "min", "max")
    }

    kwargs = _make_kwargs(
        model="command-a-03-2025", provider="cohere", model_info={ECOLOGITS_ZONE_METADATA_KEY: zone}
    )
    await logger.async_logging_hook(
        kwargs=kwargs, result=_make_response(), call_type="completion"
    )

    deltas = {
        bound: _counter_value("litellm_ecologits_energy_kwh_total", bound=bound, **common) - before[bound]
        for bound in ("value", "min", "max")
    }
    assert deltas["value"] > 0
    assert deltas["min"] == 0
    assert deltas["max"] == 0


@pytest.mark.asyncio
async def test_async_logging_hook_labels_unconfigured_zone_with_provider_datacenter():
    """With no zone configured the counter is labeled with the provider's
    datacenter zone (openai -> USA), matching the estimate the SDK made."""
    logger = EcoLogitsLogger()

    labels = dict(
        model="gpt-4o",
        custom_llm_provider="openai",
        electricity_mix_zone="USA",
        bound="min",
    )
    before = _counter_value("litellm_ecologits_energy_kwh_total", **labels)

    kwargs = _make_kwargs()  # no model_info zone, no env var (autouse fixture)
    await logger.async_logging_hook(
        kwargs=kwargs, result=_make_response(), call_type="completion"
    )

    assert _counter_value("litellm_ecologits_energy_kwh_total", **labels) - before > 0


def test_iter_impact_bounds_skips_entries_without_numeric_value():
    """An impact entry that has neither value nor min/max yields no series —
    guards against bogus 0-valued samples."""
    assert list(EcoLogitsLogger._iter_impact_bounds({"unit": "kWh"})) == []


# ---------------------------------------------------------------------------
# Skip paths — enrichment must never propagate failures
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_logging_hook_skips_unknown_model_without_enrichment():
    logger = EcoLogitsLogger()

    kwargs = _make_kwargs(model="gpt-not-real")
    new_kwargs, new_result = await logger.async_logging_hook(
        kwargs=kwargs, result=_make_response(), call_type="completion"
    )

    assert ECOLOGITS_KWARGS_RESULT_KEY not in new_kwargs["litellm_params"]["metadata"]
    assert new_result is not None


@pytest.mark.asyncio
async def test_async_logging_hook_skips_unregistered_zone():
    logger = EcoLogitsLogger()

    kwargs = _make_kwargs(model_info={ECOLOGITS_ZONE_METADATA_KEY: "ZZZ"})
    new_kwargs, _ = await logger.async_logging_hook(
        kwargs=kwargs, result=_make_response(), call_type="completion"
    )

    assert ECOLOGITS_KWARGS_RESULT_KEY not in new_kwargs["litellm_params"]["metadata"]


@pytest.mark.asyncio
async def test_async_logging_hook_swallows_engine_exceptions():
    """If the engine itself raises, the call is logged without enrichment;
    nothing propagates to the caller."""
    def _boom(**_kwargs: Any) -> Any:
        raise RuntimeError("boom")

    logger = EcoLogitsLogger(sdk_loader=lambda: _stub_sdk(_boom))

    new_kwargs, new_result = await logger.async_logging_hook(
        kwargs=_make_kwargs(), result=_make_response(), call_type="completion"
    )

    assert ECOLOGITS_KWARGS_RESULT_KEY not in new_kwargs["litellm_params"]["metadata"]
    assert new_result is not None


@pytest.mark.asyncio
async def test_async_logging_hook_is_noop_without_sdk():
    """Without the ecologits package the callback stays constructible (so
    registration never breaks) and is a plain pass-through."""
    logger = EcoLogitsLogger(sdk_loader=lambda: None)

    kwargs = _make_kwargs()
    result = _make_response()

    new_kwargs, new_result = await logger.async_logging_hook(
        kwargs=kwargs, result=result, call_type="completion"
    )

    assert new_kwargs is kwargs
    assert new_result is result


@pytest.mark.asyncio
async def test_async_logging_hook_skips_when_payload_incomplete():
    logger = EcoLogitsLogger()

    kwargs = _make_kwargs()
    kwargs.pop("end_time")  # makes latency unresolvable

    new_kwargs, _ = await logger.async_logging_hook(
        kwargs=kwargs, result=_make_response(), call_type="completion"
    )

    assert ECOLOGITS_KWARGS_RESULT_KEY not in new_kwargs["litellm_params"]["metadata"]


# ---------------------------------------------------------------------------
# no-log gate — a no-log request must not carry data into the enrichment
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_logging_hook_respects_no_log_flag():
    """A request marked ``litellm_params["no-log"] = True`` must not be
    enriched at all.

    ``async_logging_hook`` runs in the first loop of
    ``async_log_success_event`` — before ``should_run_callback()`` applies the
    no-log gate to the regular success handlers — so the gate has to be
    re-applied inside the hook itself.
    """
    logger = EcoLogitsLogger()

    kwargs = _make_kwargs()
    kwargs["litellm_params"]["no-log"] = True
    result = _make_response()

    new_kwargs, new_result = await logger.async_logging_hook(
        kwargs=kwargs, result=result, call_type="completion"
    )

    assert ECOLOGITS_KWARGS_RESULT_KEY not in new_kwargs["litellm_params"]["metadata"]
    assert new_result is result


@pytest.mark.asyncio
async def test_async_logging_hook_ignores_no_log_when_globally_disabled(monkeypatch):
    """``litellm.global_disable_no_log_param`` is the admin override that makes
    LiteLLM ignore per-request no-log flags. When it's set, a no-log request
    must still be enriched — mirroring ``should_run_callback``."""
    monkeypatch.setattr(litellm, "global_disable_no_log_param", True)
    logger = EcoLogitsLogger()

    kwargs = _make_kwargs()
    kwargs["litellm_params"]["no-log"] = True

    new_kwargs, _ = await logger.async_logging_hook(
        kwargs=kwargs, result=_make_response(), call_type="completion"
    )

    assert ECOLOGITS_KWARGS_RESULT_KEY in new_kwargs["litellm_params"]["metadata"]


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

    ecologits_logger = EcoLogitsLogger()

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
    assert enriched is not None
    assert "impacts" in enriched and "energy" in enriched["impacts"]


# ---------------------------------------------------------------------------
# Payload metadata bucket attachment
# ---------------------------------------------------------------------------


_ATTACH_IMPACTS: Final = {
    "ecologits_payload": {"model_name": "claude-haiku-4-5", "output_token_count": 42},
    "impacts": {"energy": {"value": 1.3e-05, "unit": "kWh"}},
}


def test_attach_writes_to_metadata_bucket():
    kwargs = {"litellm_params": {"metadata": {"user_api_key": "k"}}}
    EcoLogitsLogger._attach_to_metadata(kwargs, _ATTACH_IMPACTS)
    assert kwargs["litellm_params"]["metadata"][ECOLOGITS_KWARGS_RESULT_KEY] == _ATTACH_IMPACTS


def test_attach_also_writes_to_litellm_metadata_when_present():
    """When a caller (e.g. Claude Code CLI) sends litellm_metadata, get_litellm_metadata_from_kwargs
    returns litellm_params["litellm_metadata"], not litellm_params["metadata"].  The spend-log pipeline
    reads through that helper, so the hook must write to both buckets."""
    kwargs = {
        "litellm_params": {
            "metadata": {"user_api_key": "k"},
            "litellm_metadata": {"user_api_key": "k", "trace_id": "t"},
        }
    }
    EcoLogitsLogger._attach_to_metadata(kwargs, _ATTACH_IMPACTS)
    assert kwargs["litellm_params"]["litellm_metadata"][ECOLOGITS_KWARGS_RESULT_KEY] == _ATTACH_IMPACTS
    assert kwargs["litellm_params"]["metadata"][ECOLOGITS_KWARGS_RESULT_KEY] == _ATTACH_IMPACTS


def test_attach_does_not_create_litellm_metadata_when_absent():
    kwargs = {"litellm_params": {"metadata": {"user_api_key": "k"}}}
    EcoLogitsLogger._attach_to_metadata(kwargs, _ATTACH_IMPACTS)
    assert "litellm_metadata" not in kwargs["litellm_params"]


def test_attach_propagates_to_standard_logging_object():
    kwargs = {
        "litellm_params": {"metadata": {"user_api_key": "k"}},
        "standard_logging_object": {"metadata": {}},
    }
    EcoLogitsLogger._attach_to_metadata(kwargs, _ATTACH_IMPACTS)
    assert kwargs["standard_logging_object"]["metadata"][ECOLOGITS_KWARGS_RESULT_KEY] == _ATTACH_IMPACTS


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
