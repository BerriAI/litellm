"""
EcoLogits observability callback.

Computes the environmental footprint of every successful LLM call (energy, GWP,
ADPe, PE, WCF) locally with the ``ecologits`` SDK, attaches the impacts under
the ``ecologits`` key in both ``litellm_params["metadata"]`` (consumed by
Langfuse) and ``standard_logging_object["metadata"]`` (consumed by Datadog,
SpendLogs, and other downstream loggers), and increments Prometheus Counters
for each impact value so users can graph energy, CO2eq, water, etc., for
example, in Grafana the same way they graph token counts.

Everything is computed from the JSON repositories bundled in the SDK; there
are no network calls and no request cancellation can lose enrichment. The
optional package must be installed explicitly:

    pip install "litellm[ecologits]"

EcoLogits' own instrumentation (``EcoLogits.init()``) is deliberately not
used: it monkey-patches the module-level ``litellm.completion`` entrypoints in
consumer apps, which duplicates what a callback already does and would
re-instrument litellm here. Models absent from the SDK's registry and
providers other than the first-party ones in ``ECOLOGITS_PROVIDER_MAP`` are
skipped rather than estimated with wrong architectures' factors.

The hook used is ``async_logging_hook``. ``async_log_success_event`` runs two
sequential loops over the callback list: the first calls ``async_logging_hook``
on every callback, the second calls ``async_log_success_event`` on every
callback. The first loop completes before the second begins, so this
enrichment is in place before any downstream logger (Langfuse, Prometheus,
Datadog, SpendLogs) reads the data in its success handler. Order in the
callbacks list does not matter, though listing ecologits first can help convey
the mental model:

    litellm_settings:
      callbacks: ["ecologits", "prometheus", "langfuse"]

The ``electricity_mix_zone`` request field is a property of the deployment
(it reflects where the model physically runs), so it is set per model in
the proxy YAML under ``model_list[].model_info.ecologits_electricity_mix_zone``.
Falls back to env var ``ECOLOGITS_ELECTRICITY_MIX_ZONE``; if neither is set
the SDK estimates with the provider's datacenter zone (``WOR`` world average
as last resort) and the Prometheus counter is labeled accordingly.

Failures (missing SDK, unregistered model, serialization) NEVER propagate —
at worst the call is logged without ecologits enrichment.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from prometheus_client import Counter

    from ecologits.model_repository import ModelRepository
    from ecologits.tracers.utils import ImpactsOutput, _ProviderConfig as EcoLogitsProviderConfig

ECOLOGITS_ZONE_METADATA_KEY = "ecologits_electricity_mix_zone"
ECOLOGITS_KWARGS_RESULT_KEY = "ecologits"
ECOLOGITS_DEFAULT_ZONE_LABEL = "WOR"

ECOLOGITS_PROMETHEUS_LABELS = (
    "model",
    "custom_llm_provider",
    "electricity_mix_zone",
    "bound",  # one of "value" | "min" | "max"
)

# litellm custom_llm_provider -> ecologits provider name. Strictly first-party
# vendors: the SDK registry only carries vendor-published architectures, so
# marketplaces serving third-party weights (azure_ai, groq, together_ai,
# openrouter, bedrock, ollama, ...) are skipped instead of being billed to
# another vendor's factors. azure (Azure OpenAI) is excluded too because the
# model name alone cannot guarantee OpenAI weights on every deployment.
ECOLOGITS_PROVIDER_MAP: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "mistral": "mistralai",
    "codestral": "mistralai",
    "cohere": "cohere",
    "cohere_chat": "cohere",
    "vertex_ai": "google_genai",
    "vertex_ai_beta": "google_genai",
    "gemini": "google_genai",
    "huggingface": "huggingface_hub",
}

ECOLOGITS_IMPACT_FIELDS: tuple[str, ...] = ("energy", "gwp", "adpe", "pe", "wcf", "usage", "embodied")

# Impacts attached to the metadata. Each entry is (json_key, prom_metric_suffix, unit, description).
ECOLOGITS_IMPACTS = (
    ("energy", "energy_kwh", "kWh", "electrical energy consumed"),
    ("gwp", "gwp_kgco2eq", "kgCO2eq", "global warming potential"),
    ("adpe", "adpe_kgsbeq", "kgSbeq", "abiotic depletion potential (elements)"),
    ("pe", "pe_mj", "MJ", "primary energy consumed"),
    ("wcf", "wcf_l", "L", "water consumption footprint"),
)


class _LLMImpactsFunction(Protocol):
    def __call__(
        self,
        provider: str,
        model_name: str,
        output_token_count: int,
        request_latency: float,
        electricity_mix_zone: str | None = None,
    ) -> ImpactsOutput: ...


@dataclass(frozen=True, slots=True)
class EcoLogitsSdk:
    """The pieces of the ecologits SDK the callback needs, loaded once.

    Injected into ``EcoLogitsLogger`` so tests can stub it, and ``None`` when
    the package is not installed (the callback then logs without enrichment).
    """

    llm_impacts: _LLMImpactsFunction
    models: ModelRepository
    provider_config: Mapping[str, EcoLogitsProviderConfig]


def _load_ecologits_sdk() -> EcoLogitsSdk | None:
    try:
        from ecologits.model_repository import models
        from ecologits.tracers.utils import PROVIDER_CONFIG_MAP, llm_impacts
    except ImportError:
        return None
    return EcoLogitsSdk(llm_impacts=llm_impacts, models=models, provider_config=PROVIDER_CONFIG_MAP)


@dataclass(frozen=True, slots=True)
class _CallPayload:
    provider: str
    model_name: str
    output_token_count: int
    request_latency: float
    electricity_mix_zone: str | None

    def to_dict(self) -> dict:
        payload = {
            "provider": self.provider,
            "model_name": self.model_name,
            "output_token_count": self.output_token_count,
            "request_latency": self.request_latency,
        }
        if self.electricity_mix_zone:
            payload["electricity_mix_zone"] = self.electricity_mix_zone
        return payload


def _build_ecologits_counters() -> dict[str, Counter] | None:
    """Build one Prometheus Counter per impact, or ``None`` if prometheus_client
    is not installed.

    Guarding the import (rather than importing it at module level) means users
    who only want the Langfuse/Datadog enrichment can enable the ``ecologits``
    callback without installing ``prometheus-client``; they simply get no
    Prometheus metrics.
    """
    try:
        from prometheus_client import Counter
    except ImportError:
        verbose_logger.warning(
            "Missing prometheus_client. Run `pip install prometheus-client` to "
            "emit EcoLogits Prometheus metrics. EcoLogits enrichment "
            "(Langfuse/Datadog/SpendLogs) still works without it."
        )
        return None

    counters: dict[str, Counter] = {}
    for json_key, suffix, unit, description in ECOLOGITS_IMPACTS:
        counters[json_key] = Counter(
            name=f"litellm_ecologits_{suffix}_total",
            documentation=f"Cumulative {description} ({unit}) attributed to LLM calls, from EcoLogits.",
            labelnames=ECOLOGITS_PROMETHEUS_LABELS,
        )
    return counters


_ECOLOGITS_COUNTERS = _build_ecologits_counters()


class EcoLogitsLogger(CustomLogger):
    def __init__(
        self,
        default_electricity_mix_zone: str | None = None,
        sdk_loader: Callable[[], EcoLogitsSdk | None] = _load_ecologits_sdk,
    ) -> None:
        super().__init__()
        self.default_electricity_mix_zone = default_electricity_mix_zone or os.getenv("ECOLOGITS_ELECTRICITY_MIX_ZONE")
        self._sdk = sdk_loader()
        if self._sdk is None:
            verbose_logger.warning(
                "Missing ecologits package. Run `pip install 'litellm[ecologits]'` to "
                "emit EcoLogits impact data. Calls will be logged without enrichment."
            )

    async def async_logging_hook(self, kwargs: dict, result: object, call_type: str) -> tuple[dict, object]:
        try:
            # Respect the `no-log` gate. This hook runs in the FIRST loop of
            # async_log_success_event, BEFORE should_run_callback() applies the
            # no-log gate to the regular success handlers, so we must re-apply
            # it here — otherwise a no-log request would still carry model,
            # provider, token count, latency, and zone into the enrichment.
            if kwargs.get("litellm_params", {}).get("no-log") is True and not litellm.global_disable_no_log_param:
                return kwargs, result

            sdk = self._sdk
            if sdk is None:
                return kwargs, result

            payload = self._build_payload(kwargs=kwargs, result=result, sdk=sdk)
            if payload is None:
                return kwargs, result

            output = sdk.llm_impacts(
                provider=payload.provider,
                model_name=payload.model_name,
                output_token_count=payload.output_token_count,
                request_latency=payload.request_latency,
                electricity_mix_zone=payload.electricity_mix_zone,
            )
            if output.has_errors or output.energy is None:
                verbose_logger.debug(
                    "EcoLogits: no impact estimation for model=%s provider=%s errors=%s",
                    payload.model_name,
                    payload.provider,
                    [error.message for error in (output.errors or [])],
                )
                return kwargs, result

            # The SDK engine is pure in-memory arithmetic over its bundled JSON
            # (no I/O), so it runs on the event loop; a thread hop would cost
            # more than the computation itself.
            ecologits_data = self._build_response_dict(payload=payload, output=output)
            self._attach_to_metadata(kwargs, ecologits_data)
            self._emit_prometheus_counters(
                ecologits_data=ecologits_data,
                model=payload.model_name,
                custom_llm_provider=kwargs.get("custom_llm_provider"),
                zone=self._prometheus_zone_label(payload=payload, sdk=sdk),
            )
        except Exception as e:
            verbose_logger.warning(
                "EcoLogits enrichment skipped for model=%s: %s",
                kwargs.get("model"),
                e,
            )

        return kwargs, result

    @staticmethod
    def _attach_to_metadata(kwargs: dict, ecologits_data: object) -> None:
        litellm_params = kwargs.setdefault("litellm_params", {})
        metadata = litellm_params.get("metadata") or {}
        metadata[ECOLOGITS_KWARGS_RESULT_KEY] = ecologits_data
        litellm_params["metadata"] = metadata

        litellm_metadata = litellm_params.get("litellm_metadata")
        if isinstance(litellm_metadata, dict):
            litellm_metadata[ECOLOGITS_KWARGS_RESULT_KEY] = ecologits_data

        standard_logging_object = kwargs.get("standard_logging_object")
        if isinstance(standard_logging_object, dict):
            slo_metadata = standard_logging_object.get("metadata") or {}
            slo_metadata[ECOLOGITS_KWARGS_RESULT_KEY] = ecologits_data
            standard_logging_object["metadata"] = slo_metadata

    @staticmethod
    def _emit_prometheus_counters(
        ecologits_data: object,
        model: str | None,
        custom_llm_provider: str | None,
        zone: str,
    ) -> None:
        if _ECOLOGITS_COUNTERS is None:
            return
        if not isinstance(ecologits_data, dict):
            return
        impacts = ecologits_data.get("impacts")
        if not isinstance(impacts, dict):
            return

        base_labels = {
            "model": model or "",
            "custom_llm_provider": custom_llm_provider or "",
            "electricity_mix_zone": zone,
        }
        for json_key, _suffix, _unit, _doc in ECOLOGITS_IMPACTS:
            entry = impacts.get(json_key)
            if not isinstance(entry, dict):
                continue
            counter = _ECOLOGITS_COUNTERS[json_key]
            for bound, bound_value in EcoLogitsLogger._iter_impact_bounds(entry):
                if not isinstance(bound_value, (int, float)) or bound_value < 0:
                    continue
                counter.labels(**base_labels, bound=bound).inc(float(bound_value))

    @staticmethod
    def _iter_impact_bounds(entry: dict) -> Iterator[tuple[str, object]]:
        """Yield ``(bound, numeric_value)`` pairs from one impact entry.

        Impacts come in two shapes depending on whether the model has
        deterministic factors or a range (models with undisclosed MoE
        parameters, dumped from the SDK's ``RangeValue``):

        * deterministic: ``{"value": 1.3e-5, "unit": "kWh"}``
        * range:         ``{"value": {"min": 1e-5, "max": 2e-5}, "unit": "kWh"}``
        """
        raw_value = entry.get("value")
        if isinstance(raw_value, (int, float)):
            yield "value", raw_value
        elif isinstance(raw_value, dict):
            for bound in ("min", "max"):
                if bound in raw_value:
                    yield bound, raw_value[bound]

    def _build_payload(self, kwargs: dict, result: object, sdk: EcoLogitsSdk) -> _CallPayload | None:
        provider = kwargs.get("custom_llm_provider") or kwargs.get("litellm_params", {}).get("custom_llm_provider")
        ecologits_provider = ECOLOGITS_PROVIDER_MAP.get(provider) if isinstance(provider, str) else None
        if ecologits_provider is None:
            verbose_logger.debug(
                "EcoLogits: provider %s has no first-party match in the SDK registry; skipping enrichment",
                provider,
            )
            return None

        model_name = self._resolve_model_name(
            model=kwargs.get("model"),
            result=result,
            ecologits_provider=ecologits_provider,
            sdk=sdk,
        )
        output_token_count = self._extract_completion_tokens(result)
        request_latency = self._extract_request_latency_seconds(kwargs)

        if not model_name or output_token_count is None or request_latency is None:
            return None

        return _CallPayload(
            provider=ecologits_provider,
            model_name=model_name,
            output_token_count=int(output_token_count),
            request_latency=float(request_latency),
            electricity_mix_zone=self._resolve_zone(kwargs),
        )

    @staticmethod
    def _resolve_model_name(
        model: object,
        result: object,
        ecologits_provider: str,
        sdk: EcoLogitsSdk,
    ) -> str | None:
        """Find the registry name for this call, exact matches only.

        ``kwargs["model"]`` is tried with a single leading segment stripped
        only when that segment names a litellm provider (so ``openai/gpt-4o``
        resolves to ``gpt-4o`` while huggingface hub ids stay whole), then the
        canonical id echoed back on the response. Anything the registry does
        not know exactly is skipped; no fuzzy matching, no date-suffix cuts.
        """
        candidates: tuple[str, ...] = ()
        if isinstance(model, str) and model:
            prefix, sep, base = model.partition("/")
            if sep and prefix in ECOLOGITS_PROVIDER_MAP:
                candidates = (base, model)
            else:
                candidates = (model,)

        response_model = getattr(result, "model", None)
        if isinstance(response_model, str) and response_model not in candidates:
            candidates += (response_model,)

        return next(
            (name for name in candidates if sdk.models.find_model(provider=ecologits_provider, model_name=name) is not None),
            None,
        )

    def _resolve_zone(self, kwargs: dict) -> str | None:
        """Resolve the electricity-mix zone for this call.

        The zone reflects where the model physically runs, so it's a
        property of the deployment, not of the caller. Two sources:

        1. ``model_list[].model_info`` in the proxy YAML — at hook time the
           proxy router has stashed this under
           ``kwargs["litellm_params"]["metadata"]["model_info"]`` (the
           top-level ``kwargs["model_info"]`` is ``None`` on the logging
           path, verified against runtime kwargs dump). Custom keys placed
           directly under ``litellm_params`` get forwarded to the provider
           and rejected (Mistral/OpenAI return 422 on unknown body fields),
           so ``model_info`` is the right home in the YAML.
        2. ``self.default_electricity_mix_zone`` — env-var fallback set at
           ``EcoLogitsLogger`` construction, or omitted entirely (the SDK
           then estimates with the provider's datacenter zone).
        """
        model_info = self._extract_model_info(kwargs)
        zone = model_info.get(ECOLOGITS_ZONE_METADATA_KEY)
        if zone:
            return zone
        return self.default_electricity_mix_zone

    @staticmethod
    def _prometheus_zone_label(payload: _CallPayload, sdk: EcoLogitsSdk) -> str:
        if payload.electricity_mix_zone:
            return payload.electricity_mix_zone
        provider_config = sdk.provider_config.get(payload.provider)
        if provider_config is not None:
            return provider_config.datacenter_location
        return ECOLOGITS_DEFAULT_ZONE_LABEL

    @staticmethod
    def _extract_model_info(kwargs: dict) -> dict:
        """Find the deployment ``model_info`` block.

        Checked in order: top-level ``kwargs["model_info"]`` (some call
        paths populate it), then the canonical location
        ``kwargs["litellm_params"]["metadata"]["model_info"]`` used by the
        proxy router.
        """
        top_level = kwargs.get("model_info")
        if isinstance(top_level, dict) and top_level:
            return top_level

        litellm_params = kwargs.get("litellm_params") or {}
        metadata = litellm_params.get("metadata") or {}
        nested = metadata.get("model_info")
        if isinstance(nested, dict):
            return nested
        return {}

    @staticmethod
    def _build_response_dict(payload: _CallPayload, output: ImpactsOutput) -> dict:
        """Serialize SDK output into the shape downstream consumers already read.

        Range-valued impacts (models with undisclosed parameters) dump as
        ``{"value": {"min": .., "max": ..}, "unit": ..}`` and deterministic
        ones as ``{"value": x, "unit": ..}``; both are what
        ``_iter_impact_bounds`` and the Langfuse/Datadog metadata consumers
        expect.
        """
        data: dict = {
            "ecologits_payload": payload.to_dict(),
            "impacts": {
                field: impact.model_dump(mode="json")
                for field in ECOLOGITS_IMPACT_FIELDS
                if (impact := getattr(output, field)) is not None
            },
        }
        if output.has_warnings:
            data["warnings"] = [warning.model_dump() for warning in output.warnings or []]
        return data

    @staticmethod
    def _extract_completion_tokens(result: object) -> int | None:
        if result is None:
            return None
        usage = None
        if isinstance(result, (litellm.ModelResponse, litellm.EmbeddingResponse)) and hasattr(result, "usage"):
            usage = result["usage"]
        elif isinstance(result, dict):
            usage = result.get("usage")
        else:
            usage = getattr(result, "usage", None)

        if usage is None:
            return None

        if isinstance(usage, dict):
            return usage.get("completion_tokens")
        return getattr(usage, "completion_tokens", None)

    @staticmethod
    def _extract_request_latency_seconds(kwargs: dict) -> float | None:
        start = kwargs.get("start_time")
        end = kwargs.get("end_time") or kwargs.get("completion_start_time")
        if start is None or end is None:
            return None
        try:
            delta = end - start
        except TypeError:
            return None
        if hasattr(delta, "total_seconds"):
            return delta.total_seconds()
        try:
            return float(delta)
        except (TypeError, ValueError):
            return None