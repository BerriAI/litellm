"""
EcoLogits observability callback.

Calls the EcoLogits public REST API
(https://api.ecologits.ai/v1beta/estimations) on every successful LLM call,
attaches the impacts response under the ``ecologits`` key in both
``litellm_params["metadata"]`` (consumed by Langfuse) and
``standard_logging_object["metadata"]`` (consumed by Datadog, SpendLogs,
and other downstream loggers), and increments Prometheus Counters for
each impact value so users can graph energy, CO2eq, water, etc., for example, in
Grafana the same way they graph token counts.

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
the field is omitted and EcoLogits defaults to ``"WOR"``.

Failures (timeout, non-200, bad payload) NEVER propagate — at worst the call
is logged without ecologits enrichment.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from prometheus_client import Counter

ECOLOGITS_DEFAULT_API_BASE = "https://api.ecologits.ai"
ECOLOGITS_ESTIMATIONS_PATH = "/v1beta/estimations"
ECOLOGITS_DEFAULT_TIMEOUT_S = 2.0
ECOLOGITS_ZONE_METADATA_KEY = "ecologits_electricity_mix_zone"
ECOLOGITS_KWARGS_RESULT_KEY = "ecologits"
ECOLOGITS_DEFAULT_ZONE_LABEL = "WOR"

ECOLOGITS_CONVERSION_MODEL_NAME = {
    "mistral": "mistralai",
}

ECOLOGITS_PROMETHEUS_LABELS = (
    "model",
    "custom_llm_provider",
    "electricity_mix_zone",
    "bound",  # one of "value" | "min" | "max" — see ECOLOGITS_BOUND_KEYS
)

# EcoLogits returns either {"value": x} for deterministic models or
# {"min": x, "max": y} for models with uncertain parameters. We emit one
# series per bound that's actually present in the response.
ECOLOGITS_BOUND_KEYS = ("value", "min", "max")

# Impacts reported by the EcoLogits API. Each entry is (json_key, prom_metric_suffix, unit, description).
ECOLOGITS_IMPACTS = (
    ("energy", "energy_kwh", "kWh", "electrical energy consumed"),
    ("gwp", "gwp_kgco2eq", "kgCO2eq", "global warming potential"),
    ("adpe", "adpe_kgsbeq", "kgSbeq", "abiotic depletion potential (elements)"),
    ("pe", "pe_mj", "MJ", "primary energy consumed"),
    ("wcf", "wcf_l", "L", "water consumption footprint"),
)


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
        api_base: str | None = None,
        default_electricity_mix_zone: str | None = None,
        timeout: float | None = None,
    ) -> None:
        super().__init__()
        self.api_base = (
            api_base or os.getenv("ECOLOGITS_API_BASE") or ECOLOGITS_DEFAULT_API_BASE
        )
        self.default_electricity_mix_zone = default_electricity_mix_zone or os.getenv(
            "ECOLOGITS_ELECTRICITY_MIX_ZONE"
        )
        self.timeout = timeout if timeout is not None else ECOLOGITS_DEFAULT_TIMEOUT_S
        self.async_http_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )

    async def async_logging_hook(
        self, kwargs: dict, result: object, call_type: str
    ) -> tuple[dict, object]:
        try:
            # Respect the `no-log` gate. This hook runs in the FIRST loop of
            # async_log_success_event, BEFORE should_run_callback() applies the
            # no-log gate to the regular success handlers, so we must re-apply
            # it here — otherwise a no-log request would still POST model,
            # provider, token count, latency, and zone to the EcoLogits API.
            if (
                kwargs.get("litellm_params", {}).get("no-log") is True
                and not litellm.global_disable_no_log_param
            ):
                return kwargs, result

            payload = self._build_payload(kwargs=kwargs, result=result)
            if payload is None:
                return kwargs, result

            response = await self.async_http_handler.post(
                url=f"{self.api_base.rstrip('/')}{ECOLOGITS_ESTIMATIONS_PATH}",
                json=payload,
                timeout=self.timeout,
            )

            if response.status_code != 200:
                verbose_logger.warning(
                    "EcoLogits: estimations endpoint returned status %s for "
                    "model=%s provider=%s",
                    response.status_code,
                    payload.get("model_name"),
                    payload.get("provider"),
                )
                return kwargs, result

            ecologits_data = response.json()
            ecologits_data["ecologits_payload"] = payload
            self._attach_to_metadata(kwargs, ecologits_data)
            self._emit_prometheus_counters(
                ecologits_data=ecologits_data,
                model=payload.get("model_name"),
                custom_llm_provider=kwargs.get("custom_llm_provider"),
                zone=payload.get("electricity_mix_zone")
                or ECOLOGITS_DEFAULT_ZONE_LABEL,
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
        """Place the impacts payload where downstream loggers actually look.

        Langfuse reads from ``kwargs["litellm_params"]["metadata"]`` and
        Datadog/SpendLogs read from ``kwargs["standard_logging_object"]["metadata"]``.
        Writing to the top level of kwargs (the previous behaviour) was only
        visible to OTEL, which dumps the raw kwargs.
        """
        litellm_params = kwargs.setdefault("litellm_params", {})
        metadata = litellm_params.get("metadata") or {}
        metadata[ECOLOGITS_KWARGS_RESULT_KEY] = ecologits_data
        litellm_params["metadata"] = metadata

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

        EcoLogits returns two shapes depending on whether the model has
        deterministic impact factors or a range:

        * deterministic: ``{"value": 1.3e-5, "unit": "kWh"}``
        * range:         ``{"value": {"min": 1e-5, "max": 2e-5}, "unit": "kWh"}``

        Flat top-level ``min`` / ``max`` siblings of ``value`` are also
        tolerated, for an API revision that exposes the range that way. The
        fallback runs only when ``value`` produced no bounds, so an entry
        carrying both shapes never double-counts.
        """
        raw_value = entry.get("value")
        if isinstance(raw_value, (int, float)):
            yield "value", raw_value
        elif isinstance(raw_value, dict):
            for bound in ("min", "max"):
                if bound in raw_value:
                    yield bound, raw_value[bound]
        else:
            for bound in ("min", "max"):
                if bound in entry:
                    yield bound, entry[bound]

    def _build_payload(self, kwargs: dict, result: object) -> dict | None:
        provider = kwargs.get("custom_llm_provider") or kwargs.get(
            "litellm_params", {}
        ).get("custom_llm_provider")

        model = kwargs.get("model")

        output_token_count = self._extract_completion_tokens(result)
        request_latency = self._extract_request_latency_seconds(kwargs)

        if (
            not provider
            or not model
            or output_token_count is None
            or request_latency is None
        ):
            return None

        ecologits_provider = ECOLOGITS_CONVERSION_MODEL_NAME.get(provider, provider)

        payload: dict = {
            "provider": ecologits_provider,
            "model_name": model,
            "output_token_count": int(output_token_count),
            "request_latency": float(request_latency),
        }

        zone = self._resolve_zone(kwargs)
        if zone:
            payload["electricity_mix_zone"] = zone

        return payload

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
           ``EcoLogitsLogger`` construction, or omitted entirely (EcoLogits
           then falls back to ``"WOR"``).
        """
        model_info = self._extract_model_info(kwargs)
        zone = model_info.get(ECOLOGITS_ZONE_METADATA_KEY)
        if zone:
            return zone
        return self.default_electricity_mix_zone

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
    def _extract_completion_tokens(result: object) -> int | None:
        if result is None:
            return None
        usage = None
        if isinstance(
            result, (litellm.ModelResponse, litellm.EmbeddingResponse)
        ) and hasattr(result, "usage"):
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
