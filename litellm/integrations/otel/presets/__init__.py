"""Integration presets — each one returns an :class:`OpenTelemetryV2Config`.

A preset is a callable that reads an integration's env vars and returns an
``OpenTelemetryV2Config`` describing the exporter destination, the mapper
vocabularies to apply, and any resource attributes. ``PRESET_BY_CALLBACK``
maps a callback name (``"arize"``, ``"langfuse_otel"``, ...) to its preset so
the factory in ``litellm_logging`` can resolve a name and build a single
``OpenTelemetryV2`` instance from the result.
"""

from typing import Callable

from litellm.integrations.otel.presets.agentops import agentops_preset
from litellm.integrations.otel.presets.arize import arize_dynamic_headers, arize_preset
from litellm.integrations.otel.presets.base import Preset
from litellm.integrations.otel.presets.langfuse import (
    langfuse_dynamic_headers,
    langfuse_preset,
)
from litellm.integrations.otel.presets.langtrace import langtrace_preset
from litellm.integrations.otel.presets.levo import levo_preset
from litellm.integrations.otel.presets.phoenix import phoenix_preset
from litellm.integrations.otel.presets.weave import weave_dynamic_headers, weave_preset
from litellm.types.utils import StandardCallbackDynamicParams

#: Callback name → preset. The ``Preset`` annotation makes mypy verify every
#: registered value matches the preset interface.
PRESET_BY_CALLBACK: dict[str, Preset] = {
    "agentops": agentops_preset,
    "arize": arize_preset,
    "arize_phoenix": phoenix_preset,
    "langfuse_otel": langfuse_preset,
    "langtrace": langtrace_preset,
    "levo": levo_preset,
    "weave_otel": weave_preset,
}

#: Callback name → per-request OTLP header builder (team/key multi-tenant
#: routing). Only integrations that support dynamic credentials appear here —
#: Arize-Phoenix/Langtrace/Levo/AgentOps don't, so they use the logger's
#: default tracer.
DYNAMIC_HEADERS_BY_CALLBACK: dict[str, Callable[[StandardCallbackDynamicParams], dict[str, str]]] = {
    "arize": arize_dynamic_headers,
    "langfuse_otel": langfuse_dynamic_headers,
    "weave_otel": weave_dynamic_headers,
}


def dynamic_otlp_headers(
    callback_name: str | None,
    dynamic_params: StandardCallbackDynamicParams | None,
) -> dict[str, str] | None:
    """Per-request OTLP headers for ``callback_name``, or ``None`` if N/A.

    ``None`` means "no per-request routing" — the caller uses its default tracer.
    """
    builder = DYNAMIC_HEADERS_BY_CALLBACK.get(callback_name or "")
    if builder is None or not dynamic_params:
        return None
    headers = builder(dynamic_params)
    return headers or None


__all__ = [
    "PRESET_BY_CALLBACK",
    "DYNAMIC_HEADERS_BY_CALLBACK",
    "Preset",
    "dynamic_otlp_headers",
    "agentops_preset",
    "arize_preset",
    "langfuse_preset",
    "langtrace_preset",
    "levo_preset",
    "phoenix_preset",
    "weave_preset",
]
