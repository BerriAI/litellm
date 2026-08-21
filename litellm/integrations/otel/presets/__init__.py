"""Integration presets — each one returns an :class:`OpenTelemetryV2Config`.

A preset is a callable that reads an integration's env vars and returns an
``OpenTelemetryV2Config`` describing the exporter destination, the mapper
vocabularies to apply, and any resource attributes. ``PRESET_BY_CALLBACK``
maps a callback name (``"arize"``, ``"langfuse_otel"``, ...) to its preset so
the factory in ``litellm_logging`` can resolve a name and build a single
``OpenTelemetryV2`` instance from the result.
"""

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

from litellm.integrations.otel.presets.agentops import agentops_preset
from litellm.integrations.otel.presets.arize import arize_dynamic_headers, arize_preset
from litellm.integrations.otel.presets.base import Preset
from litellm.integrations.otel.presets.langfuse import (
    langfuse_dynamic_headers,
    langfuse_preset,
)
from litellm.integrations.otel.presets.langtrace import langtrace_preset
from litellm.integrations.otel.presets.levo import levo_preset
from litellm.integrations.otel.presets.phoenix import (
    phoenix_preset,
    phoenix_project_headers,
)
from litellm.integrations.otel.presets.weave import weave_dynamic_headers, weave_preset
from litellm.types.utils import StandardCallbackDynamicParams

#: Callback name → preset. The ``Preset`` annotation makes mypy verify every
#: registered value matches the preset interface.
PRESET_BY_CALLBACK: Final[dict[str, Preset]] = {
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
DYNAMIC_HEADERS_BY_CALLBACK: Final[dict[str, Callable[[StandardCallbackDynamicParams], dict[str, str]]]] = {
    "arize": arize_dynamic_headers,
    "langfuse_otel": langfuse_dynamic_headers,
    "weave_otel": weave_dynamic_headers,
}


#: Callback name → per-request *routing* header builder, sourced from the key/team
#: config the proxy resolved at auth. Deliberately separate from
#: ``DYNAMIC_HEADERS_BY_CALLBACK``: that one is fed
#: ``StandardCallbackDynamicParams``, which is populated from client-supplied
#: request metadata. Naming a destination project is a data-exfiltration
#: primitive, so it must only ever come from server-set key/team config.
PROJECT_HEADERS_BY_CALLBACK: Final[Mapping[str, Callable[[Mapping[str, str] | None], Mapping[str, str]]]] = (
    MappingProxyType(
        {
            "arize_phoenix": phoenix_project_headers,
        }
    )
)

_NO_PROJECT_HEADERS: Final[Mapping[str, str]] = MappingProxyType({})


def dynamic_otlp_headers(
    callback_name: str | None,
    dynamic_params: StandardCallbackDynamicParams | None,
) -> dict[str, str] | None:
    """Per-request OTLP headers for ``callback_name``, or ``None`` if N/A.

    ``None`` means "no per-request routing" — the caller uses its default tracer.
    """
    builder: Final = DYNAMIC_HEADERS_BY_CALLBACK.get(callback_name or "")
    if builder is None or not dynamic_params:
        return None
    headers: Final = builder(dynamic_params)
    return headers or None


def project_routing_headers(
    callback_name: str | None,
    auth_metadata: Mapping[str, str] | None,
) -> Mapping[str, str]:
    """Per-request project-routing headers from trusted key/team config.

    Empty means "no per-request project" — the caller keeps its default tracer,
    whose resource attributes carry the env-configured project.
    """
    builder: Final = PROJECT_HEADERS_BY_CALLBACK.get(callback_name or "")
    if builder is None:
        return _NO_PROJECT_HEADERS
    return builder(auth_metadata)


__all__ = [
    "DYNAMIC_HEADERS_BY_CALLBACK",
    "PRESET_BY_CALLBACK",
    "PROJECT_HEADERS_BY_CALLBACK",
    "Preset",
    "agentops_preset",
    "arize_preset",
    "dynamic_otlp_headers",
    "langfuse_preset",
    "langtrace_preset",
    "levo_preset",
    "phoenix_preset",
    "project_routing_headers",
    "weave_preset",
]
