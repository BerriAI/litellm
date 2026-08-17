"""Integration presets — each one returns an :class:`OpenTelemetryV2Config`.

A preset is a callable that reads an integration's env vars and returns an
``OpenTelemetryV2Config`` describing the exporter destination, the mapper
vocabularies to apply, and any resource attributes. ``PRESET_BY_CALLBACK``
maps a callback name (``"arize"``, ``"langfuse_otel"``, ...) to its preset so
the factory in ``litellm_logging`` can resolve a name and build a single
``OpenTelemetryV2`` instance from the result.

Admin-owned trace destinations are resolved server-side from a named credential
into an ``OtelDestination`` (see ``litellm.integrations.otel.presets.destinations``
and ``plumbing.routing``). Separately, per-request team/key OTLP credentials from
``standard_callback_dynamic_params`` route the gen-AI span to a credential-scoped
tracer for the integrations that support it; ``dynamic_otlp_headers`` below builds
those per-request headers.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Final

from litellm.integrations.otel.presets.agentops import agentops_preset
from litellm.integrations.otel.presets.arize import arize_dynamic_headers, arize_preset
from litellm.integrations.otel.presets.base import Preset
from litellm.integrations.otel.presets.generic import generic_preset
from litellm.integrations.otel.presets.langfuse import (
    langfuse_dynamic_headers,
    langfuse_preset,
)
from litellm.integrations.otel.presets.langtrace import langtrace_preset
from litellm.integrations.otel.presets.levo import levo_preset
from litellm.integrations.otel.presets.phoenix import phoenix_preset
from litellm.integrations.otel.presets.weave import weave_dynamic_headers, weave_preset
from litellm.types.utils import StandardCallbackDynamicParams

if TYPE_CHECKING:
    from litellm.integrations.otel.model.destination import OtelDestination

#: Callback name → per-request OTLP header builder (team/key multi-tenant
#: routing). Only integrations that support dynamic credentials appear here —
#: Arize-Phoenix/Langtrace/Levo/AgentOps/generic don't, so they use the logger's
#: default tracer.
DYNAMIC_HEADERS_BY_CALLBACK: Final[dict[str, Callable[[StandardCallbackDynamicParams], dict[str, str]]]] = {
    "arize": arize_dynamic_headers,
    "langfuse_otel": langfuse_dynamic_headers,
    "weave_otel": weave_dynamic_headers,
}


def dynamic_otlp_headers(
    callback_name: str | None,
    dynamic_params: "StandardCallbackDynamicParams | None",
) -> dict[str, str] | None:
    """Per-request OTLP headers for ``callback_name``, or ``None`` if N/A.

    ``None`` means "no per-request routing" — the caller uses its default tracer.
    """
    builder: Final = DYNAMIC_HEADERS_BY_CALLBACK.get(callback_name or "")
    if builder is None or not dynamic_params:
        return None
    headers: Final = builder(dynamic_params)
    return headers or None


def dynamic_otlp_destination(
    callback_name: str | None,
    dynamic_params: "StandardCallbackDynamicParams | None",
) -> "OtelDestination | None":
    """The destination a request's own team/key credentials export to, or ``None``.

    Resolved through the admin-destination builders so a team's ``callback_vars`` reach
    exactly the account an equivalent destination would, honouring per-tenant overrides
    such as ``langfuse_host``. The builder's ``protocol`` comes with it: a transport the
    values pin is the tenant's, not the backend's intrinsic default.
    """
    from litellm.integrations.otel.presets.destinations import build_destination

    if callback_name not in DYNAMIC_HEADERS_BY_CALLBACK or not dynamic_params:
        return None
    values: Final = {str(key): str(value) for key, value in dynamic_params.items() if isinstance(value, str)}
    return build_destination(callback_name or "", values)


#: Callback name → preset. The ``Preset`` annotation makes mypy verify every
#: registered value matches the preset interface.
PRESET_BY_CALLBACK: Final[dict[str, Preset]] = {
    "agentops": agentops_preset,
    "arize": arize_preset,
    "arize_phoenix": phoenix_preset,
    "generic": generic_preset,
    "langfuse_otel": langfuse_preset,
    "langtrace": langtrace_preset,
    "levo": levo_preset,
    "weave_otel": weave_preset,
}


__all__ = [
    "DYNAMIC_HEADERS_BY_CALLBACK",
    "PRESET_BY_CALLBACK",
    "Preset",
    "agentops_preset",
    "arize_preset",
    "dynamic_otlp_destination",
    "dynamic_otlp_headers",
    "generic_preset",
    "langfuse_preset",
    "langtrace_preset",
    "levo_preset",
    "phoenix_preset",
    "weave_preset",
]
