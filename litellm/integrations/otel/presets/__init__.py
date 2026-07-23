"""Integration presets — each one returns an :class:`OpenTelemetryV2Config`.

A preset is a callable that reads an integration's env vars and returns an
``OpenTelemetryV2Config`` describing the exporter destination, the mapper
vocabularies to apply, and any resource attributes. ``PRESET_BY_CALLBACK``
maps a callback name (``"arize"``, ``"langfuse_otel"``, ...) to its preset so
the factory in ``litellm_logging`` can resolve a name and build a single
``OpenTelemetryV2`` instance from the result.

Per-key/team routing does not live here. A trace destination is admin-owned
infrastructure config, resolved server-side from a named credential into an
``OtelDestination`` (see ``litellm.integrations.otel.presets.destinations`` and
``plumbing.routing``); nothing in this package reads vendor credentials or a
host off a request.
"""

from litellm.integrations.otel.presets.agentops import agentops_preset
from litellm.integrations.otel.presets.arize import arize_preset
from litellm.integrations.otel.presets.base import Preset
from litellm.integrations.otel.presets.generic import generic_preset
from litellm.integrations.otel.presets.langfuse import langfuse_preset
from litellm.integrations.otel.presets.langtrace import langtrace_preset
from litellm.integrations.otel.presets.levo import levo_preset
from litellm.integrations.otel.presets.phoenix import phoenix_preset
from litellm.integrations.otel.presets.weave import weave_preset

#: Callback name → preset. The ``Preset`` annotation makes mypy verify every
#: registered value matches the preset interface.
PRESET_BY_CALLBACK: dict[str, Preset] = {
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
    "PRESET_BY_CALLBACK",
    "Preset",
    "agentops_preset",
    "arize_preset",
    "generic_preset",
    "langfuse_preset",
    "langtrace_preset",
    "levo_preset",
    "phoenix_preset",
    "weave_preset",
]
