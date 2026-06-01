"""Preset interface.

A preset is a callable that reads its integration's env vars and produces an
:class:`OpenTelemetryV2Config` (exporter list + mapper-name list + resource
attributes). This ``Protocol`` pins that contract so ``PRESET_BY_CALLBACK`` and
the factory in ``litellm_logging`` are type-checked structurally against it,
matching the ``AttributeMapper`` protocol the mappers use.
"""

from typing import Protocol, runtime_checkable

from litellm.integrations.otel.model.config import OpenTelemetryV2Config


@runtime_checkable
class Preset(Protocol):
    """Reads an integration's env config and returns an ``OpenTelemetryV2Config``.

    ``config_overrides`` lets one preset layer onto another's config (or onto
    test-supplied defaults); the factory calls presets with no arguments.
    """

    def __call__(
        self, *, config_overrides: OpenTelemetryV2Config | None = None
    ) -> OpenTelemetryV2Config: ...
