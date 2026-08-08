"""Resolve an admin-owned named credential into a typed OTLP destination.

The destination (endpoint + auth headers) is admin infrastructure config. Each
OTEL backend stores its own fields on the named credential's free-form
``credential_values``; the adapter here maps those fields to the universal
``OtelDestination`` the v2 router exports through. A backend with no bespoke
adapter is still reachable through the generic ``otel_endpoint`` / ``otel_headers``
passthrough, so the registry covers every OTEL destination rather than an
enumerated few. Nothing here reads request data; callers pass admin-resolved
credential values only.
"""

import os
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

from litellm.integrations.langfuse.langfuse_otel import (
    LANGFUSE_CLOUD_US_ENDPOINT,
    LangfuseOtelLogger,
)
from litellm.integrations.otel.model.destination import OtelDestination
from litellm.integrations.weave.weave_otel import _get_weave_authorization_header


def _parse_header_string(raw: str) -> Mapping[str, str]:
    pairs: Final = (item.split("=", 1) for item in raw.split(",") if "=" in item)
    return MappingProxyType({key.strip(): value.strip() for key, value in pairs})


def _langfuse_endpoint(host: str) -> str:
    normalized: Final = host if host.startswith("http") else f"https://{host}"
    return f"{normalized.rstrip('/')}/api/public/otel"


def _langfuse_destination(values: Mapping[str, str]) -> OtelDestination | None:
    public_key: Final = values.get("langfuse_public_key")
    secret_key: Final = values.get("langfuse_secret_key")
    if not public_key or not secret_key:
        return None
    host: Final = values.get("langfuse_host")
    endpoint: Final = _langfuse_endpoint(host) if host else LANGFUSE_CLOUD_US_ENDPOINT
    auth: Final = LangfuseOtelLogger._get_langfuse_authorization_header(public_key=public_key, secret_key=secret_key)
    return OtelDestination(endpoint=endpoint, headers=MappingProxyType({"Authorization": auth}))


def _arize_destination(values: Mapping[str, str]) -> OtelDestination | None:
    space: Final = values.get("arize_space_id") or values.get("arize_space_key")
    api_key: Final = values.get("arize_api_key")
    if not space or not api_key:
        return None
    # Mirrors the global config's ARIZE_ENDPOINT / ARIZE_HTTP_ENDPOINT split: Arize's
    # own endpoint is gRPC, but a destination may point at an HTTP collector, and the
    # URL scheme cannot express that (the gRPC endpoint is also https://).
    http_endpoint: Final = values.get("arize_http_endpoint")
    endpoint: Final = values.get("arize_endpoint") or http_endpoint or "https://otlp.arize.com/v1"
    project: Final = (
        values.get("arize_project_name") or values.get("project_name") or os.environ.get("ARIZE_PROJECT_NAME")
    )
    resource_attributes: Final = (
        MappingProxyType({"model_id": project, "arize.project.name": project}) if project else _EMPTY
    )
    return OtelDestination(
        endpoint=endpoint,
        headers=MappingProxyType({"space_id": space, "api_key": api_key}),
        resource_attributes=resource_attributes,
        protocol="otlp_http" if http_endpoint and not values.get("arize_endpoint") else None,
    )


def _weave_destination(values: Mapping[str, str]) -> OtelDestination | None:
    api_key: Final = values.get("wandb_api_key")
    if not api_key:
        return None
    from litellm.integrations.weave.weave_otel import (
        WEAVE_BASE_URL,
        WEAVE_OTEL_ENDPOINT,
    )

    base: Final = (values.get("weave_endpoint") or WEAVE_BASE_URL).rstrip("/")
    endpoint: Final = base if base.endswith("/v1/traces") else base.removesuffix("/otel") + WEAVE_OTEL_ENDPOINT
    project_id: Final = values.get("weave_project_id")
    headers: Final = MappingProxyType(
        {  # mutable-ok: the OTLP exporter is handed a concrete header map
            "Authorization": _get_weave_authorization_header(api_key=api_key),
            **({"project_id": project_id} if project_id else {}),  # mutable-ok: optional key, spread inline
        }
    )
    return OtelDestination(endpoint=endpoint, headers=headers)


def _generic_destination(values: Mapping[str, str]) -> OtelDestination | None:
    """Any OTLP backend: an explicit endpoint plus raw headers. The catch-all that
    makes the registry cover self-hosted collectors / Phoenix / Honeycomb / etc.

    The protocol is pinned rather than left for the router to default, because this
    fallback also builds destinations for named backends whose own adapter declined the
    values (an ``arize`` credential carrying only ``otel_endpoint``). Leaving it unset
    there made the router apply the backend's intrinsic transport -- gRPC for Arize --
    to the plain HTTP URL the admin typed, so the destination silently delivered nothing
    while still being disclosed as active.
    """
    endpoint: Final = values.get("otel_endpoint")
    if not endpoint:
        return None
    return OtelDestination(
        endpoint=endpoint,
        headers=_parse_header_string(values.get("otel_headers", "")),
        protocol="otlp_http",
    )


_EMPTY: Final[Mapping[str, str]] = MappingProxyType({})

_ADAPTERS: Final[Mapping[str, Callable[[Mapping[str, str]], OtelDestination | None]]] = MappingProxyType(
    {
        "langfuse_otel": _langfuse_destination,
        "arize": _arize_destination,
        "weave_otel": _weave_destination,
    }
)

OTEL_V2_DESTINATION_CALLBACKS: Final = frozenset(_ADAPTERS)


def build_destination(callback_name: str, values: Mapping[str, str]) -> OtelDestination | None:
    """Map an admin credential's ``values`` to an ``OtelDestination`` for
    ``callback_name``, falling back to the generic OTLP passthrough.

    Values are trimmed first: a stray leading/trailing space in an endpoint or
    host (an easy slip in the create form) yields a malformed OTLP URL the
    exporter rejects with a 404, so whitespace is never significant here.
    """
    trimmed: Final = MappingProxyType(
        {key: value.strip() for key, value in values.items()}
    )
    adapter: Final = _ADAPTERS.get(callback_name)
    if adapter is not None:
        destination: Final = adapter(trimmed)
        if destination is not None:
            return destination
    return _generic_destination(trimmed)
