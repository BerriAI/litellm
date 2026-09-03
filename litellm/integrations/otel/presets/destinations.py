"""Map a key's or team's callback vars to the OTLP destination its traces export to.

The auth path resolves one destination per backend the caller configured, and the
fan-out span processor exports the whole request through it. Header building is
delegated to each preset's existing ``*_dynamic_headers`` builder, so a destination
authenticates exactly the way the per-request tracer route already did; only the
endpoint needs a per-backend rule, because a backend's host is either fixed, taken
from a region table, or named by the tenant alongside its own key pair.
"""

import os
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

from litellm.integrations.otel.model.destination import OtelDestination
from litellm.types.utils import StandardCallbackDynamicParams

#: gRPC is Arize's own transport; an explicitly named HTTP collector overrides it.
_ARIZE_GRPC_ENDPOINT: Final = "https://otlp.arize.com/v1"


def _langfuse_endpoint(params: StandardCallbackDynamicParams) -> str | None:
    """The tenant's own Langfuse host, else the operator's, else Langfuse US cloud.

    Falling back to the operator's host is safe and is what V1 does: the tenant's
    own key pair still selects its own project, and a self-hosted deployment where
    every team lives on one Langfuse server is the common shape.
    """
    from litellm.integrations.langfuse.langfuse_otel import (
        LANGFUSE_CLOUD_US_ENDPOINT,
        LangfuseOtelLogger,
    )

    host: Final = params.get("langfuse_host") or LangfuseOtelLogger._get_langfuse_otel_host()  # pyright: ignore[reportPrivateUsage]  # reuse the backend's own env host resolver rather than duplicating it
    if not host:
        return LANGFUSE_CLOUD_US_ENDPOINT
    normalized: Final = host if host.startswith("http") else f"https://{host}"
    return f"{normalized.rstrip('/')}/api/public/otel"


def _arize_endpoint(params: StandardCallbackDynamicParams) -> str | None:
    return os.environ.get("ARIZE_ENDPOINT") or _ARIZE_GRPC_ENDPOINT


def _arize_protocol(params: StandardCallbackDynamicParams) -> str | None:
    return "otlp_http" if os.environ.get("ARIZE_HTTP_ENDPOINT") and not os.environ.get("ARIZE_ENDPOINT") else None


def _weave_endpoint(params: StandardCallbackDynamicParams) -> str | None:
    from litellm.integrations.weave.weave_otel import WEAVE_BASE_URL, WEAVE_OTEL_ENDPOINT

    return WEAVE_BASE_URL + WEAVE_OTEL_ENDPOINT


def _newrelic_endpoint(params: StandardCallbackDynamicParams) -> str | None:
    from litellm.integrations.otel.presets.newrelic import newrelic_dynamic_endpoint

    return newrelic_dynamic_endpoint(params)


#: Callback name -> endpoint resolver. A backend is destination-capable exactly
#: when it appears here AND in ``DYNAMIC_HEADERS_BY_CALLBACK``: without a header
#: builder the destination would carry no tenant credentials, and the exporter
#: would post the tenant's traffic to the operator's account.
_ENDPOINT_BY_CALLBACK: Final[Mapping[str, Callable[[StandardCallbackDynamicParams], str | None]]] = MappingProxyType(
    {
        "langfuse_otel": _langfuse_endpoint,
        "arize": _arize_endpoint,
        "weave_otel": _weave_endpoint,
        "newrelic": _newrelic_endpoint,
    }
)

_PROTOCOL_BY_CALLBACK: Final[Mapping[str, Callable[[StandardCallbackDynamicParams], str | None]]] = MappingProxyType(
    {
        "arize": _arize_protocol,
    }
)

_NO_ATTRS: Final[Mapping[str, str]] = MappingProxyType({})


def destination_capable_backends() -> frozenset[str]:
    """Backends a key or team can point at its own account."""
    from litellm.integrations.otel.presets import DYNAMIC_HEADERS_BY_CALLBACK

    return frozenset(_ENDPOINT_BY_CALLBACK) & frozenset(DYNAMIC_HEADERS_BY_CALLBACK)


def destination_for(callback_name: str, params: StandardCallbackDynamicParams) -> OtelDestination | None:
    """The destination ``params`` names for ``callback_name``, or ``None``.

    ``None`` means the caller configured nothing usable for this backend, so the
    request keeps the operator's global exporters. A partial config (a host with
    no key pair) resolves to ``None`` rather than to the operator's endpoint with
    the tenant's host, which would post the operator's credentials elsewhere.
    """
    from litellm.integrations.otel.presets import DYNAMIC_HEADERS_BY_CALLBACK

    header_builder: Final = DYNAMIC_HEADERS_BY_CALLBACK.get(callback_name)
    endpoint_builder: Final = _ENDPOINT_BY_CALLBACK.get(callback_name)
    if header_builder is None or endpoint_builder is None:
        return None
    headers: Final = header_builder(params)
    if not headers:
        return None
    endpoint: Final = endpoint_builder(params)
    if not endpoint:
        return None
    protocol_builder: Final = _PROTOCOL_BY_CALLBACK.get(callback_name)
    return OtelDestination(
        endpoint=endpoint,
        headers=MappingProxyType(dict(headers)),
        resource_attributes=_NO_ATTRS,
        callback_name=callback_name,
        protocol=protocol_builder(params) if protocol_builder is not None else None,
    )
