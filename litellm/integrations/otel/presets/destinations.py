"""Map a key's or team's callback vars to the OTLP destination its traces export to.

Header building is delegated to each preset's existing ``*_dynamic_headers`` builder,
so a destination authenticates exactly the way the per-request tracer route already
did; only the endpoint and transport need a per-backend rule.
"""

import os
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

from litellm._logging import verbose_logger
from litellm.integrations.otel.model.destination import OtelDestination
from litellm.litellm_core_utils.url_utils import SSRFError, assert_public_url
from litellm.types.utils import StandardCallbackDynamicParams


def _langfuse_endpoint(params: StandardCallbackDynamicParams) -> str | None:
    """The tenant's own Langfuse host, else the operator's, else Langfuse US cloud.

    A host the tenant named goes through the proxy's SSRF guard first. Anyone who can
    mint a key can write it, so without the check it points the exporter, and the
    tenant credentials it carries, at any address the proxy can reach. The operator's
    own ``LANGFUSE_HOST`` is not checked: an internal collector is a normal
    deployment and the operator is the one configuring it.
    """
    from litellm.integrations.langfuse.langfuse_otel import (
        LANGFUSE_CLOUD_US_ENDPOINT,
        LangfuseOtelLogger,
    )

    tenant_host: Final = params.get("langfuse_host") or None
    host: Final = tenant_host or LangfuseOtelLogger._get_langfuse_otel_host()  # pyright: ignore[reportPrivateUsage]  # reuse the backend's own env host resolver rather than duplicating it
    if not host:
        return LANGFUSE_CLOUD_US_ENDPOINT
    normalized: Final = host if host.startswith("http") else f"https://{host}"
    endpoint: Final = f"{normalized.rstrip('/')}/api/public/otel"
    if tenant_host is None:
        return endpoint
    try:
        assert_public_url(endpoint)
    except SSRFError as exc:
        verbose_logger.warning(
            "OTel V2: not exporting to key/team Langfuse host '%s' (%s). "
            "Add it to general_settings.user_url_allowed_hosts to permit it",
            host,
            exc,
        )
        return None
    return endpoint


def _arize_endpoint(params: StandardCallbackDynamicParams) -> str | None:
    from litellm.integrations.arize.arize import ArizeLogger

    return ArizeLogger.get_arize_config().endpoint


def _arize_protocol(params: StandardCallbackDynamicParams) -> str | None:
    from litellm.integrations.arize.arize import ArizeLogger

    return ArizeLogger.get_arize_config().protocol


def _weave_endpoint(params: StandardCallbackDynamicParams) -> str | None:
    from litellm.integrations.weave.weave_otel import weave_otel_endpoint

    return weave_otel_endpoint(os.environ.get("WANDB_HOST"))


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

#: Headers a destination must carry to authenticate. Several dynamic-header builders
#: gate each credential independently, so a half-configured backend yields a non-empty
#: but unusable header set; accepting it would suppress the operator's own exporter and
#: send the request's whole trace where it cannot be stored.
_REQUIRED_HEADERS_BY_CALLBACK: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "langfuse_otel": frozenset({"Authorization"}),
        "arize": frozenset({"arize-space-id", "api_key"}),
        "weave_otel": frozenset({"Authorization", "project_id"}),
        "newrelic": frozenset({"api-key"}),
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
    request keeps the operator's global exporters.
    """
    from litellm.integrations.otel.presets import DYNAMIC_HEADERS_BY_CALLBACK

    header_builder: Final = DYNAMIC_HEADERS_BY_CALLBACK.get(callback_name)
    endpoint_builder: Final = _ENDPOINT_BY_CALLBACK.get(callback_name)
    if header_builder is None or endpoint_builder is None:
        return None
    headers: Final = header_builder(params)
    if not _REQUIRED_HEADERS_BY_CALLBACK.get(callback_name, frozenset()) <= frozenset(headers):
        return None
    endpoint: Final = endpoint_builder(params)
    if not endpoint:
        return None
    protocol_builder: Final = _PROTOCOL_BY_CALLBACK.get(callback_name)
    return OtelDestination(
        endpoint=endpoint,
        headers=MappingProxyType(dict(headers)),  # mutable-ok: MappingProxyType needs a concrete mapping to wrap
        resource_attributes=_NO_ATTRS,
        callback_name=callback_name,
        protocol=protocol_builder(params) if protocol_builder is not None else None,
    )
