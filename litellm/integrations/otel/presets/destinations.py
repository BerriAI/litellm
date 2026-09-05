"""Map a key's or team's callback vars to the OTLP destination its traces export to.

Header building is delegated to each preset's existing ``*_dynamic_headers`` builder,
so a destination authenticates exactly the way the per-request tracer route already
did; only the endpoint and transport need a per-backend rule.
"""

import os
from collections.abc import Callable, Mapping
from functools import lru_cache
from types import MappingProxyType
from typing import Final

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.otel.model.destination import OtelDestination
from litellm.litellm_core_utils.url_utils import is_url_destination_allowed_by_host
from litellm.types.utils import StandardCallbackDynamicParams

#: An endpoint plus the OTLP transport to reach it with, or ``None`` when the backend
#: names no destination. The transport is ``None`` where the backend has only one.
_Destination = tuple[str, str | None]


@lru_cache(maxsize=128)
def _warn_host_not_allowlisted(host: str) -> None:
    """Cached so one misconfigured team logs once rather than once per request."""
    verbose_logger.warning(
        "OTel V2: not exporting to key/team Langfuse host '%s'. Add it to "
        "litellm_settings.provider_url_destination_allowed_hosts to permit it",
        host,
    )


def _langfuse_destination(params: StandardCallbackDynamicParams) -> "_Destination | None":
    """The tenant's own Langfuse host, else the operator's, else Langfuse US cloud.

    A host the tenant named has to be allowlisted by the operator, the same way a
    URL-valued ``model`` is: anyone who can mint a key can write it, and it becomes an
    endpoint the proxy posts the request's whole trace to, carrying the tenant's own
    credentials. The operator's own ``LANGFUSE_HOST`` is not checked, since an internal
    collector there is a deployment choice.
    """
    from litellm.integrations.langfuse.langfuse_otel import (
        LANGFUSE_CLOUD_US_ENDPOINT,
        LangfuseOtelLogger,
    )

    tenant_host: Final = params.get("langfuse_host") or None
    host: Final = tenant_host or LangfuseOtelLogger._get_langfuse_otel_host()  # pyright: ignore[reportPrivateUsage]  # reuse the backend's own env host resolver rather than duplicating it
    if not host:
        return (LANGFUSE_CLOUD_US_ENDPOINT, None)
    normalized: Final = host if host.startswith("http") else f"https://{host}"
    endpoint: Final = f"{normalized.rstrip('/')}/api/public/otel"
    if tenant_host is None:
        return (endpoint, None)
    if not is_url_destination_allowed_by_host(endpoint, litellm.provider_url_destination_allowed_hosts):
        _warn_host_not_allowlisted(host)
        return None
    return (endpoint, None)


def _arize_destination(params: StandardCallbackDynamicParams) -> "_Destination | None":
    from litellm.integrations.arize.arize import ArizeLogger

    config: Final = ArizeLogger.get_arize_config()
    return (config.endpoint, config.protocol)


def _weave_destination(params: StandardCallbackDynamicParams) -> "_Destination | None":
    from litellm.integrations.weave.weave_otel import weave_otel_endpoint

    return (weave_otel_endpoint(os.environ.get("WANDB_HOST")), None)


def _newrelic_destination(params: StandardCallbackDynamicParams) -> "_Destination | None":
    from litellm.integrations.otel.presets.newrelic import newrelic_dynamic_endpoint

    endpoint: Final = newrelic_dynamic_endpoint(params)
    return (endpoint, None) if endpoint else None


#: Callback name -> destination resolver. A backend is destination-capable exactly
#: when it appears here AND in ``DYNAMIC_HEADERS_BY_CALLBACK``: without a header
#: builder the destination would carry no tenant credentials, and the exporter
#: would post the tenant's traffic to the operator's account.
_DESTINATION_BY_CALLBACK: Final[Mapping[str, Callable[[StandardCallbackDynamicParams], "_Destination | None"]]] = (
    MappingProxyType(
        {
            "langfuse_otel": _langfuse_destination,
            "arize": _arize_destination,
            "weave_otel": _weave_destination,
            "newrelic": _newrelic_destination,
        }
    )
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

    return frozenset(_DESTINATION_BY_CALLBACK) & frozenset(DYNAMIC_HEADERS_BY_CALLBACK)


def destination_for(
    callback_name: str,
    params: StandardCallbackDynamicParams,
    service_name: str | None = None,
) -> OtelDestination | None:
    """The destination ``params`` names for ``callback_name``, or ``None``.

    ``None`` means the caller configured nothing usable for this backend, so the
    request keeps the operator's global exporters. ``service_name`` is the key's or
    team's ``otel_service_name``, which the per-request tracer route applies when the
    backend is not overridden and the destination has to apply once it is.
    """
    from litellm.integrations.otel.presets import DYNAMIC_HEADERS_BY_CALLBACK

    header_builder: Final = DYNAMIC_HEADERS_BY_CALLBACK.get(callback_name)
    destination_builder: Final = _DESTINATION_BY_CALLBACK.get(callback_name)
    if header_builder is None or destination_builder is None:
        return None
    headers: Final = header_builder(params)
    if not headers or not _REQUIRED_HEADERS_BY_CALLBACK[callback_name] <= frozenset(headers):
        return None
    resolved: Final = destination_builder(params)
    if resolved is None:
        return None
    endpoint, protocol = resolved
    return OtelDestination(
        endpoint=endpoint,
        headers=MappingProxyType(dict(headers)),  # mutable-ok: MappingProxyType needs a concrete mapping to wrap
        resource_attributes=MappingProxyType({"service.name": service_name}) if service_name else _NO_ATTRS,
        callback_name=callback_name,
        protocol=protocol,
    )
