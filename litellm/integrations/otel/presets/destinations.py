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

from typing import Callable, Mapping, Optional

from litellm.constants import LITELLM_LOGGING_CREDENTIAL_NAME_KEY
from litellm.integrations.langfuse.langfuse_otel import (
    LANGFUSE_CLOUD_US_ENDPOINT,
    LangfuseOtelLogger,
)
from litellm.integrations.otel.model.destination import OtelDestination
from litellm.integrations.weave.weave_otel import _get_weave_authorization_header

#: Reserved ``callback_vars`` key binding a key/team's logging callback to a named
#: credential in the registry. It is a reference, resolved server-side; it is never
#: forwarded as a request parameter.
LOGGING_CREDENTIAL_NAME_KEY = LITELLM_LOGGING_CREDENTIAL_NAME_KEY


def _parse_header_string(raw: str) -> dict[str, str]:
    pairs = (item.split("=", 1) for item in raw.split(",") if "=" in item)
    return {key.strip(): value.strip() for key, value in pairs}


def _langfuse_endpoint(host: str) -> str:
    normalized = host if host.startswith("http") else f"https://{host}"
    return f"{normalized.rstrip('/')}/api/public/otel"


def _langfuse_destination(values: Mapping[str, str]) -> Optional[OtelDestination]:
    public_key = values.get("langfuse_public_key")
    secret_key = values.get("langfuse_secret_key")
    if not public_key or not secret_key:
        return None
    host = values.get("langfuse_host")
    endpoint = _langfuse_endpoint(host) if host else LANGFUSE_CLOUD_US_ENDPOINT
    auth = LangfuseOtelLogger._get_langfuse_authorization_header(
        public_key=public_key, secret_key=secret_key
    )
    return OtelDestination(endpoint=endpoint, headers={"Authorization": auth})


def _arize_destination(values: Mapping[str, str]) -> Optional[OtelDestination]:
    space = values.get("arize_space_id") or values.get("arize_space_key")
    api_key = values.get("arize_api_key")
    if not space or not api_key:
        return None
    endpoint = values.get("arize_endpoint") or "https://otlp.arize.com/v1"
    return OtelDestination(
        endpoint=endpoint, headers={"space_id": space, "api_key": api_key}
    )


def _weave_destination(values: Mapping[str, str]) -> Optional[OtelDestination]:
    api_key = values.get("wandb_api_key")
    endpoint = values.get("weave_endpoint")
    if not api_key or not endpoint:
        return None
    headers = {"Authorization": _get_weave_authorization_header(api_key=api_key)}
    project_id = values.get("weave_project_id")
    if project_id:
        headers["project_id"] = project_id
    return OtelDestination(endpoint=endpoint, headers=headers)


def _generic_destination(values: Mapping[str, str]) -> Optional[OtelDestination]:
    """Any OTLP backend: an explicit endpoint plus raw headers. The catch-all that
    makes the registry cover self-hosted collectors / Phoenix / Honeycomb / etc."""
    endpoint = values.get("otel_endpoint")
    if not endpoint:
        return None
    return OtelDestination(
        endpoint=endpoint, headers=_parse_header_string(values.get("otel_headers", ""))
    )


_ADAPTERS: dict[str, Callable[[Mapping[str, str]], Optional[OtelDestination]]] = {
    "langfuse_otel": _langfuse_destination,
    "arize": _arize_destination,
    "weave_otel": _weave_destination,
}

#: OTEL v2 callbacks that can be routed to a per-key/team admin destination.
OTEL_V2_DESTINATION_CALLBACKS = frozenset(_ADAPTERS)


def build_destination(
    callback_name: str, values: Mapping[str, str]
) -> Optional[OtelDestination]:
    """Map an admin credential's ``values`` to an ``OtelDestination`` for
    ``callback_name``, falling back to the generic OTLP passthrough."""
    adapter = _ADAPTERS.get(callback_name)
    if adapter is not None:
        destination = adapter(values)
        if destination is not None:
            return destination
    return _generic_destination(values)
