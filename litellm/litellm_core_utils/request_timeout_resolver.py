"""Single source of truth for whether ``litellm.request_timeout`` was configured.

``litellm.request_timeout`` always holds a value (the package default,
:data:`~litellm.constants.DEFAULT_REQUEST_TIMEOUT_SECONDS`), so a bare read can't
tell "user asked for this" from "nobody set it". This resolver answers that:

* ``request_timeout_explicitly_set`` is the authoritative signal, set when the
  value comes from the ``REQUEST_TIMEOUT`` env var or ``litellm_settings``.
* A runtime value that differs from the package default (e.g. ``litellm.request_timeout
  = 300`` in SDK code) is also treated as explicit, for backwards compatibility.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Literal, Protocol

from pydantic import TypeAdapter

from litellm.constants import DEFAULT_REQUEST_TIMEOUT_SECONDS

_TIMEOUTS_BY_SERVICE_TIER: Final[TypeAdapter[Mapping[str, float] | None]] = TypeAdapter(Mapping[str, float] | None)


def get_configured_request_timeout() -> float | None:
    """Return the explicitly-configured ``litellm.request_timeout``, else ``None``."""
    import litellm

    timeout: Final = float(litellm.request_timeout)
    if litellm.request_timeout_explicitly_set:
        return timeout
    if timeout != float(DEFAULT_REQUEST_TIMEOUT_SECONDS):
        return timeout
    return None


class _DeploymentParams(Protocol):
    def get(self, key: str, /) -> object: ...


def get_service_tier_timeout(
    service_tier: object,
    litellm_params: _DeploymentParams,
    key: Literal["timeout_by_service_tier", "stream_timeout_by_service_tier"],
) -> float | None:
    """Return the deployment's ``key`` timeout for the request's ``service_tier``, else ``None``."""
    if not isinstance(service_tier, str):
        return None
    timeouts: Final = _TIMEOUTS_BY_SERVICE_TIER.validate_python(litellm_params.get(key))
    return None if timeouts is None else timeouts.get(service_tier)
