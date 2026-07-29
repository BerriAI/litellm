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

from typing import Optional

from litellm.constants import DEFAULT_REQUEST_TIMEOUT_SECONDS


def get_configured_request_timeout() -> Optional[float]:
    """Return the explicitly-configured ``litellm.request_timeout``, else ``None``."""
    import litellm

    timeout = float(litellm.request_timeout)
    if litellm.request_timeout_explicitly_set:
        return timeout
    if timeout != float(DEFAULT_REQUEST_TIMEOUT_SECONDS):
        return timeout
    return None
