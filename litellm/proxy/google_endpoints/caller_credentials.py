"""Per-request provider credentials on the managed-agent and interaction routes.

These routes are reachable by any authenticated LLM key and are not routed through
``model_list``, so the only credential sources are the per-request
``litellm_params_template`` and the proxy's provider environment variables. The
environment fallback is reserved for proxy admins; anyone else brings their own key,
which is also what scopes them to their own agents and sessions on the provider side.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from fastapi import HTTPException, Request, status
from pydantic import TypeAdapter, ValidationError

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

_TEMPLATE: Final = TypeAdapter(dict[str, object])  # mutable-ok: parsed once and frozen right below
_NO_TEMPLATE: Final[Mapping[str, object]] = MappingProxyType({})


def is_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> bool:
    return user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN


def query_template(request: Request) -> Mapping[str, object]:
    """
    GET/DELETE endpoints cannot carry a JSON body, so per-request
    ``litellm_params_template`` (credentials, ``custom_llm_provider``) travels
    as one JSON-encoded query parameter:

    .. code-block:: bash

        curl "http://localhost:4000/v1beta/agents?litellm_params_template=%7B%22api_key%22%3A%22AIza...%22%7D" \\
            -H "Authorization: Bearer sk-..."

    Credentials MUST NOT be passed as plain flat query parameters (e.g.
    ``?api_key=AIza...``) because URL query strings appear verbatim in
    web-server access logs, CDN edge logs, browser history, and Referer
    headers.
    """
    raw_template: Final = request.query_params.get("litellm_params_template")
    if not raw_template:
        return _NO_TEMPLATE
    try:
        template: Final = _TEMPLATE.validate_json(raw_template)
    except ValidationError:
        return _NO_TEMPLATE
    return MappingProxyType(template)


def enforce_caller_supplied_provider_key(data: Mapping[str, object], user_api_key_dict: UserAPIKeyAuth) -> None:
    if is_proxy_admin(user_api_key_dict) or data.get("api_key"):
        return
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=(
            "Managed-agent endpoints require a caller-supplied provider "
            "api_key (via 'litellm_params_template'). Falling back to the "
            "proxy's provider environment variables is only permitted for "
            "proxy admins."
        ),
    )
