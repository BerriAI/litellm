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
from litellm.proxy.auth.auth_utils import (
    is_request_body_safe,  # pyright: ignore[reportUnknownVariableType]  # legacy helper typed with bare dicts
)

TEMPLATE_HEADER: Final = "x-litellm-params-template"
TEMPLATE_QUERY_PARAM: Final = "litellm_params_template"
DEFAULT_PROVIDER: Final = "gemini"
_TEMPLATE: Final = TypeAdapter(dict[str, object])  # mutable-ok: parsed once and frozen right below
_NO_TEMPLATE: Final[Mapping[str, object]] = MappingProxyType({})


def is_proxy_admin(user_api_key_dict: UserAPIKeyAuth) -> bool:
    return user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN


def query_template(request: Request) -> Mapping[str, object]:
    """
    GET/DELETE endpoints cannot carry a JSON body, so per-request
    ``litellm_params_template`` (credentials, ``custom_llm_provider``, ``api_base``)
    travels as JSON in the ``x-litellm-params-template`` header, or as one
    JSON-encoded query parameter when a header cannot be set:

    .. code-block:: bash

        curl "http://localhost:4000/v1beta/agents" \\
            -H "Authorization: Bearer sk-..." \\
            -H 'x-litellm-params-template: {"api_key": "AIza..."}'

    The header is preferred: query strings appear verbatim in web-server access
    logs, CDN edge logs, browser history, and Referer headers. Flat query
    parameters (e.g. ``?api_key=AIza...``) are never read.

    The template is subject to the same client-side credential policy as a
    request body: ``api_base`` and the other endpoint-retargeting fields are
    refused unless the admin opted in with
    ``general_settings.allow_client_side_credentials``.
    """
    raw_template: Final = request.headers.get(TEMPLATE_HEADER) or request.query_params.get(TEMPLATE_QUERY_PARAM)
    if not raw_template:
        return _NO_TEMPLATE
    try:
        template: Final = _TEMPLATE.validate_json(raw_template)
    except ValidationError:
        return _NO_TEMPLATE
    _reject_banned_template_params(template)
    return MappingProxyType(template)


def _reject_banned_template_params(
    template: dict[str, object],  # mutable-ok: is_request_body_safe takes the parsed body as a dict
) -> None:
    """The template is expanded into the outbound call after the auth-time body check ran, so
    it gets the same banned-param policy here: a caller who can name any ``api_base`` can make
    the proxy request arbitrary hosts it can reach."""
    from litellm.proxy.proxy_server import (
        general_settings,  # pyright: ignore[reportUnknownVariableType]  # module global typed as a bare dict
        llm_router,
    )

    try:
        is_request_body_safe(request_body=template, general_settings=general_settings, llm_router=llm_router, model="")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"error": str(e)}) from e


def enforce_caller_key_for_overrides(data: Mapping[str, object], user_api_key_dict: UserAPIKeyAuth) -> None:
    """A non-admin who points the call at another provider or another ``api_base`` brings the key
    for it: otherwise the proxy's environment credential would be sent wherever the caller says."""
    if data.get("custom_llm_provider", DEFAULT_PROVIDER) != DEFAULT_PROVIDER or data.get("api_base"):
        enforce_caller_supplied_provider_key(data, user_api_key_dict)


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
