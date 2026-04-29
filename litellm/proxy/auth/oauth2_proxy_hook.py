from typing import Any, Dict, FrozenSet

from fastapi import Request

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth

# Fields on ``UserAPIKeyAuth`` that grant privileges directly (``user_role``
# is the canonical privesc — coerced from the string ``"proxy_admin"`` into
# ``LitellmUserRoles.PROXY_ADMIN`` by Pydantic) or break trust assumptions
# (``api_key`` / ``token`` short-circuit the validated-key contract;
# ``permissions`` / ``allowed_routes`` directly grant route access; budget
# and limit fields can be set to wild values to bypass enforcement;
# ``metadata`` is too broad to safely admit from caller-controlled headers).
#
# Operators who legitimately need any of these to flow from a trusted
# upstream proxy should switch to JWT authentication, which validates a
# signature on the assertion rather than blindly trusting headers.
PRIVILEGED_OAUTH2_PROXY_FIELDS: FrozenSet[str] = frozenset(
    {
        "user_role",
        "api_key",
        "token",
        "key_alias",
        "key_name",
        "permissions",
        "allowed_routes",
        "max_budget",
        "spend",
        "model_max_budget",
        "model_spend",
        "tpm_limit",
        "rpm_limit",
        "team_max_budget",
        "team_spend",
        "blocked",
        "metadata",
    }
)


async def handle_oauth2_proxy_request(request: Request) -> UserAPIKeyAuth:
    """
    Resolve a ``UserAPIKeyAuth`` from request headers per the admin-set
    ``oauth2_config_mappings``.

    The auth model assumes the proxy is deployed behind a trusted OAuth2
    reverse proxy that injects authenticated identity headers (e.g.
    oauth2-proxy, Authelia). Two safeguards above and beyond that
    deployment assumption:

    1. **Premium gate.** The sibling auth paths (``enable_oauth2_auth``
       and ``enable_jwt_auth``) require ``premium_user``; this path
       previously did not, which let any open-source deployment turn
       the feature on without realising it requires a hardened
       deployment topology.
    2. **Privileged-field denylist.** ``oauth2_config_mappings`` maps
       header names to ``UserAPIKeyAuth`` fields. Without a denylist,
       an admin who maps the wrong header to ``user_role`` (or who
       hasn't fully locked down their reverse proxy) lets any caller
       set the ``user_role`` header to ``"proxy_admin"`` and gain full
       admin privileges — Pydantic coerces the string into the enum.
       Mapping any privileged field is rejected at startup-style auth
       time so the misconfiguration surfaces loudly rather than as a
       silent privesc.
    """
    from litellm.proxy.proxy_server import general_settings, premium_user

    if premium_user is not True:
        raise ValueError(
            "Oauth2 proxy auth is an enterprise-only feature. "
            + CommonProxyErrors.not_premium_user.value
        )

    verbose_proxy_logger.debug("Handling oauth2 proxy request")
    oauth2_config_mappings: Dict[str, str] = (
        general_settings.get("oauth2_config_mappings") or {}
    )
    verbose_proxy_logger.debug(f"Oauth2 config mappings: {oauth2_config_mappings}")

    if not oauth2_config_mappings:
        raise ValueError("Oauth2 config mappings not found in general_settings")

    privileged_mapped = sorted(
        set(oauth2_config_mappings.keys()) & PRIVILEGED_OAUTH2_PROXY_FIELDS
    )
    if privileged_mapped:
        raise ValueError(
            "Oauth2 proxy auth refuses to map privileged UserAPIKeyAuth "
            f"fields from request headers: {privileged_mapped}. These "
            "fields would grant privileges (e.g. proxy_admin), bypass "
            "budget enforcement, or short-circuit key validation if a "
            "caller can spoof the corresponding header. If you need a "
            "trusted upstream to assert one of these, use JWT auth "
            "(signature-validated) instead of header-trust."
        )

    auth_data: Dict[str, Any] = {}
    for key, header in oauth2_config_mappings.items():
        value = request.headers.get(header)
        if not value:
            continue
        if key == "max_budget":
            auth_data[key] = float(value)
        elif key == "models":
            auth_data[key] = [model.strip() for model in value.split(",")]
        else:
            auth_data[key] = value

    verbose_proxy_logger.debug(
        "Auth data before creating UserAPIKeyAuth object: keys=%s",
        list(auth_data.keys()),
    )
    user_api_key_auth = UserAPIKeyAuth(**auth_data)
    verbose_proxy_logger.debug(
        "UserAPIKeyAuth object created with keys: %s",
        list(user_api_key_auth.__fields_set__),
    )
    return user_api_key_auth
