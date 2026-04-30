from typing import Any, Dict, FrozenSet

from fastapi import Request

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.trusted_proxy_utils import require_trusted_proxy_request

# OAuth2-proxy header trust is for **identity assertion** from a trusted
# upstream auth proxy (oauth2-proxy, Authelia, etc.). The allowlist below
# is the only safe surface — anything else (``user_role``, ``api_key``,
# ``permissions``, ``max_budget``, ``user_max_budget``,
# ``team_tpm_limit``, ``end_user_max_budget``, ``allowed_model_region``,
# and dozens of similar policy fields scattered across the
# ``LiteLLM_VerificationTokenView`` hierarchy) is a privilege grant that
# would let a caller forge their own enforcement parameters by sending
# the matching header.
#
# A denylist of "privileged fields" is unmaintainable in this codebase:
# the auth model has ~50 budget/spend/limit/permission fields and gains
# more with each release. An allowlist scoped to identity assertion is
# default-secure — new fields are blocked automatically.
#
# Operators who need a trusted upstream to assert anything beyond
# identity should switch to JWT authentication, which validates a
# signature on the assertion rather than blindly trusting headers.
ALLOWED_OAUTH2_PROXY_FIELDS: FrozenSet[str] = frozenset(
    {
        "user_id",
        "user_email",
        "team_id",
        "team_alias",
        "org_id",
        "models",
    }
)


async def handle_oauth2_proxy_request(request: Request) -> UserAPIKeyAuth:
    """
    Resolve a ``UserAPIKeyAuth`` from request headers per the admin-set
    ``oauth2_config_mappings``.

    The auth model assumes the proxy is deployed behind a trusted OAuth2
    reverse proxy that injects authenticated identity headers (e.g.
    oauth2-proxy, Authelia).

    **Identity-only allowlist.** ``oauth2_config_mappings`` maps header
    names to ``UserAPIKeyAuth`` fields. Without an allowlist, an admin
    who maps the wrong header to ``user_role`` lets any caller send
    ``X-User-Role: proxy_admin`` and gain full admin privileges
    (Pydantic coerces the string into the enum). Only fields in
    ``ALLOWED_OAUTH2_PROXY_FIELDS`` (identity assertion only — see the
    constant's comment) may be mapped; any other mapping is rejected at
    request time so the misconfiguration surfaces loudly rather than as
    a silent privesc.
    """
    from litellm.proxy.proxy_server import general_settings

    verbose_proxy_logger.debug("Handling oauth2 proxy request")
    require_trusted_proxy_request(
        request=request,
        general_settings=general_settings,
        feature_name="OAuth2 proxy auth",
    )

    oauth2_config_mappings: Dict[str, str] = (
        general_settings.get("oauth2_config_mappings") or {}
    )
    verbose_proxy_logger.debug(f"Oauth2 config mappings: {oauth2_config_mappings}")

    if not oauth2_config_mappings:
        raise ValueError("Oauth2 config mappings not found in general_settings")

    disallowed = sorted(
        set(oauth2_config_mappings.keys()) - ALLOWED_OAUTH2_PROXY_FIELDS
    )
    if disallowed:
        raise ValueError(
            "Oauth2 proxy auth refuses to map non-identity UserAPIKeyAuth "
            f"fields from request headers: {disallowed}. Only identity "
            f"fields are accepted ({sorted(ALLOWED_OAUTH2_PROXY_FIELDS)}); "
            "anything else (privileges, budgets, rate limits, metadata) "
            "would let a caller forge enforcement parameters by spoofing "
            "the matching header. If you need a trusted upstream to "
            "assert anything beyond identity, use JWT auth "
            "(signature-validated) instead of header-trust."
        )

    auth_data: Dict[str, Any] = {}
    for key, header in oauth2_config_mappings.items():
        value = request.headers.get(header)
        if not value:
            continue
        if key == "models":
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
