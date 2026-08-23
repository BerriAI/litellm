"""Anthropic workload identity federation: exchanges an external OIDC identity
token for a short-lived ``sk-ant-oat01`` token via the shared RFC 7523 engine."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, NoReturn
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict
from typing_extensions import assert_never

import litellm
from litellm.llms.base_llm.auth.token_exchange import (
    JwtBearerTokenExchangeEngine,
    default_token_exchange_engine,
)
from litellm.llms.base_llm.auth.types import (
    AssertionSourceError,
    ExchangeError,
    ExchangeResult,
    InsecureTokenUrl,
    MalformedTokenResponse,
    MintedToken,
    TokenEndpointError,
    TokenExchangeSpec,
    TokenTransportError,
)
from litellm.types.llms.anthropic import ANTHROPIC_TOKEN_EXCHANGE_PATH

_JWT_BEARER_GRANT_TYPE: Final = "urn:ietf:params:oauth:grant-type:jwt-bearer"
_DEFAULT_API_BASE: Final = "https://api.anthropic.com"
_INLINE_ENV_VAR: Final = "ANTHROPIC_IDENTITY_TOKEN"
_DISABLE_WIF_PARAM: Final = "anthropic_disable_workload_identity_federation"
_ACCEPTED_REF_PREFIX: Final = "oidc/"
_CHAT_BASE_SUFFIXES: Final = ("/v1/messages", "/v1")
_REJECTED_REF_PREFIX: Final = "oidc/env_path/"
_WORKSPACE_HINT: Final = (
    " If the federation rule is scoped to a workspace, set ANTHROPIC_WORKSPACE_ID"
    " (or the anthropic_workspace_id litellm param) to that workspace id."
)
_ALLOWLIST_HINT: Final = (
    " Identity token files must sit under an allowed credential directory"
    " (/var/run/secrets or /run/secrets by default);"
    " set LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS to extend the allowlist."
)


class AnthropicWifParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    federation_rule_id: str
    organization_id: str
    service_account_id: str | None = None
    workspace_id: str | None = None
    assertion_ref: str


def resolve_anthropic_wif_params(litellm_params: Mapping[str, object] | None) -> AnthropicWifParams | None:
    if litellm_params is not None and litellm_params.get(_DISABLE_WIF_PARAM) is True:
        return None
    federation_rule_id: Final = _config_value(
        litellm_params, "anthropic_federation_rule_id", "ANTHROPIC_FEDERATION_RULE_ID"
    )
    organization_id: Final = _config_value(litellm_params, "anthropic_organization_id", "ANTHROPIC_ORGANIZATION_ID")
    if federation_rule_id is None or organization_id is None:
        return None
    assertion_ref: Final = _resolve_assertion_ref(litellm_params)
    if assertion_ref is None:
        return None
    return AnthropicWifParams(
        federation_rule_id=federation_rule_id,
        organization_id=organization_id,
        service_account_id=_config_value(
            litellm_params, "anthropic_service_account_id", "ANTHROPIC_SERVICE_ACCOUNT_ID"
        ),
        workspace_id=_config_value(litellm_params, "anthropic_workspace_id", "ANTHROPIC_WORKSPACE_ID"),
        assertion_ref=assertion_ref,
    )


def build_anthropic_wif_spec(params: AnthropicWifParams, api_base: str) -> TokenExchangeSpec:
    return TokenExchangeSpec(
        token_url=api_base.rstrip("/") + ANTHROPIC_TOKEN_EXCHANGE_PATH,
        assertion_ref=params.assertion_ref,
        assertion_field="assertion",
        static_body=MappingProxyType(
            {
                name: value
                for name, value in (
                    ("grant_type", _JWT_BEARER_GRANT_TYPE),
                    ("federation_rule_id", params.federation_rule_id),
                    ("organization_id", params.organization_id),
                    ("service_account_id", params.service_account_id),
                    ("workspace_id", params.workspace_id),
                )
                if value is not None
            }
        ),
        body_encoding="json",
        request_headers=MappingProxyType({}),
        cache_key_identity=(
            params.federation_rule_id,
            params.organization_id,
            params.service_account_id or "",
            params.workspace_id or "",
        ),
    )


def get_anthropic_wif_token(
    litellm_params: Mapping[str, object] | None,
    api_base: str | None,
    model: str,
    engine: JwtBearerTokenExchangeEngine = default_token_exchange_engine,
) -> str | None:
    params: Final = resolve_anthropic_wif_params(litellm_params)
    if params is None:
        return None
    result: Final = engine.get_token(build_anthropic_wif_spec(params, _token_exchange_base(api_base)))
    return _token_from_result(result, model, params)


async def aget_anthropic_wif_token(
    litellm_params: Mapping[str, object] | None,
    api_base: str | None,
    model: str,
    engine: JwtBearerTokenExchangeEngine = default_token_exchange_engine,
) -> str | None:
    params: Final = resolve_anthropic_wif_params(litellm_params)
    if params is None:
        return None
    result: Final = await engine.aget_token(build_anthropic_wif_spec(params, _token_exchange_base(api_base)))
    return _token_from_result(result, model, params)


def _token_from_result(result: ExchangeResult, model: str, params: AnthropicWifParams) -> str:
    match result:
        case MintedToken():
            return result.access_token.get_secret_value()
        case _:
            _raise_anthropic_wif_error(result, model=model, workspace_id_set=params.workspace_id is not None)


def _token_exchange_base(api_base: str | None) -> str:
    """Exchange base for any caller-supplied form of the deployment base: trailing
    slashes and chat-appended ``/v1/messages`` suffixes stripped, so every tier
    derives the same token URL (and cache key) for the same deployment."""
    return _strip_chat_suffix(api_base if api_base is not None else _resolve_default_api_base())


def _resolve_default_api_base() -> str:
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    return AnthropicModelInfo.get_api_base(None) or _DEFAULT_API_BASE


def _strip_chat_suffix(base: str) -> str:
    parts: Final = urlsplit(base)
    if not parts.scheme or not parts.netloc:
        return base.rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, _strip_path_suffixes(parts.path), "", ""))


def _strip_path_suffixes(path: str) -> str:
    """Drop the chat-surface suffixes a deployment base may carry, so every tier derives the same
    token URL. Recursion depth is bounded by the path's own segment count."""
    trimmed: Final = path.rstrip("/")
    shortened: Final = next(
        (trimmed.removesuffix(suffix) for suffix in _CHAT_BASE_SUFFIXES if trimmed.endswith(suffix)),
        trimmed,
    )
    return trimmed if shortened == trimmed else _strip_path_suffixes(shortened)


def _config_value(litellm_params: Mapping[str, object] | None, param_key: str, env_name: str) -> str | None:
    return _param_str(litellm_params, param_key) or _env_str(env_name)


def _param_str(litellm_params: Mapping[str, object] | None, key: str) -> str | None:
    if litellm_params is None:
        return None
    value: Final = litellm_params.get(key)
    return value if isinstance(value, str) and value else None


def _env_str(name: str) -> str | None:
    from litellm.secret_managers.main import get_secret_str

    value: Final = get_secret_str(name)
    return value if isinstance(value, str) and value else None


def _resolve_assertion_ref(litellm_params: Mapping[str, object] | None) -> str | None:
    file_param: Final = _param_str(litellm_params, "anthropic_identity_token_file")
    if file_param is not None:
        return f"oidc/file/{file_param}"
    inline_param: Final = _param_str(litellm_params, "anthropic_identity_token")
    if inline_param is not None:
        return _validated_inline_ref(inline_param)
    file_env: Final = _env_str("ANTHROPIC_IDENTITY_TOKEN_FILE")
    if file_env is not None:
        return f"oidc/file/{file_env}"
    if _env_str(_INLINE_ENV_VAR) is not None:
        return f"oidc/env/{_INLINE_ENV_VAR}"
    return None


def _validated_inline_ref(value: str) -> str:
    if value.startswith(_ACCEPTED_REF_PREFIX) and not value.startswith(_REJECTED_REF_PREFIX):
        return value
    raise litellm.AuthenticationError(
        message=(
            "anthropic_identity_token must be an oidc/ secret reference such as oidc/env/VAR_NAME,"
            " oidc/file//absolute/path, oidc/github/<audience>, or oidc/google/<audience>."
            " Raw identity tokens and oidc/env_path/ references are not accepted;"
            " to pass a token directly, export it and reference it as oidc/env/VAR_NAME"
        ),
        llm_provider="anthropic",
        model="",
    )


def _raise_anthropic_wif_error(error: ExchangeError, model: str, workspace_id_set: bool) -> NoReturn:
    raise litellm.AuthenticationError(
        message=f"Anthropic workload identity federation failed. {_error_detail(error, workspace_id_set)}",
        llm_provider="anthropic",
        model=model,
    )


def _error_detail(error: ExchangeError, workspace_id_set: bool) -> str:
    match error:
        case AssertionSourceError() if error.kind == "disallowed_path":
            return f"Could not read the OIDC identity token from {error.source_ref}.{_ALLOWLIST_HINT}"
        case AssertionSourceError():
            return f"Could not obtain the OIDC identity token ({error.kind}) from {error.source_ref}."
        case InsecureTokenUrl():
            return f"The token endpoint must use https; refusing to send the identity token to host {error.host!r}."
        case TokenEndpointError() if error.status_code == 401 and not workspace_id_set:
            return f"The token endpoint returned HTTP 401: {error.redacted_body}{_WORKSPACE_HINT}"
        case TokenEndpointError():
            return f"The token endpoint returned HTTP {error.status_code}: {error.redacted_body}"
        case TokenTransportError():
            return f"Could not reach the token endpoint: {error.detail}."
        case MalformedTokenResponse():
            return f"The token endpoint returned an unusable response: {error.detail}."
        case _:
            assert_never(error)
