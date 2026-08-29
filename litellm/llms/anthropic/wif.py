"""Anthropic workload identity federation: exchanges an external OIDC identity
token for a short-lived ``sk-ant-oat01`` token via the shared RFC 7523 engine."""

import os
from collections.abc import Callable, Mapping
from itertools import chain
from types import MappingProxyType
from typing import Final, NoReturn, TypeVar
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, ValidationError
from typing_extensions import assert_never

import litellm
from litellm.llms.base_llm.auth.client_credentials import keycloak_assertion_source
from litellm.llms.base_llm.auth.identity_source import (
    AnthropicIdentitySourceKind,
    InternalIssuerSource,
    KeycloakSource,
    identity_source_ref,
)
from litellm.llms.base_llm.auth.internal_issuer import internal_issuer_assertion_source
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
# Hosts a federated exchange may talk to. api_base decides where the workload's assertion is sent
# AND where the minted org-scoped token is presented, so anyone able to write api_base on a
# federated deployment could otherwise redirect both. Gating each write path does not terminate:
# a deployment, a referenced credential and a future endpoint all reach the same value. This is the
# one place a federated exchange is built, so the trust decision is enforced here instead, and the
# allowlist is server-owned -- read from the environment, never from a model or credential API.
_TRUSTED_EXCHANGE_HOSTS_ENV: Final = "LITELLM_ANTHROPIC_WIF_ALLOWED_HOSTS"
_DEFAULT_TRUSTED_EXCHANGE_HOST: Final = "api.anthropic.com"
_REJECTED_REF_PREFIX: Final = "oidc/env_path/"
_IDENTITY_SOURCE_PARAM: Final = "anthropic_identity_source"
_IDENTITY_SOURCE_ENV: Final = "ANTHROPIC_IDENTITY_SOURCE"

# litellm_params key -> InternalIssuerSource/KeycloakSource field name. Every key here must
# also be listed in ANTHROPIC_WIF_KWARGS_KEYS (get_litellm_params.py), which is what makes it
# request-banned and cleared on a client-redirected api_base -- see types/utils.py's
# anthropic_wif_litellm_params, derived from that same set.
_INTERNAL_ISSUER_FIELD_MAP: Final[Mapping[str, str]] = MappingProxyType(
    {
        "anthropic_issuer_url": "issuer_url",
        "anthropic_issuer_subject": "subject",
        "anthropic_issuer_audience": "audience",
        "anthropic_issuer_ttl_seconds": "ttl_seconds",
        "anthropic_issuer_signing_key_ref": "signing_key_ref",
    }
)
_KEYCLOAK_FIELD_MAP: Final[Mapping[str, str]] = MappingProxyType(
    {
        "anthropic_keycloak_token_url": "token_url",
        "anthropic_keycloak_client_id": "client_id",
        "anthropic_keycloak_auth_method": "auth_method",
        "anthropic_keycloak_client_secret_ref": "client_secret_ref",
        "anthropic_keycloak_scope": "scope",
    }
)
_WORKSPACE_HINT: Final = (
    " If the federation rule is scoped to a workspace, set ANTHROPIC_WORKSPACE_ID"
    " (or the anthropic_workspace_id litellm param) to that workspace id."
)
_ALLOWLIST_HINT: Final = (
    " Identity token files must sit under an allowed credential directory"
    " (/var/run/secrets or /run/secrets by default);"
    " set LITELLM_OIDC_ALLOWED_CREDENTIAL_DIRS to extend the allowlist."
)
_EMPTY_PARAMS: Final[Mapping[str, object]] = MappingProxyType({})

_IdentitySourceVariant = TypeVar("_IdentitySourceVariant", bound="InternalIssuerSource | KeycloakSource")


class AnthropicWifParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    federation_rule_id: str
    organization_id: str
    service_account_id: str | None = None
    workspace_id: str | None = None
    assertion_ref: str
    assertion_source: Callable[[], str | None] | None = None


def resolve_anthropic_wif_params(litellm_params: Mapping[str, object] | None) -> AnthropicWifParams | None:
    if litellm_params is not None and litellm_params.get(_DISABLE_WIF_PARAM) is True:
        return None
    federation_rule_id: Final = _config_value(
        litellm_params, "anthropic_federation_rule_id", "ANTHROPIC_FEDERATION_RULE_ID"
    )
    organization_id: Final = _config_value(litellm_params, "anthropic_organization_id", "ANTHROPIC_ORGANIZATION_ID")
    if federation_rule_id is None or organization_id is None:
        return None
    identity_source: Final = _resolve_identity_source(litellm_params)
    if identity_source is None:
        return None
    assertion_ref, assertion_source = identity_source
    return AnthropicWifParams(
        federation_rule_id=federation_rule_id,
        organization_id=organization_id,
        service_account_id=_config_value(
            litellm_params, "anthropic_service_account_id", "ANTHROPIC_SERVICE_ACCOUNT_ID"
        ),
        workspace_id=_config_value(litellm_params, "anthropic_workspace_id", "ANTHROPIC_WORKSPACE_ID"),
        assertion_ref=assertion_ref,
        assertion_source=assertion_source,
    )


def _resolve_identity_source(
    litellm_params: Mapping[str, object] | None,
) -> tuple[str, Callable[[], str] | None] | None:
    """Dispatches on ``anthropic_identity_source``. Absent (the default) keeps today's
    token_file/env resolution byte-identical, with no ``assertion_source`` closure -- the engine
    falls back to its own reader exactly as it does today. A recognized kind builds the matching
    frozen config, hashes it into the ``oidc/<kind>/<hash>`` cache-key ref (``identity_source_ref``),
    and closes the source's fetch/mint function over it. An unset-but-invalid config (unknown
    kind, a missing required field, or a field from the other variant) fails closed here rather
    than silently falling back to token_file."""
    source_kind: Final = _config_value(litellm_params, _IDENTITY_SOURCE_PARAM, _IDENTITY_SOURCE_ENV)
    if source_kind is None:
        legacy_ref: Final = _resolve_assertion_ref(litellm_params)
        return (legacy_ref, None) if legacy_ref is not None else None
    params: Final[Mapping[str, object]] = MappingProxyType(
        {key: value for key, value in (litellm_params or _EMPTY_PARAMS).items() if value is not None}
    )
    match source_kind:
        case AnthropicIdentitySourceKind.internal_issuer.value:
            _reject_foreign_variant_fields(params, foreign_field_map=_KEYCLOAK_FIELD_MAP, chosen_kind=source_kind)
            issuer_config: Final = _build_variant(InternalIssuerSource, params, _INTERNAL_ISSUER_FIELD_MAP)
            return identity_source_ref(issuer_config), internal_issuer_assertion_source(issuer_config)
        case AnthropicIdentitySourceKind.keycloak.value:
            _reject_foreign_variant_fields(
                params, foreign_field_map=_INTERNAL_ISSUER_FIELD_MAP, chosen_kind=source_kind
            )
            keycloak_config: Final = _build_variant(KeycloakSource, params, _KEYCLOAK_FIELD_MAP)
            return identity_source_ref(keycloak_config), keycloak_assertion_source(keycloak_config)
        case _:
            raise litellm.AuthenticationError(
                message=(
                    f"{_IDENTITY_SOURCE_PARAM} must be one of "
                    f"{', '.join(kind.value for kind in AnthropicIdentitySourceKind)}; got {source_kind!r}."
                ),
                llm_provider="anthropic",
                model="",
            )


def _reject_foreign_variant_fields(
    litellm_params: Mapping[str, object], foreign_field_map: Mapping[str, str], chosen_kind: str
) -> None:
    foreign_keys_present: Final = tuple(param for param in foreign_field_map if param in litellm_params)
    if foreign_keys_present:
        raise litellm.AuthenticationError(
            message=(
                f"{_IDENTITY_SOURCE_PARAM} is {chosen_kind!r}, but {', '.join(sorted(foreign_keys_present))} "
                "belongs to a different identity source and cannot be set alongside it."
            ),
            llm_provider="anthropic",
            model="",
        )


def _build_variant(
    model: type[_IdentitySourceVariant],
    litellm_params: Mapping[str, object],
    field_map: Mapping[str, str],
) -> _IdentitySourceVariant:
    fields: Final = MappingProxyType(
        {field_map[key]: value for key, value in litellm_params.items() if key in field_map}
    )
    try:
        return model.model_validate(fields)
    except ValidationError as e:
        # hide_input_in_errors=True on both variant models keeps a secret pasted into the
        # wrong field (e.g. a client_secret typed as signing_key_ref) out of str(e).
        raise litellm.AuthenticationError(
            message=f"Invalid {_IDENTITY_SOURCE_PARAM} configuration: {e}",
            llm_provider="anthropic",
            model="",
        ) from e


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
        assertion_source=params.assertion_source,
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
    exchange_base: Final = _token_exchange_base(api_base)
    _raise_if_exchange_host_untrusted(exchange_base, model)
    result: Final = engine.get_token(build_anthropic_wif_spec(params, exchange_base))
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
    exchange_base: Final = _token_exchange_base(api_base)
    _raise_if_exchange_host_untrusted(exchange_base, model)
    result: Final = await engine.aget_token(build_anthropic_wif_spec(params, exchange_base))
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
    return anthropic_base_without_chat_suffix(api_base if api_base is not None else _resolve_default_api_base())


def _trusted_exchange_hosts() -> frozenset[str]:
    """Hostnames a federated exchange may reach: Anthropic's own, plus whatever the operator put in
    the environment. Comma separated, case folded, entries given as a URL reduced to their host."""
    configured: Final = os.getenv(_TRUSTED_EXCHANGE_HOSTS_ENV) or ""
    extra: Final = (entry.strip() for entry in configured.split(",") if entry.strip())
    return frozenset(
        chain(
            (_DEFAULT_TRUSTED_EXCHANGE_HOST,),
            ((urlsplit(entry).hostname or entry.split("/")[0]).lower() for entry in extra),
        )
    )


def _raise_if_exchange_host_untrusted(exchange_base: str, model: str) -> None:
    """The federated exchange refuses any host the operator has not vouched for, whatever wrote the
    deployment's api_base. Exact hostname match, never a substring: ``api.anthropic.com.evil.test``
    contains the real host and must not pass."""
    host: Final = (urlsplit(exchange_base).hostname or "").lower()
    if host and host in _trusted_exchange_hosts():
        return
    raise litellm.AuthenticationError(
        message=(
            f"Anthropic workload identity federation refused to use host {host or exchange_base!r}. "
            f"A federated exchange sends the workload's identity token to this host and presents the "
            f"minted token to it, so only {_DEFAULT_TRUSTED_EXCHANGE_HOST} is trusted by default. To "
            f"use a private Anthropic-compatible gateway, add its hostname to the "
            f"{_TRUSTED_EXCHANGE_HOSTS_ENV} environment variable (comma separated); that is a "
            f"decision to trust it with org-scoped credentials, so it is deliberately server-owned "
            f"and cannot be set through the model or credential APIs."
        ),
        llm_provider="anthropic",
        model=model,
    )


def _resolve_default_api_base() -> str:
    from litellm.llms.anthropic.common_utils import AnthropicModelInfo

    return AnthropicModelInfo.get_api_base(None) or _DEFAULT_API_BASE


def anthropic_base_without_chat_suffix(base: str) -> str:
    """A deployment base with its chat-surface suffix removed, so the token URL and model
    discovery both derive from the same value whatever form the operator configured."""
    parts: Final = urlsplit(base)
    if not parts.scheme or not parts.netloc:
        return base.rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, _strip_path_suffixes(parts.path), "", ""))


def _strip_path_suffixes(path: str) -> str:
    """Drop the chat-surface suffixes a deployment base may carry, so every tier derives the same
    token URL. Each pass removes at most one suffix, so the loop is bounded by the segment count."""
    trimmed = path.rstrip("/")  # rebind-ok: fixed-point strip, one suffix per pass
    while True:
        shortened = next(  # rebind-ok: one suffix removed per iteration
            (trimmed.removesuffix(suffix) for suffix in _CHAT_BASE_SUFFIXES if trimmed.endswith(suffix)),
            trimmed,
        )
        if shortened == trimmed:
            return trimmed
        # Re-strip: a doubled suffix leaves a trailing slash that would stop the next match.
        trimmed = shortened.rstrip("/")


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
            base: Final = f"Could not obtain the OIDC identity token ({error.kind}) from {error.source_ref}."
            return f"{base} {error.detail}" if error.detail else base
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
