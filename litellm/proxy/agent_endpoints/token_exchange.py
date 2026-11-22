import base64
from collections.abc import Mapping
from typing import Final, NoReturn

from pydantic import TypeAdapter, ValidationError

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.mcp_server.outbound_credentials.adapter import raise_public, to_subject
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok
from litellm.proxy.mcp_server.outbound_credentials.token_exchange_provider import build_token_exchanger
from litellm.proxy.mcp_server.outbound_credentials.token_exchanger import TokenExchanger
from litellm.proxy.mcp_server.outbound_credentials.types import CredError, ServerSpec, TokenExchangeConfig
from litellm.secret_managers.main import get_secret_str

TOKEN_EXCHANGE_PARAM: Final = "token_exchange"
TOKEN_EXCHANGE_PLAIN_KEYS: Final = frozenset(
    {TOKEN_EXCHANGE_PARAM, "token_exchange_endpoint", "subject_token_type", "token_endpoint_auth_method"}
)
_CONFIG_FIELDS: Final = (
    "profile",
    "token_exchange_endpoint",
    "client_id",
    "client_secret",
    "audience",
    "scopes",
    "subject_token_type",
    "token_endpoint_auth_method",
)
_BLOCK: Final = TypeAdapter(Mapping[str, object])
_EXCHANGER: Final = build_token_exchanger()


def has_token_exchange(litellm_params: Mapping[str, object]) -> bool:
    return bool(litellm_params.get(TOKEN_EXCHANGE_PARAM))


def without_token_exchange(
    litellm_params: Mapping[str, object],
) -> dict[str, object]:  # mutable-ok: asend_message takes litellm_params as a dict
    return {key: value for key, value in litellm_params.items() if key != TOKEN_EXCHANGE_PARAM}


def _expand(value: object) -> object:
    return get_secret_str(value) if isinstance(value, str) and value.startswith("os.environ/") else value


def _token_exchange_config(litellm_params: Mapping[str, object]) -> TokenExchangeConfig | str:
    try:
        block: Final = _BLOCK.validate_python(litellm_params.get(TOKEN_EXCHANGE_PARAM))
    except ValidationError:
        return "it must be a mapping"
    fields: Final = {name: _expand(block[name]) for name in _CONFIG_FIELDS if name in block}
    unresolved: Final = tuple(name for name, value in fields.items() if value is None and block[name] is not None)
    if unresolved:
        return f"unresolved os.environ/ reference in {', '.join(unresolved)}"
    try:
        return TokenExchangeConfig.model_validate(fields)
    except ValidationError:
        return "a field has an unsupported value"


def _raise_challenge(error: CredError) -> NoReturn:
    if error.tag != "unauthorized" or error.unauthorized.www_authenticate:
        raise_public(error)
    claims: Final = error.unauthorized.claims
    challenge: Final = (
        f'Bearer error="insufficient_claims", claims="{base64.b64encode(claims.encode()).decode()}"'
        if claims
        else 'Bearer error="invalid_token"'
    )
    raise_public(CredError.of_unauthorized(error.unauthorized.detail, www_authenticate=challenge))


async def exchange_agent_token(
    litellm_params: Mapping[str, object],
    agent_id: str,
    resource: str,
    request_headers: Mapping[str, str],
    user_api_key_auth: UserAPIKeyAuth,
    exchanger: TokenExchanger = _EXCHANGER,
) -> Mapping[str, str]:
    from litellm.proxy.mcp_server.mcp_server_manager import MCPServerManager

    config: Final = _token_exchange_config(litellm_params)
    if isinstance(config, str):
        raise_public(
            CredError.of_misconfigured(f"agent '{agent_id}' has an invalid {TOKEN_EXCHANGE_PARAM} block: {config}")
        )
    subject_token: Final = MCPServerManager.extract_subject_token(None, request_headers, user_api_key_auth)
    if subject_token is None:
        raise_public(
            CredError.of_unauthorized(
                "Token exchange requires a caller token to exchange (OBO).",
                www_authenticate='Bearer error="invalid_request"',
            )
        )
    server: Final = ServerSpec(server_id=agent_id, resource=resource, config=config)
    tenant_id: Final = to_subject(user_api_key_auth, subject_token).tenant_id
    match await exchanger.exchange(subject_token, server, config, tenant_id=tenant_id):
        case Ok(token):
            name, value = config.header(token.access_token)
            return {name: value}
        case Error(err):
            _raise_challenge(err)
