import base64
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, NoReturn

import httpx
from pydantic import SecretStr, TypeAdapter, ValidationError

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.mcp_server.faults.traversal import iter_exception_tree
from litellm.proxy.mcp_server.outbound_credentials.adapter import raise_public, to_subject
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok
from litellm.proxy.mcp_server.outbound_credentials.token_exchange_provider import build_token_exchanger
from litellm.proxy.mcp_server.outbound_credentials.token_exchanger import TokenExchanger
from litellm.proxy.mcp_server.outbound_credentials.types import CredError, ServerSpec, TokenExchangeConfig
from litellm.secret_managers.main import get_secret_str

TOKEN_EXCHANGE_PARAM: Final = "token_exchange"
TOKEN_EXCHANGE_PLAIN_FIELDS: Final = frozenset(
    {"token_exchange_endpoint", "subject_token_type", "token_endpoint_auth_method"}
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
_REQUIRED_FIELDS: Final = ("token_exchange_endpoint", "client_id", "client_secret")
_BLOCK: Final = TypeAdapter(Mapping[str, object])
_ITEMS: Final = TypeAdapter(tuple[object, ...])
_EXCHANGER: Final = build_token_exchanger()
_REJECTIONS: Final = MappingProxyType(
    {
        401: "unauthorized: agent '{}' rejected the exchanged token",
        403: "forbidden: agent '{}' refused the request made with the exchanged token",
    }
)


class ExchangedTokenRejected(Exception):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code: Final = status_code
        self.message: Final = message


@dataclass(frozen=True, slots=True)
class ExchangedAgentToken:
    header: Mapping[str, str] = field(repr=False)
    subject_token: SecretStr
    server: ServerSpec
    config: TokenExchangeConfig
    tenant_id: str
    exchanger: TokenExchanger = field(repr=False)

    async def rejection(self, error: BaseException) -> ExchangedTokenRejected | None:
        status: Final = _rejected_status(error)
        if status is None:
            return None
        if status == 401:
            await self.exchanger.invalidate(
                self.subject_token.get_secret_value(), self.server, self.config, tenant_id=self.tenant_id
            )
        return ExchangedTokenRejected(status, _REJECTIONS[status].format(self.server.server_id))


def _rejected_status(error: BaseException) -> int | None:
    statuses: Final = (
        link.response.status_code for link in iter_exception_tree(error) if isinstance(link, httpx.HTTPStatusError)
    )
    return next((status for status in statuses if status in _REJECTIONS), None)


def has_token_exchange(litellm_params: Mapping[str, object]) -> bool:
    block: Final = litellm_params.get(TOKEN_EXCHANGE_PARAM)
    return block is not None and block is not False


def without_token_exchange(
    litellm_params: Mapping[str, object],
) -> dict[str, object]:  # mutable-ok: asend_message takes litellm_params as a dict
    return {key: value for key, value in litellm_params.items() if key != TOKEN_EXCHANGE_PARAM}


def _expand(value: object) -> object:
    return get_secret_str(value) if isinstance(value, str) and value.startswith("os.environ/") else value


def _expand_field(value: object) -> object:
    return tuple(_expand(item) for item in _ITEMS.validate_python(value)) if isinstance(value, list) else _expand(value)


def _unresolved(raw: object, expanded: object) -> bool:
    if isinstance(expanded, tuple):
        return None in expanded
    return expanded is None and raw is not None


def _token_exchange_config(litellm_params: Mapping[str, object]) -> TokenExchangeConfig | str:
    try:
        block: Final = _BLOCK.validate_python(litellm_params.get(TOKEN_EXCHANGE_PARAM))
    except ValidationError:
        return "it must be a mapping"
    unknown: Final = tuple(sorted(name for name in block if name not in _CONFIG_FIELDS))
    if unknown:
        return f"unknown field(s) {', '.join(unknown)}"
    fields: Final = {name: _expand_field(block[name]) for name in _CONFIG_FIELDS if name in block}
    unresolved: Final = tuple(name for name, value in fields.items() if _unresolved(block[name], value))
    if unresolved:
        return f"unresolved os.environ/ reference in {', '.join(unresolved)}"
    missing: Final = tuple(name for name in _REQUIRED_FIELDS if not fields.get(name))
    if missing:
        return f"missing {', '.join(missing)}"
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
) -> ExchangedAgentToken:
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
            return ExchangedAgentToken(
                header=MappingProxyType({name: value}),
                subject_token=SecretStr(subject_token),
                server=server,
                config=config,
                tenant_id=tenant_id,
                exchanger=exchanger,
            )
        case Error(err):
            _raise_challenge(err)
