from base64 import b64encode
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Final, Protocol

import httpx
from typing_extensions import assert_never

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    AGENT_KILL_SWITCH_RESPONSE_BODY_MAX_CHARS,
    AGENT_KILL_SWITCH_TIMEOUT_SECONDS,
    REDACTED_BY_LITELM_STRING,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.types.agents import (
    AgentKillSwitchApiKeyAuth,
    AgentKillSwitchAuth,
    AgentKillSwitchBasicAuth,
    AgentKillSwitchBearerAuth,
    AgentKillSwitchConfig,
    AgentKillSwitchResult,
)


def _with_auth(config: AgentKillSwitchConfig, auth: AgentKillSwitchAuth) -> AgentKillSwitchConfig:
    return AgentKillSwitchConfig(
        url=config.url,
        method=config.method,
        headers=config.headers,
        query_params=config.query_params,
        body=config.body,
        auth=auth,
    )


def redact_kill_switch(config: AgentKillSwitchConfig | None) -> AgentKillSwitchConfig | None:
    if config is None or config.auth is None:
        return config
    return _with_auth(config, _redact_auth(config.auth))


def _redact_auth(auth: AgentKillSwitchAuth) -> AgentKillSwitchAuth:
    match auth:
        case AgentKillSwitchBearerAuth():
            return AgentKillSwitchBearerAuth(type="bearer", token=REDACTED_BY_LITELM_STRING)
        case AgentKillSwitchApiKeyAuth():
            return AgentKillSwitchApiKeyAuth(
                type="api_key", header_name=auth.header_name, api_key=REDACTED_BY_LITELM_STRING
            )
        case AgentKillSwitchBasicAuth():
            return AgentKillSwitchBasicAuth(type="basic", username=auth.username, password=REDACTED_BY_LITELM_STRING)
        case _:
            assert_never(auth)


def restore_kill_switch(
    incoming: AgentKillSwitchConfig | None,
    existing: AgentKillSwitchConfig | None,
) -> AgentKillSwitchConfig | None:
    """Put the stored secret back behind an auth field echoed as the redaction
    marker; a marker with no stored secret of the same auth type becomes ""."""
    if incoming is None or incoming.auth is None:
        return incoming
    existing_auth: Final = existing.auth if existing is not None else None
    return _with_auth(incoming, _restore_auth(incoming.auth, existing_auth))


def _restore_secret(incoming_value: str, existing_value: str | None) -> str:
    if incoming_value != REDACTED_BY_LITELM_STRING:
        return incoming_value
    return existing_value if existing_value is not None else ""


def _restore_auth(incoming: AgentKillSwitchAuth, existing: AgentKillSwitchAuth | None) -> AgentKillSwitchAuth:
    match incoming:
        case AgentKillSwitchBearerAuth():
            stored_token: Final = existing.token if isinstance(existing, AgentKillSwitchBearerAuth) else None
            return AgentKillSwitchBearerAuth(type="bearer", token=_restore_secret(incoming.token, stored_token))
        case AgentKillSwitchApiKeyAuth():
            stored_key: Final = existing.api_key if isinstance(existing, AgentKillSwitchApiKeyAuth) else None
            return AgentKillSwitchApiKeyAuth(
                type="api_key",
                header_name=incoming.header_name,
                api_key=_restore_secret(incoming.api_key, stored_key),
            )
        case AgentKillSwitchBasicAuth():
            stored_password: Final = existing.password if isinstance(existing, AgentKillSwitchBasicAuth) else None
            return AgentKillSwitchBasicAuth(
                type="basic",
                username=incoming.username,
                password=_restore_secret(incoming.password, stored_password),
            )
        case _:
            assert_never(incoming)


@dataclass(frozen=True, slots=True)
class KillSwitchRequest:
    method: str
    url: str
    headers: Mapping[str, str]
    json_body: Mapping[str, object] | None


def _auth_headers(auth: AgentKillSwitchAuth | None) -> Mapping[str, str]:
    match auth:
        case None:
            return MappingProxyType({})
        case AgentKillSwitchBearerAuth():
            return MappingProxyType({"Authorization": f"Bearer {auth.token}"})
        case AgentKillSwitchApiKeyAuth():
            return MappingProxyType({auth.header_name: auth.api_key})
        case AgentKillSwitchBasicAuth():
            credentials: Final = b64encode(f"{auth.username}:{auth.password}".encode()).decode()
            return MappingProxyType({"Authorization": f"Basic {credentials}"})
        case _:
            assert_never(auth)


def build_kill_switch_request(config: AgentKillSwitchConfig) -> KillSwitchRequest:
    url: Final = httpx.URL(config.url).copy_merge_params(config.query_params)
    return KillSwitchRequest(
        method=config.method,
        url=str(url),
        headers=MappingProxyType({**config.headers, **_auth_headers(config.auth)}),
        json_body=config.body,
    )


class KillSwitchHttpClient(Protocol):
    async def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, object] | None,
        timeout: float,
    ) -> httpx.Response: ...


@lru_cache(maxsize=1)
def default_kill_switch_http_client() -> KillSwitchHttpClient:
    return AsyncHTTPHandler(
        timeout=AGENT_KILL_SWITCH_TIMEOUT_SECONDS, client_alias="agent_kill_switch", follow_redirects=False
    ).client


async def fire_kill_switch(
    *,
    agent_id: str,
    config: AgentKillSwitchConfig,
    http_client: KillSwitchHttpClient,
    timeout: float = AGENT_KILL_SWITCH_TIMEOUT_SECONDS,
) -> AgentKillSwitchResult:
    request: Final = build_kill_switch_request(config)
    reported_url: Final = str(httpx.URL(request.url).copy_with(query=None))
    verbose_proxy_logger.info("Firing kill switch for agent %s: %s %s", agent_id, request.method, reported_url)
    try:
        response: Final = await http_client.request(
            request.method,
            request.url,
            headers=request.headers,
            json=request.json_body,
            timeout=timeout,
        )
    except httpx.HTTPError as exc:
        verbose_proxy_logger.warning("Kill switch for agent %s failed: %s", agent_id, type(exc).__name__)
        return AgentKillSwitchResult(
            agent_id=agent_id,
            url=reported_url,
            method=config.method,
            error=type(exc).__name__,
        )
    return AgentKillSwitchResult(
        agent_id=agent_id,
        url=reported_url,
        method=config.method,
        status_code=response.status_code,
        response_body=response.text[:AGENT_KILL_SWITCH_RESPONSE_BODY_MAX_CHARS],
    )
