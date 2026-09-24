from base64 import b64encode
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias

import httpx
from typing_extensions import assert_never

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.constants import (
    AGENT_KILL_SWITCH_RESPONSE_BODY_MAX_CHARS,
    AGENT_KILL_SWITCH_TIMEOUT_SECONDS,
    REDACTED_BY_LITELM_STRING,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # its params arg is a bare dict in http_handler
)
from litellm.proxy._types import LiteLLM_AuditLogs, LitellmTableNames, UserAPIKeyAuth
from litellm.proxy.management_helpers.audit_logs import create_audit_log_for_update, get_audit_log_changed_by
from litellm.types.agents import (
    AgentKillSwitchApiKeyAuth,
    AgentKillSwitchAuth,
    AgentKillSwitchBasicAuth,
    AgentKillSwitchBearerAuth,
    AgentKillSwitchConfig,
    AgentKillSwitchResult,
)
from litellm.types.llms.custom_http import httpxSpecialProvider


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
    def build_request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, object] | None,
        timeout: float,
    ) -> httpx.Request: ...

    async def send(self, request: httpx.Request, *, stream: bool, follow_redirects: bool) -> httpx.Response: ...


def default_kill_switch_http_client() -> KillSwitchHttpClient:
    return get_async_httpx_client(llm_provider=httpxSpecialProvider.AgentKillSwitch).client


KillSwitchAuditLogWriter: TypeAlias = Callable[[LiteLLM_AuditLogs], Awaitable[None]]  # mutable-ok: Callable params


def default_kill_switch_audit_log_writer() -> KillSwitchAuditLogWriter:
    return create_audit_log_for_update


def build_kill_switch_audit_log(
    *,
    result: AgentKillSwitchResult,
    user_api_key_dict: UserAPIKeyAuth,
    litellm_proxy_admin_name: str | None,
) -> LiteLLM_AuditLogs:
    return LiteLLM_AuditLogs(
        id=str(uuid.uuid4()),
        updated_at=datetime.now(timezone.utc),
        changed_by=get_audit_log_changed_by(
            litellm_changed_by=None,
            user_api_key_dict=user_api_key_dict,
            litellm_proxy_admin_name=litellm_proxy_admin_name,
        ),
        changed_by_api_key=user_api_key_dict.api_key,
        table_name=LitellmTableNames.AGENT_TABLE_NAME,
        object_id=result.agent_id,
        action="kill_switch_fired",
        updated_values=result.model_dump_json(exclude_none=True),
    )


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
        response: Final = await http_client.send(
            http_client.build_request(
                request.method,
                request.url,
                headers=request.headers,
                json=request.json_body,
                timeout=timeout,
            ),
            stream=True,
            follow_redirects=False,
        )
        body: Final = await _read_text_prefix(response, AGENT_KILL_SWITCH_RESPONSE_BODY_MAX_CHARS)
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
        response_body=body,
    )


async def _read_text_prefix(response: httpx.Response, max_chars: int) -> str:
    try:
        return await _take_text(response.aiter_text(), max_chars)
    finally:
        await response.aclose()


async def _take_text(chunks: AsyncIterator[str], max_chars: int) -> str:
    taken = ""  # rebind-ok: running prefix of a stream that is abandoned once the cap is hit
    async for chunk in chunks:
        taken += chunk  # rebind-ok: see above
        if len(taken) >= max_chars:
            break
    return taken[:max_chars]
