"""Microsoft Agent 365 governance guardrail for MCP tool calls.

Before the gateway executes an MCP tool, the pending call is sent to the
Agent 365 tool-evaluation endpoint, where Microsoft Defender scores it and
Agent 365 records it for observability. The returned allow/block verdict is
enforced here. Authentication is either the Entra On-Behalf-Of flow (the
caller's incoming bearer token, audienced to this gateway's app registration,
is exchanged for a delegated Agent 365 token, so Defender evaluates and audits
as the signed-in user) or a Microsoft Entra Agent ID (the gateway mints a
delegated token for the agent's own user account, so callers need no Entra
token).
"""

import asyncio
import hashlib
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Final, Literal, NoReturn

import httpx
from fastapi import HTTPException
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_proxy_logger
from litellm.exceptions import Timeout as LitellmTimeout
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.proxy.guardrails.guardrail_hooks.agent_365 import (
    AGENT_365_PROD_API_BASE,
    AGENT_365_PROD_RESOURCE_APP_ID,
    AGENT_365_SCOPE_NAME,
    Agent365GuardrailConfigModel,
)

if TYPE_CHECKING:
    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel
    from litellm.types.utils import GuardrailStatus

TOKEN_ENDPOINT_TEMPLATE: Final = "https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
EVALUATE_PATH: Final = "/agents/tool-evaluation/evaluate"
MCP_SESSION_ID_HEADER: Final = "mcp-session-id"
_MCP_CALL_TYPES: Final[tuple[str, ...]] = ("mcp_call", "call_mcp_tool")
_OBO_CACHE_MAX_ENTRIES: Final = 1000
_DEFAULT_TOKEN_TTL_SECONDS: Final = 3599.0
_TOKEN_EXPIRY_SLACK_SECONDS: Final = 60.0
_JWT_BEARER_ASSERTION_TYPE: Final = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
_AGENT_ID_EXCHANGE_SCOPE: Final = "api://AzureADTokenExchange/.default"
_AGENT_USER_CACHE_KEY: Final = "agent_user"


@dataclass(frozen=True, slots=True)
class AgentIdentityCredentials:
    client_id: str
    agent_user_upn: str


def _parse_expires_in(raw: object) -> float:
    if not isinstance(raw, (int, float, str)):
        return _DEFAULT_TOKEN_TTL_SECONDS
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_TOKEN_TTL_SECONDS


class _DefenderResult(TypedDict, total=False):
    status: ReadOnly[str]
    verdict: ReadOnly[str | None]
    message: ReadOnly[str | None]


class _EvaluateResponse(TypedDict, total=False):
    allowed: ReadOnly[bool]
    defender: ReadOnly[_DefenderResult]
    correlationId: ReadOnly[str]


class _ToolReference(TypedDict):
    name: ReadOnly[str]


class _UnavailableDetail(TypedDict):
    error: ReadOnly[str]
    message: ReadOnly[str]
    tool: ReadOnly[str]


class _BlockedDetail(TypedDict):
    error: ReadOnly[str]
    message: ReadOnly[str]
    tool: ReadOnly[str]
    correlation_id: ReadOnly[str | None]


class Agent365TokenExchangeError(Exception):
    def __init__(self, status_code: int, error_code: str, description: str) -> None:
        super().__init__(f"{error_code}: {description}")
        self.status_code = status_code
        self.error_code = error_code
        self.description = description


class Agent365MalformedResponseError(Exception):
    pass


class Agent365ThrottledError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class Agent365Guardrail(CustomGuardrail):
    """Pre-MCP-call guardrail enforcing Microsoft Agent 365 tool-evaluation verdicts."""

    records_own_guardrail_information: ClassVar[bool] = True

    def __init__(
        self,
        guardrail_name: str,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        api_base: str = AGENT_365_PROD_API_BASE,
        resource_app_id: str = AGENT_365_PROD_RESOURCE_APP_ID,
        agent_id: str | None = None,
        agent_identity: AgentIdentityCredentials | None = None,
        request_timeout: float = 10.0,
        unreachable_fallback: Literal["fail_closed", "fail_open"] = "fail_closed",
        async_handler: AsyncHTTPHandler | None = None,
        **kwargs,  # noqa: ANN003  # kwargs-ok: forwarded verbatim to CustomGuardrail (event_hook, default_on)
    ) -> None:
        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=self.get_supported_event_hooks(),
            **kwargs,
        )
        self.guardrail_provider = "agent_365"
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.api_base = api_base.rstrip("/")
        self.resource_app_id = resource_app_id
        self.agent_id = agent_id
        self.agent_identity = agent_identity
        self.request_timeout = request_timeout
        self.unreachable_fallback: Literal["fail_closed", "fail_open"] = (
            "fail_open" if unreachable_fallback == "fail_open" else "fail_closed"
        )
        self.async_handler = async_handler or get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self._obo_token_cache: OrderedDict[str, tuple[str, float]] = OrderedDict()  # mutable-ok: lock-guarded LRU
        self._obo_cache_lock = threading.Lock()
        self._agent_user_token_lock = asyncio.Lock()
        verbose_proxy_logger.info("Initialized Microsoft Agent 365 guardrail: %s", guardrail_name)

    @staticmethod
    def get_config_model() -> "type[GuardrailConfigModel] | None":
        return Agent365GuardrailConfigModel

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:  # mutable-ok: CustomGuardrail contract
        return [GuardrailEventHooks.pre_mcp_call]  # mutable-ok: CustomGuardrail contract expects a list

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        cache: "DualCache",
        data: dict,  # mutable-ok: hook contract; guardrail logging appends into the request metadata in place
        call_type: str,
    ) -> Exception | str | dict | None:  # mutable-ok: CustomGuardrail.async_pre_call_hook contract
        if call_type not in _MCP_CALL_TYPES:
            return data
        if "mcp_tool_name" not in data:
            return data
        if self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.pre_mcp_call) is not True:
            return data

        tool_name: Final = str(data.get("mcp_tool_name") or "")
        principal: Final = self._resolve_principal(data=data, tool_name=tool_name)

        try:
            token: Final = await (
                self._get_agent_user_token(principal)
                if isinstance(principal, AgentIdentityCredentials)
                else self._get_obo_token(principal)
            )
        except Agent365TokenExchangeError as exc:
            if isinstance(principal, AgentIdentityCredentials):
                return self._handle_unavailable(
                    data=data,
                    tool_name=tool_name,
                    reason=f"the Entra agent identity token request was rejected ({exc.error_code}: {exc.description})",
                )
            self._handle_caller_fault(
                data=data,
                tool_name=tool_name,
                status_code=401,
                reason=f"the Entra On-Behalf-Of token exchange was rejected ({exc.error_code})",
            )
        except Agent365ThrottledError as exc:
            self._handle_throttled(
                data=data,
                tool_name=tool_name,
                reason=f"the Entra token endpoint returned HTTP {exc.status_code}",
                latency_ms=None,
            )
        except (httpx.HTTPError, LitellmTimeout, TimeoutError) as exc:
            return self._handle_unavailable(
                data=data,
                tool_name=tool_name,
                reason=f"the Entra token endpoint could not be reached ({type(exc).__name__})",
            )
        except Agent365MalformedResponseError as exc:
            return self._handle_unavailable(
                data=data,
                tool_name=tool_name,
                reason=str(exc),
            )

        start: Final = time.perf_counter()
        try:
            response: Final = await self._post_allowing_error_status(
                url=f"{self.api_base}{EVALUATE_PATH}",
                json=self._build_evaluate_payload(data=data, user_api_key_dict=user_api_key_dict),
                headers={"Authorization": f"Bearer {token}"},  # mutable-ok: httpx header dict
            )
        except (httpx.HTTPError, LitellmTimeout, TimeoutError) as exc:
            return self._handle_unavailable(
                data=data,
                tool_name=tool_name,
                reason=f"the Agent 365 endpoint could not be reached ({type(exc).__name__})",
            )
        latency_ms: Final = (time.perf_counter() - start) * 1000.0
        fallback: Final = self._handle_evaluate_error(
            data=data,
            tool_name=tool_name,
            token_cache_key=self._token_cache_key(principal),
            token=token,
            response=response,
            latency_ms=latency_ms,
        )
        if fallback is not None:
            return fallback
        return self._enforce_verdict(data=data, tool_name=tool_name, response=response, latency_ms=latency_ms)

    def _resolve_principal(
        self,
        data: dict[str, object],  # mutable-ok: guardrail logging appends into the request metadata in place
        tool_name: str,
    ) -> str | AgentIdentityCredentials:
        if self.agent_identity is not None:
            return self.agent_identity
        assertion: Final = data.get("incoming_bearer_token")
        if isinstance(assertion, str) and assertion.count(".") == 2:
            return assertion
        self._handle_caller_fault(
            data=data,
            tool_name=tool_name,
            status_code=401,
            reason=(
                "the caller did not present an Entra bearer token; the Agent 365 guardrail "
                "authorizes tool calls On-Behalf-Of the signed-in user"
            ),
        )

    def _handle_evaluate_error(
        self,
        data: dict,  # mutable-ok: guardrail logging appends into the request metadata in place
        tool_name: str,
        token_cache_key: str,
        token: str,
        response: httpx.Response,
        latency_ms: float,
    ) -> dict | None:  # mutable-ok: returns the request data dict per hook contract on fail_open
        if response.status_code in (408, 429):
            self._handle_throttled(
                data=data,
                tool_name=tool_name,
                reason=f"the Agent 365 endpoint returned HTTP {response.status_code}",
                latency_ms=latency_ms,
            )
        if 400 <= response.status_code < 500:
            if response.status_code == 401:
                self._evict_token(token_cache_key, token)
            self._record_verdict(
                data=data,
                verdict="Rejected",
                guardrail_status="guardrail_intervened",
                defender_status=None,
                correlation_id=None,
                latency_ms=latency_ms,
                reason=f"HTTP {response.status_code}: {response.text[:512]}",
            )
            rejected_detail: Final[_UnavailableDetail] = {
                "error": "Agent 365 rejected the tool evaluation request",
                "message": response.text[:512]
                if response.status_code == 400
                else f"the Agent 365 evaluation request failed with HTTP {response.status_code}",
                "tool": tool_name,
            }
            raise HTTPException(status_code=400, detail=rejected_detail)
        if response.status_code != 200:
            return self._handle_unavailable(
                data=data,
                tool_name=tool_name,
                reason=f"the Agent 365 endpoint returned HTTP {response.status_code}",
            )
        return None

    def _enforce_verdict(
        self,
        data: dict,  # mutable-ok: guardrail logging appends into the request metadata in place
        tool_name: str,
        response: httpx.Response,
        latency_ms: float,
    ) -> dict:  # mutable-ok: returns the request data dict per hook contract
        try:
            parsed_verdict: Final = response.json()
        except ValueError:
            return self._handle_unavailable(
                data=data,
                tool_name=tool_name,
                reason="the Agent 365 endpoint returned a non-JSON body",
            )
        if not isinstance(parsed_verdict, dict):
            return self._handle_unavailable(
                data=data,
                tool_name=tool_name,
                reason="the Agent 365 endpoint returned a non-object JSON body",
            )
        verdict: Final[_EvaluateResponse] = parsed_verdict
        allowed: Final = verdict.get("allowed")
        if not isinstance(allowed, bool):
            return self._handle_unavailable(
                data=data,
                tool_name=tool_name,
                reason="the Agent 365 endpoint returned a verdict without a boolean 'allowed' field",
            )
        raw_defender: Final = verdict.get("defender")
        defender: Final = raw_defender if isinstance(raw_defender, dict) else _DefenderResult()
        raw_correlation_id: Final = verdict.get("correlationId")
        correlation_id: Final = raw_correlation_id if isinstance(raw_correlation_id, str) else None
        self._record_verdict(
            data=data,
            verdict="Allow" if allowed else "Block",
            guardrail_status="success" if allowed else "guardrail_intervened",
            defender_status=defender.get("status"),
            correlation_id=correlation_id,
            latency_ms=latency_ms,
        )
        if not allowed:
            blocked_detail: Final[_BlockedDetail] = {
                "error": "Blocked by Microsoft Defender",
                "message": (
                    defender.get("message")
                    or f"Invocation of '{tool_name}' is blocked by Microsoft Threat Detection policies "
                    "configured by your administrator."
                ),
                "tool": tool_name,
                "correlation_id": correlation_id,
            }
            raise HTTPException(status_code=400, detail=blocked_detail)
        return data

    def _build_evaluate_payload(
        self,
        data: Mapping[str, object],
        user_api_key_dict: "UserAPIKeyAuth",
    ) -> dict[str, object]:  # mutable-ok: JSON body for AsyncHTTPHandler.post, which requires dict
        tool_name: Final = str(data.get("mcp_tool_name") or "")
        arguments: Final = data.get("mcp_arguments")
        server_name: Final = str(data.get("mcp_server_name") or "litellm")
        agent_id: Final = self.agent_id or user_api_key_dict.key_alias
        tool_reference: Final[_ToolReference] = {"name": tool_name}
        payload: Final[dict[str, object]] = {  # mutable-ok: JSON body with optional fields added below
            "tool": tool_reference,
            "serverName": server_name,
            "conversationId": self._resolve_conversation_id(data),
        }
        if isinstance(arguments, dict):
            payload["arguments"] = arguments
        if agent_id:
            payload["agentId"] = str(agent_id)
        return payload

    @staticmethod
    def _resolve_conversation_id(data: Mapping[str, object]) -> str:
        metadata: Final = next(
            (m for m in (data.get("metadata"), data.get("litellm_metadata")) if isinstance(m, Mapping)),
            None,
        )
        headers: Final = metadata.get("headers") if isinstance(metadata, Mapping) else None
        if isinstance(headers, Mapping):
            session_id: Final = next(
                (value for name, value in headers.items() if str(name).lower() == MCP_SESSION_ID_HEADER),
                None,
            )
            if isinstance(session_id, str) and session_id:
                return session_id
        raw_logging_obj: Final = data.get("litellm_logging_obj")
        logging_obj: Final = raw_logging_obj if isinstance(raw_logging_obj, LiteLLMLoggingObj) else None
        if logging_obj is not None:
            tool_call_metadata: Final = logging_obj.model_call_details.get("mcp_tool_call_metadata")
            session_from_logging: Final = (
                tool_call_metadata.get("mcp_session_id") if isinstance(tool_call_metadata, Mapping) else None
            )
            if isinstance(session_from_logging, str) and session_from_logging:
                return session_from_logging
        call_id: Final = data.get("litellm_call_id") or (logging_obj.litellm_call_id if logging_obj else None)
        if isinstance(call_id, str) and call_id:
            return call_id
        return str(uuid.uuid4())

    @staticmethod
    def _token_cache_key(principal: str | AgentIdentityCredentials) -> str:
        if isinstance(principal, AgentIdentityCredentials):
            return _AGENT_USER_CACHE_KEY
        return hashlib.sha256(principal.encode("utf-8")).hexdigest()

    def _cached_token(self, cache_key: str) -> str | None:
        now: Final = time.time()
        with self._obo_cache_lock:
            cached: Final = self._obo_token_cache.get(cache_key)
            if cached and cached[1] > now + _TOKEN_EXPIRY_SLACK_SECONDS:
                self._obo_token_cache.move_to_end(cache_key)
                return cached[0]
        return None

    def _store_token(self, cache_key: str, access_token: str, expires_at: float) -> None:
        with self._obo_cache_lock:
            self._obo_token_cache[cache_key] = (access_token, expires_at)
            self._obo_token_cache.move_to_end(cache_key)
            while len(self._obo_token_cache) > _OBO_CACHE_MAX_ENTRIES:
                self._obo_token_cache.popitem(last=False)

    async def _get_obo_token(self, assertion: str) -> str:
        cache_key: Final = self._token_cache_key(assertion)
        cached: Final = self._cached_token(cache_key)
        if cached is not None:
            return cached
        access_token, expires_at = await self._request_token(
            {  # mutable-ok: OAuth form body; AsyncHTTPHandler.post requires dict
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "assertion": assertion,
                "scope": f"{self.resource_app_id}/{AGENT_365_SCOPE_NAME}",
                "requested_token_use": "on_behalf_of",
            }
        )
        self._store_token(cache_key, access_token, expires_at)
        return access_token

    async def _get_agent_user_token(self, agent_identity: AgentIdentityCredentials) -> str:
        """Entra Agent ID chain: blueprint (client secret) -> agent identity (FIC) -> agent user (user_fic)."""
        async with self._agent_user_token_lock:
            cached: Final = self._cached_token(_AGENT_USER_CACHE_KEY)
            if cached is not None:
                return cached
            blueprint_token, _ = await self._request_token(
                {  # mutable-ok: OAuth form body; AsyncHTTPHandler.post requires dict
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "scope": _AGENT_ID_EXCHANGE_SCOPE,
                    "fmi_path": agent_identity.client_id,
                }
            )
            agent_identity_token, _ = await self._request_token(
                {  # mutable-ok: OAuth form body; AsyncHTTPHandler.post requires dict
                    "grant_type": "client_credentials",
                    "client_id": agent_identity.client_id,
                    "client_assertion_type": _JWT_BEARER_ASSERTION_TYPE,
                    "client_assertion": blueprint_token,
                    "scope": _AGENT_ID_EXCHANGE_SCOPE,
                }
            )
            access_token, expires_at = await self._request_token(
                {  # mutable-ok: OAuth form body; AsyncHTTPHandler.post requires dict
                    "grant_type": "user_fic",
                    "client_id": agent_identity.client_id,
                    "client_assertion_type": _JWT_BEARER_ASSERTION_TYPE,
                    "client_assertion": blueprint_token,
                    "user_federated_identity_credential": agent_identity_token,
                    "username": agent_identity.agent_user_upn,
                    "scope": f"{self.resource_app_id}/{AGENT_365_SCOPE_NAME}",
                    "requested_token_use": "on_behalf_of",
                }
            )
            self._store_token(_AGENT_USER_CACHE_KEY, access_token, expires_at)
            return access_token

    async def _request_token(
        self,
        form: dict[str, str],  # mutable-ok: OAuth form body; AsyncHTTPHandler.post requires dict
    ) -> tuple[str, float]:
        response: Final = await self._post_allowing_error_status(
            url=TOKEN_ENDPOINT_TEMPLATE.format(tenant_id=self.tenant_id),
            data=form,
            headers={"Content-Type": "application/x-www-form-urlencoded"},  # mutable-ok: httpx header dict
        )
        if response.status_code in (408, 429):
            raise Agent365ThrottledError(status_code=response.status_code)
        if response.status_code >= 500:
            raise httpx.HTTPStatusError(
                f"Entra token endpoint returned {response.status_code}",
                request=response.request,
                response=response,
            )
        try:
            parsed_body: Final = response.json()
        except ValueError as exc:
            raise Agent365MalformedResponseError("the Entra token endpoint returned a non-JSON body") from exc
        if not isinstance(parsed_body, dict):
            raise Agent365MalformedResponseError("the Entra token endpoint returned a non-object JSON body")
        body: Final = parsed_body
        if response.status_code >= 400:
            raise Agent365TokenExchangeError(
                status_code=response.status_code,
                error_code=str(body.get("error", "invalid_grant")),
                description=str(body.get("error_description", ""))[:512],
            )
        if "access_token" not in body:
            raise Agent365MalformedResponseError("the Entra token endpoint returned no access_token")
        raw_access_token: Final = body.get("access_token")
        if not isinstance(raw_access_token, str) or not raw_access_token:
            raise Agent365MalformedResponseError("the Entra token endpoint returned a non-string access_token")
        return raw_access_token, time.time() + _parse_expires_in(body.get("expires_in", 3599))

    async def _post_allowing_error_status(
        self,
        url: str,
        headers: dict[str, str],  # mutable-ok: AsyncHTTPHandler.post requires dict
        data: dict[str, str] | None = None,  # mutable-ok: AsyncHTTPHandler.post requires dict
        json: dict[str, object] | None = None,  # mutable-ok: AsyncHTTPHandler.post requires dict
    ) -> httpx.Response:
        try:
            return await self.async_handler.post(
                url=url,
                data=data,
                json=json,
                headers=headers,
                timeout=self.request_timeout,
            )
        except httpx.HTTPStatusError as exc:
            return exc.response

    def _handle_caller_fault(
        self,
        data: dict,  # mutable-ok: guardrail logging appends into the request metadata in place
        tool_name: str,
        status_code: int,
        reason: str,
    ) -> NoReturn:
        self._record_verdict(
            data=data,
            verdict="Rejected",
            guardrail_status="guardrail_intervened",
            defender_status=None,
            correlation_id=None,
            latency_ms=None,
            reason=reason,
        )
        caller_fault_detail: Final[_UnavailableDetail] = {
            "error": "Agent 365 guardrail rejected the tool call",
            "message": f"Tool call '{tool_name}' was blocked because {reason}.",
            "tool": tool_name,
        }
        raise HTTPException(status_code=status_code, detail=caller_fault_detail)

    def _handle_throttled(
        self,
        data: dict,  # mutable-ok: guardrail logging appends into the request metadata in place
        tool_name: str,
        reason: str,
        latency_ms: float | None,
    ) -> NoReturn:
        self._record_verdict(
            data=data,
            verdict="Throttled",
            guardrail_status="guardrail_failed_to_respond",
            defender_status=None,
            correlation_id=None,
            latency_ms=latency_ms,
            reason=reason,
        )
        throttled_detail: Final[_UnavailableDetail] = {
            "error": "Agent 365 guardrail could not authorize the tool call",
            "message": f"Tool call '{tool_name}' was blocked because {reason}; "
            "throttled evaluations block regardless of unreachable_fallback.",
            "tool": tool_name,
        }
        raise HTTPException(status_code=503, detail=throttled_detail)

    def _evict_token(self, cache_key: str, token: str) -> None:
        with self._obo_cache_lock:
            cached: Final = self._obo_token_cache.get(cache_key)
            if cached is not None and cached[0] == token:
                self._obo_token_cache.pop(cache_key)

    def _handle_unavailable(
        self,
        data: dict[str, object],  # mutable-ok: guardrail logging appends into the request metadata in place
        tool_name: str,
        reason: str,
    ) -> dict[str, object]:  # mutable-ok: returns the request data dict per hook contract
        if self.unreachable_fallback == "fail_open":
            verbose_proxy_logger.warning(
                "Agent 365 guardrail (%s): %s; unreachable_fallback='fail_open', allowing tool call '%s' unscanned",
                self.guardrail_name,
                reason,
                tool_name,
            )
            self._record_verdict(
                data=data,
                verdict="Unscanned",
                guardrail_status="guardrail_failed_to_respond",
                defender_status=None,
                correlation_id=None,
                latency_ms=None,
                reason=reason,
            )
            return data
        self._record_verdict(
            data=data,
            verdict="Unavailable",
            guardrail_status="guardrail_failed_to_respond",
            defender_status=None,
            correlation_id=None,
            latency_ms=None,
            reason=reason,
        )
        unavailable_detail: Final[_UnavailableDetail] = {
            "error": "Agent 365 guardrail could not authorize the tool call",
            "message": f"Tool call '{tool_name}' was blocked because {reason} and unreachable_fallback is "
            "'fail_closed'.",
            "tool": tool_name,
        }
        raise HTTPException(status_code=503, detail=unavailable_detail)

    def _record_verdict(
        self,
        data: dict[str, object],  # mutable-ok: standard guardrail logging appends into the request metadata in place
        verdict: str,
        guardrail_status: "GuardrailStatus",
        defender_status: str | None,
        correlation_id: str | None,
        latency_ms: float | None,
        reason: str | None = None,
    ) -> None:
        payload: Final[dict[str, object]] = {"verdict": verdict}  # mutable-ok: optional fields added below
        if defender_status:
            payload["defender_status"] = defender_status
        if correlation_id:
            payload["correlation_id"] = correlation_id
        if latency_ms is not None:
            payload["latency_ms"] = round(latency_ms, 1)
        if reason:
            payload["reason"] = reason
        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_json_response=payload,
            request_data=data,
            guardrail_status=guardrail_status,
            duration=(latency_ms / 1000.0) if latency_ms is not None else None,
            guardrail_provider=self.guardrail_provider,
            event_type=GuardrailEventHooks.pre_mcp_call,
        )
