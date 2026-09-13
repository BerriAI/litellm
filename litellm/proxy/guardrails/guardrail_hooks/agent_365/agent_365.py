"""Microsoft Agent 365 governance guardrail for MCP tool calls.

Before the gateway executes an MCP tool, the pending call is sent to the
Agent 365 tool-evaluation endpoint, where Microsoft Defender scores it and
Agent 365 records it for observability. The returned allow/block verdict is
enforced here. Authentication is the Entra On-Behalf-Of flow: the caller's
incoming bearer token (audienced to this gateway's app registration) is
exchanged for a delegated Agent 365 token, so Defender evaluates and audits
as the signed-in user.
"""

import hashlib
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, Final, Literal, NoReturn

import httpx
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError
from typing_extensions import ReadOnly, TypedDict

import litellm
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
from litellm.proxy.litellm_pre_call_utils import add_guardrails_from_auth_metadata
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.mcp import MCPAuth
from litellm.types.mcp_server.mcp_server_manager import MCPServer
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
ENTRA_ISSUER_TEMPLATE: Final = "https://login.microsoftonline.com/{tenant_id}/v2.0"
EVALUATE_PATH: Final = "/agents/tool-evaluation/evaluate"
MCP_SESSION_ID_HEADER: Final = "mcp-session-id"
DEFENDER_STATUS_EVALUATED: Final = "Evaluated"
GATEWAY_SCOPE_TEMPLATE: Final = "api://{client_id}/access_as_user"
_GATEWAY_OWNED_TOKEN_ERRORS: Final = frozenset(
    {"invalid_client", "unauthorized_client", "invalid_scope", "invalid_resource"}
)
_MCP_CALL_TYPES: Final[tuple[str, ...]] = ("mcp_call", "call_mcp_tool")
_TOOL_INPUT_SCHEMA_ADAPTER: Final = TypeAdapter(dict[str, object])
_OBO_CACHE_MAX_ENTRIES: Final = 1000
_DEFAULT_TOKEN_TTL_SECONDS: Final = 3599.0
_TOKEN_EXPIRY_SLACK_SECONDS: Final = 60.0


def _parse_expires_in(raw: object) -> float:
    if not isinstance(raw, (int, float, str)):
        return _DEFAULT_TOKEN_TTL_SECONDS
    try:
        return float(raw)
    except ValueError:
        return _DEFAULT_TOKEN_TTL_SECONDS


def _parse_tool_input_schema(raw: object) -> Mapping[str, object] | None:
    try:
        return _TOOL_INPUT_SCHEMA_ADAPTER.validate_python(raw)
    except ValidationError:
        return None


class _DefenderResult(TypedDict, total=False):
    status: ReadOnly[str]
    verdict: ReadOnly[str | None]
    message: ReadOnly[str | None]


class _EvaluateResponse(TypedDict, total=False):
    allowed: ReadOnly[bool]
    defender: ReadOnly[_DefenderResult]
    correlationId: ReadOnly[str]


class _ToolReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str | None = None
    input_schema: Mapping[str, object] | None = Field(default=None, serialization_alias="inputSchema")


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
        self.request_timeout = request_timeout
        self.unreachable_fallback: Literal["fail_closed", "fail_open"] = (
            "fail_open" if unreachable_fallback == "fail_open" else "fail_closed"
        )
        self.async_handler = async_handler or get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )
        self._obo_token_cache: OrderedDict[str, tuple[str, float]] = OrderedDict()  # mutable-ok: lock-guarded LRU
        self._obo_cache_lock = threading.Lock()
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
        assertion: Final = entra_assertion(data.get("incoming_bearer_token"))
        if assertion is None:
            self._handle_caller_fault(
                data=data,
                tool_name=tool_name,
                status_code=401,
                reason=(
                    "the caller did not present an Entra bearer token; the Agent 365 guardrail "
                    "authorizes tool calls On-Behalf-Of the signed-in user"
                ),
            )

        try:
            obo_token: Final = await self._get_obo_token(assertion)
        except Agent365TokenExchangeError as exc:
            if exc.error_code in _GATEWAY_OWNED_TOKEN_ERRORS:
                return self._handle_unavailable(
                    data=data,
                    tool_name=tool_name,
                    reason=(
                        f"Entra rejected the gateway's own Agent 365 credentials ({exc.error_code}); "
                        "check the guardrail's client_id, client_secret and resource_app_id"
                    ),
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
                headers={"Authorization": f"Bearer {obo_token}"},  # mutable-ok: httpx header dict
            )
        except (httpx.HTTPError, LitellmTimeout, TimeoutError) as exc:
            return self._handle_unavailable(
                data=data,
                tool_name=tool_name,
                reason=f"the Agent 365 endpoint could not be reached ({type(exc).__name__})",
            )
        latency_ms: Final = (time.perf_counter() - start) * 1000.0
        fallback: Final = self._handle_evaluate_error(
            data=data, tool_name=tool_name, assertion=assertion, response=response, latency_ms=latency_ms
        )
        if fallback is not None:
            return fallback
        return self._enforce_verdict(data=data, tool_name=tool_name, response=response, latency_ms=latency_ms)

    def _handle_evaluate_error(
        self,
        data: dict,  # mutable-ok: guardrail logging appends into the request metadata in place
        tool_name: str,
        assertion: str,
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
                self._evict_obo_token(assertion)
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
        defender_status: Final = defender.get("status")
        if allowed and defender_status != DEFENDER_STATUS_EVALUATED:
            return self._handle_unavailable(
                data=data,
                tool_name=tool_name,
                reason=f"Microsoft Defender did not evaluate the call (defender.status={defender_status or 'missing'})",
                defender_status=defender_status,
                correlation_id=correlation_id,
                latency_ms=latency_ms,
            )
        self._record_verdict(
            data=data,
            verdict="Allow" if allowed else "Block",
            guardrail_status="success" if allowed else "guardrail_intervened",
            defender_status=defender_status,
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
        description: Final = data.get("mcp_tool_description")
        tool_reference: Final = _ToolReference(
            name=tool_name,
            description=description if isinstance(description, str) and description else None,
            input_schema=_parse_tool_input_schema(data.get("mcp_tool_input_schema")),
        )
        payload: Final[dict[str, object]] = {  # mutable-ok: JSON body with optional fields added below
            "tool": tool_reference.model_dump(by_alias=True, exclude_none=True),
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
        raw_logging_obj: Final = data.get("litellm_logging_obj")
        logging_obj: Final = raw_logging_obj if isinstance(raw_logging_obj, LiteLLMLoggingObj) else None
        call_id: Final = data.get("litellm_call_id") or (logging_obj.litellm_call_id if logging_obj else None)
        if isinstance(call_id, str) and call_id:
            return call_id
        if logging_obj is not None:
            tool_call_metadata: Final = logging_obj.model_call_details.get("mcp_tool_call_metadata")
            session_from_logging: Final = (
                tool_call_metadata.get("mcp_session_id") if isinstance(tool_call_metadata, Mapping) else None
            )
            if isinstance(session_from_logging, str) and session_from_logging:
                return session_from_logging
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
        return str(uuid.uuid4())

    async def _get_obo_token(self, assertion: str) -> str:
        cache_key: Final = hashlib.sha256(assertion.encode("utf-8")).hexdigest()
        now: Final = time.time()
        with self._obo_cache_lock:
            cached: Final = self._obo_token_cache.get(cache_key)
            if cached and cached[1] > now + _TOKEN_EXPIRY_SLACK_SECONDS:
                self._obo_token_cache.move_to_end(cache_key)
                return cached[0]

        response: Final = await self._post_allowing_error_status(
            url=TOKEN_ENDPOINT_TEMPLATE.format(tenant_id=self.tenant_id),
            data={  # mutable-ok: OAuth form body; AsyncHTTPHandler.post requires dict
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "assertion": assertion,
                "scope": f"{self.resource_app_id}/{AGENT_365_SCOPE_NAME}",
                "requested_token_use": "on_behalf_of",
            },
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
        access_token: Final = raw_access_token
        expires_at: Final = time.time() + _parse_expires_in(body.get("expires_in", 3599))
        with self._obo_cache_lock:
            self._obo_token_cache[cache_key] = (access_token, expires_at)
            self._obo_token_cache.move_to_end(cache_key)
            while len(self._obo_token_cache) > _OBO_CACHE_MAX_ENTRIES:
                self._obo_token_cache.popitem(last=False)
        return access_token

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

    def _evict_obo_token(self, assertion: str) -> None:
        cache_key: Final = hashlib.sha256(assertion.encode("utf-8")).hexdigest()
        with self._obo_cache_lock:
            self._obo_token_cache.pop(cache_key, None)

    def _handle_unavailable(
        self,
        data: dict,  # mutable-ok: guardrail logging appends into the request metadata in place
        tool_name: str,
        reason: str,
        defender_status: str | None = None,
        correlation_id: str | None = None,
        latency_ms: float | None = None,
    ) -> dict:  # mutable-ok: returns the request data dict per hook contract
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
                defender_status=defender_status,
                correlation_id=correlation_id,
                latency_ms=latency_ms,
                reason=reason,
            )
            return data
        self._record_verdict(
            data=data,
            verdict="Unavailable",
            guardrail_status="guardrail_failed_to_respond",
            defender_status=defender_status,
            correlation_id=correlation_id,
            latency_ms=latency_ms,
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


def _applies_to_caller(guardrail: Agent365Guardrail, user_api_key_auth: "UserAPIKeyAuth") -> bool:
    probe: Final[dict[str, object]] = {"metadata": {}}  # mutable-ok: filled in place by the key resolver
    add_guardrails_from_auth_metadata(
        user_api_key_dict=user_api_key_auth, data=probe, metadata_variable_name="metadata"
    )
    return guardrail.should_run_guardrail(data=probe, event_type=GuardrailEventHooks.pre_mcp_call)


def _applicable_guardrails(
    server: MCPServer, user_api_key_auth: "UserAPIKeyAuth | None"
) -> tuple[Agent365Guardrail, ...]:
    """Agent 365 guardrails that gate ``server`` for this caller: every registered one for the anonymous
    discovery fetch, otherwise those the caller's key, team, or policies select. Empty when the gateway
    does not own sign-in for the server."""
    if server.auth_type == MCPAuth.oauth2 or not server.advertises_gateway_authorization_server:
        return ()
    registered: Final = tuple(
        callback
        for callback in litellm.logging_callback_manager.get_custom_loggers_for_type(Agent365Guardrail)
        if isinstance(callback, Agent365Guardrail)
    )
    if user_api_key_auth is None:
        return registered
    return tuple(g for g in registered if _applies_to_caller(g, user_api_key_auth))


def entra_assertion(value: object) -> str | None:
    """``value`` when it is a compact JWS, the only bearer shape the OBO exchange accepts as its assertion.
    A LiteLLM virtual key, session bearer, or opaque upstream token in ``Authorization`` yields ``None``."""
    return value if isinstance(value, str) and value.count(".") == 2 else None


def agent_365_subject_token_present(oauth2_headers: Mapping[str, str] | None) -> bool:
    """Whether the request's ``Authorization`` carries an Entra assertion the guardrail can exchange."""
    authorization: Final = oauth2_headers.get("Authorization", "") if oauth2_headers else ""
    if not authorization.lower().startswith("bearer "):
        return False
    return entra_assertion(authorization[len("bearer ") :].strip()) is not None


def agent_365_authorization_servers(server: MCPServer, user_api_key_auth: "UserAPIKeyAuth | None") -> tuple[str, ...]:
    """Entra issuers an MCP client signs in with before calling ``server`` through an Agent 365 guardrail."""
    return tuple(
        dict.fromkeys(
            ENTRA_ISSUER_TEMPLATE.format(tenant_id=g.tenant_id)
            for g in _applicable_guardrails(server, user_api_key_auth)
        )
    )


def agent_365_scopes_supported(server: MCPServer, user_api_key_auth: "UserAPIKeyAuth | None") -> tuple[str, ...]:
    """Scopes the client requests from Entra for ``server``: the admin's ``scopes`` when set, otherwise the
    ``access_as_user`` scope of each gating guardrail's gateway app registration (``api://<client_id>``)."""
    if server.scopes:
        return tuple(server.scopes)
    return tuple(
        dict.fromkeys(
            GATEWAY_SCOPE_TEMPLATE.format(client_id=g.client_id)
            for g in _applicable_guardrails(server, user_api_key_auth)
        )
    )
