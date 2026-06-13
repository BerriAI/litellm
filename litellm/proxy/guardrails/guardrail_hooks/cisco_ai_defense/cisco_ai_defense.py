"""
Cisco AI Defense guardrail integration for LiteLLM.

Cisco AI Defense exposes two distinct inspection surfaces, each with its own
endpoint:

* Chat inspection:  POST <base>/api/v1/inspect/chat   — LLM conversations
* MCP inspection:   POST <base>/api/v1/inspect/mcp    — MCP tool calls

Each guardrail instance targets exactly one surface, chosen via the
``inspection_type`` dropdown:

* ``chat`` — scan LLM model traffic only
* ``mcp``  — scan MCP tool-call traffic only

Configure two separate guardrails if you need both surfaces scanned. Each
request is sent with the ``X-Cisco-AI-Defense-API-Key`` header.
"""

import json
import os
from dataclasses import dataclass, replace
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    Union,
)

import httpx
from fastapi import HTTPException

from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm._version import version as litellm_version
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.callback_utils import (
    add_guardrail_to_applied_guardrails_header,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import (
    Choices,
    LLMResponseTypes,
    ModelResponse,
    ModelResponseStream,
    TextCompletionResponse,
)

from .cisco_ai_defense_mcp import _CiscoAIDefenseMcpMixin

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import (
        GuardrailConfigModel,
    )


CISCO_DEFAULT_API_BASE = "https://us.api.inspect.aidefense.security.cisco.com"
CISCO_CHAT_INSPECT_PATH = "/api/v1/inspect/chat"
CISCO_MCP_INSPECT_PATH = "/api/v1/inspect/mcp"
CISCO_API_KEY_HEADER = "X-Cisco-AI-Defense-API-Key"
DEFAULT_TIMEOUT_SECONDS = 10.0

SUPPORTED_INSPECTION_TYPES: Tuple[str, ...] = ("chat", "mcp")
DEFAULT_INSPECTION_TYPE = "chat"

# LiteLLM marks MCP guardrail calls with these call_type values; the proxy
# routes pre_mcp_call / during_mcp_call events through async_pre_call_hook /
# async_moderation_hook with the call_type set accordingly.
_MCP_CALL_TYPES: Tuple[str, ...] = ("mcp_call", "call_mcp_tool")

# Action vocabulary Cisco AI Defense can return.
_ACTION_BLOCK = "block"
_ACTION_REDACT = "redact"
_ACTION_ALLOW = "allow"


@dataclass(frozen=True, slots=True)
class _ScanContext:
    """The surface (``chat`` / ``mcp``) and direction (``input`` / ``output``) a scan targets."""

    surface: str
    direction: str


@dataclass(frozen=True, slots=True)
class _CiscoVerdict:
    """Parsed Cisco AI Defense decision plus any sanitized rewrites it carries."""

    is_safe: Optional[bool]
    classifications: List[str]
    severity: Optional[str]
    rules: List[Dict[str, Any]]
    explanation: Optional[str]
    event_id: Optional[str]
    action: Optional[str] = None
    sanitized_text: Optional[str] = None
    sanitized_messages: Optional[List[Dict[str, Any]]] = None
    sanitized_mcp_arguments: Optional[Dict[str, Any]] = None


class CiscoAIDefenseGuardrailMissingSecrets(Exception):
    """Raised when the Cisco AI Defense API key is missing."""


class CiscoAIDefenseGuardrailAPIError(Exception):
    """Raised when there is an error talking to the Cisco AI Defense API."""


class CiscoAIDefenseGuardrail(_CiscoAIDefenseMcpMixin, CustomGuardrail):
    """
    Cisco AI Defense guardrail integration.

    Each instance scans exactly one inspection surface (``chat`` or ``mcp``)
    via the corresponding Cisco AI Defense Inspection API endpoint.

    MCP-specific hooks and helpers live on ``_CiscoAIDefenseMcpMixin`` in
    ``cisco_ai_defense_mcp.py``.
    """

    SUPPORTED_ON_FLAGGED_ACTIONS: Tuple[str, ...] = ("block", "monitor")
    DEFAULT_ON_FLAGGED_ACTION: str = "block"
    SUPPORTED_FALLBACK_ACTIONS: Tuple[str, ...] = ("allow", "block")
    DEFAULT_FALLBACK_ON_ERROR: str = "block"

    _PROVIDER_NAME = "cisco_ai_defense"

    def __init__(
        self,
        guardrail_name: Optional[str] = "cisco-ai-defense",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        inspection_type: Optional[str] = None,
        inspect_path: Optional[str] = None,
        enabled_rules: Optional[List[Dict[str, Any]]] = None,
        integration_profile_id: Optional[str] = None,
        integration_profile_version: Optional[str] = None,
        integration_tenant_id: Optional[str] = None,
        integration_type: Optional[str] = None,
        on_flagged_action: Optional[str] = None,
        fallback_on_error: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        resolved_api_key = api_key or os.environ.get("CISCO_AI_DEFENSE_API_KEY")
        if not resolved_api_key:
            raise CiscoAIDefenseGuardrailMissingSecrets(
                "Cisco AI Defense API key is required. Set "
                "`CISCO_AI_DEFENSE_API_KEY` in the environment or pass "
                "`api_key` in the guardrail config."
            )
        self.api_key: str = resolved_api_key

        self.api_base: str = (
            api_base
            or os.environ.get("CISCO_AI_DEFENSE_API_BASE")
            or CISCO_DEFAULT_API_BASE
        ).rstrip("/")

        self.inspection_type: str = self._resolve_choice(
            value=inspection_type,
            env_var="CISCO_AI_DEFENSE_INSPECTION_TYPE",
            allowed=SUPPORTED_INSPECTION_TYPES,
            default=DEFAULT_INSPECTION_TYPE,
            setting_name="inspection_type",
        )

        inferred = self._infer_inspection_type_from_mode(
            kwargs.get("event_hook"), self.inspection_type
        )
        if inferred != self.inspection_type:
            verbose_proxy_logger.info(
                "Cisco AI Defense: inferred inspection_type=%s from "
                "MCP-only event_hook configuration (was %s)",
                inferred,
                self.inspection_type,
            )
            self.inspection_type = inferred

        if inspect_path:
            self.inspect_path = (
                inspect_path if inspect_path.startswith("/") else f"/{inspect_path}"
            )
        else:
            self.inspect_path = (
                CISCO_MCP_INSPECT_PATH
                if self.inspection_type == "mcp"
                else CISCO_CHAT_INSPECT_PATH
            )

        self.enabled_rules = (
            [self._normalize_rule(rule) for rule in enabled_rules]
            if enabled_rules
            else None
        )
        self.integration_profile_id = integration_profile_id
        self.integration_profile_version = integration_profile_version
        self.integration_tenant_id = integration_tenant_id
        self.integration_type = integration_type

        self.on_flagged_action = self._resolve_choice(
            value=on_flagged_action,
            env_var="CISCO_AI_DEFENSE_ON_FLAGGED_ACTION",
            allowed=self.SUPPORTED_ON_FLAGGED_ACTIONS,
            default=self.DEFAULT_ON_FLAGGED_ACTION,
            setting_name="on_flagged_action",
        )

        self.fallback_on_error = self._resolve_choice(
            value=fallback_on_error,
            env_var="CISCO_AI_DEFENSE_FALLBACK_ON_ERROR",
            allowed=self.SUPPORTED_FALLBACK_ACTIONS,
            default=self.DEFAULT_FALLBACK_ON_ERROR,
            setting_name="fallback_on_error",
        )

        resolved_timeout: Optional[float]
        if timeout is not None:
            resolved_timeout = self._coerce_timeout(timeout)
        else:
            env_timeout = os.environ.get("CISCO_AI_DEFENSE_TIMEOUT")
            resolved_timeout = (
                self._coerce_timeout(env_timeout) if env_timeout is not None else None
            )
        self.timeout: float = (
            resolved_timeout
            if resolved_timeout is not None
            else DEFAULT_TIMEOUT_SECONDS
        )

        self.async_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.GuardrailCallback
        )

        # Register broadly; runtime filtering happens in ``_surface_matches``.
        supported_event_hooks = [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.during_call,
            GuardrailEventHooks.post_call,
            GuardrailEventHooks.logging_only,
            GuardrailEventHooks.pre_mcp_call,
            GuardrailEventHooks.during_mcp_call,
        ]

        super().__init__(
            guardrail_name=guardrail_name,
            supported_event_hooks=supported_event_hooks,
            **kwargs,
        )

        self._warn_if_mode_surface_mismatch(kwargs.get("event_hook"))

        verbose_proxy_logger.debug(
            "Cisco AI Defense guardrail initialized: name=%s, "
            "inspection_type=%s, url=%s%s, on_flagged_action=%s, "
            "fallback_on_error=%s, timeout=%ss",
            guardrail_name,
            self.inspection_type,
            self.api_base,
            self.inspect_path,
            self.on_flagged_action,
            self.fallback_on_error,
            self.timeout,
        )

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_choice(
        value: Optional[str],
        env_var: str,
        allowed: Tuple[str, ...],
        default: str,
        setting_name: str,
    ) -> str:
        candidate = value if value is not None else os.environ.get(env_var)
        if candidate is None:
            return default
        if candidate in allowed:
            return candidate
        verbose_proxy_logger.warning(
            "Cisco AI Defense guardrail: invalid value '%s' for %s, falling "
            "back to default '%s'. Allowed values: %s",
            candidate,
            setting_name,
            default,
            ", ".join(allowed),
        )
        return default

    @staticmethod
    def _coerce_timeout(value: Union[str, float]) -> Optional[float]:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            verbose_proxy_logger.warning(
                "Cisco AI Defense guardrail: invalid timeout value '%s', "
                "using default %ss",
                value,
                DEFAULT_TIMEOUT_SECONDS,
            )
            return None
        if parsed < 1.0:
            return 1.0
        if parsed > 60.0:
            return 60.0
        return parsed

    @staticmethod
    def _is_mcp_call_type(call_type: Optional[str]) -> bool:
        return bool(call_type) and call_type in _MCP_CALL_TYPES

    # ------------------------------------------------------------------
    # Hook methods
    # ------------------------------------------------------------------

    @log_guardrail_information
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
            "mcp_call",
            "anthropic_messages",
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        # Trust proxy call_type, not caller-controlled request shape.
        is_mcp = self._is_mcp_call_type(call_type)

        if not self._surface_matches(is_mcp):
            verbose_proxy_logger.debug(
                "Cisco AI Defense guardrail: call_type=%s does not match "
                "configured inspection_type=%s, skipping",
                call_type,
                self.inspection_type,
            )
            return data

        event_type = (
            GuardrailEventHooks.pre_mcp_call if is_mcp else GuardrailEventHooks.pre_call
        )
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        if is_mcp:
            await self._inspect_mcp_request(
                data=data, user_api_key_dict=user_api_key_dict
            )
        else:
            messages = self._extract_inspect_messages_from_request(data)
            if not messages:
                verbose_proxy_logger.debug(
                    "Cisco AI Defense guardrail: no scannable messages in "
                    "pre-call request, skipping"
                )
                return data
            await self._inspect_chat(
                messages=messages,
                request_data=data,
                user_api_key_dict=user_api_key_dict,
            )

        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )
        return data

    @log_guardrail_information
    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal[
            "completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "responses",
            "mcp_call",
            "anthropic_messages",
        ],
    ) -> Optional[Union[Exception, str, dict]]:
        is_mcp = self._is_mcp_call_type(call_type)

        if not self._surface_matches(is_mcp):
            return data

        event_type = (
            GuardrailEventHooks.during_mcp_call
            if is_mcp
            else GuardrailEventHooks.during_call
        )
        if self.should_run_guardrail(data=data, event_type=event_type) is not True:
            return data

        if is_mcp:
            await self._inspect_mcp_request(
                data=data, user_api_key_dict=user_api_key_dict
            )
        else:
            messages = self._extract_inspect_messages_from_request(data)
            if not messages:
                return data
            await self._inspect_chat(
                messages=messages,
                request_data=data,
                user_api_key_dict=user_api_key_dict,
            )

        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )
        return data

    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: LLMResponseTypes,
    ) -> LLMResponseTypes:
        if self.inspection_type != "chat":
            return response

        if (
            self.should_run_guardrail(
                data=data, event_type=GuardrailEventHooks.post_call
            )
            is not True
        ):
            return response

        response_messages = self._extract_response_messages(response)
        if not response_messages:
            verbose_proxy_logger.debug(
                "Cisco AI Defense guardrail: no response content to scan, "
                "skipping post-call analysis"
            )
            return response

        request_messages = self._extract_inspect_messages_from_request(data)
        conversation = request_messages + response_messages

        await self._inspect_chat(
            messages=conversation,
            request_data=data,
            user_api_key_dict=user_api_key_dict,
            direction="output",
            response_obj=response,
        )

        add_guardrail_to_applied_guardrails_header(
            request_data=data, guardrail_name=self.guardrail_name
        )
        return response

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncIterator[Any],
        request_data: dict,
    ):
        """Buffer and inspect streaming chat output before delivery."""
        from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
        from litellm.main import stream_chunk_builder

        if self.inspection_type != "chat":
            async for chunk in response:
                yield chunk
            return

        if (
            self.should_run_guardrail(
                data=request_data, event_type=GuardrailEventHooks.post_call
            )
            is not True
        ):
            async for chunk in response:
                yield chunk
            return

        verbose_proxy_logger.debug(
            "Cisco AI Defense guardrail (%s): scanning streaming chat response.",
            self.guardrail_name,
        )

        all_chunks: List[Any] = []
        try:
            async for chunk in response:
                all_chunks.append(chunk)
        except Exception as exc:
            verbose_proxy_logger.error(
                "Cisco AI Defense guardrail: upstream streaming failed: %s",
                exc,
            )
            raise

        if not all_chunks:
            return

        if not isinstance(all_chunks[0], (ModelResponse, ModelResponseStream)):
            verbose_proxy_logger.warning(
                "Cisco AI Defense guardrail (%s): unsupported streaming "
                "chunk shape (%s) — failing closed.",
                self.guardrail_name,
                type(all_chunks[0]).__name__,
            )
            yield f'data: {json.dumps({"error": {"message": "Cisco AI Defense: unsupported streaming format — response withheld for safety", "type": "guardrail_unsupported_stream", "code": 400, "guardrail": self.guardrail_name}})}\n\n'
            return

        assembled = stream_chunk_builder(chunks=all_chunks)
        if assembled is None:
            for chunk in all_chunks:
                yield chunk
            return
        if not isinstance(assembled, ModelResponse):
            verbose_proxy_logger.warning(
                "Cisco AI Defense guardrail (%s): assembled streaming "
                "response has unsupported shape (%s) — failing closed.",
                self.guardrail_name,
                type(assembled).__name__,
            )
            yield f'data: {json.dumps({"error": {"message": "Cisco AI Defense: unsupported streaming format — response withheld for safety", "type": "guardrail_unsupported_stream", "code": 400, "guardrail": self.guardrail_name}})}\n\n'
            return

        response_messages = self._extract_response_messages(assembled)
        original_stream_text = self._extract_streaming_chunk_scan_text(all_chunks)
        assembled_text = " ".join(
            m.get("content", "") for m in response_messages if isinstance(m, dict)
        )
        if original_stream_text and original_stream_text not in assembled_text:
            response_messages.append(
                {"role": "assistant", "content": original_stream_text}
            )
        if not response_messages:
            for chunk in all_chunks:
                yield chunk
            return

        request_messages = self._extract_inspect_messages_from_request(request_data)
        conversation = request_messages + response_messages

        try:
            await self._inspect_chat(
                messages=conversation,
                request_data=request_data,
                user_api_key_dict=user_api_key_dict,
                direction="output",
                response_obj=assembled,
            )
        except HTTPException as exc:
            error_obj: Dict[str, Any] = self._http_exception_to_error_obj(exc)
            verbose_proxy_logger.warning(
                "Cisco AI Defense guardrail (%s): streaming response "
                "blocked — emitting SSE error event instead of "
                "delivering buffered chunks.",
                self.guardrail_name,
            )
            yield f"data: {json.dumps({'error': error_obj})}\n\n"
            return
        except Exception as exc:
            verbose_proxy_logger.error(
                "Cisco AI Defense guardrail (%s): streaming response "
                "scan failed: %s",
                self.guardrail_name,
                exc,
            )
            error_obj = {
                "message": (
                    "Cisco AI Defense streaming scan failed — response " "withheld."
                ),
                "type": "guardrail_scan_error",
                "code": 500,
                "guardrail": self.guardrail_name,
            }
            yield f"data: {json.dumps({'error': error_obj})}\n\n"
            return

        add_guardrail_to_applied_guardrails_header(
            request_data=request_data, guardrail_name=self.guardrail_name
        )

        if self._streaming_content_was_modified(all_chunks, assembled):
            mock_iterator = MockResponseIterator(model_response=assembled)
            async for chunk in mock_iterator:
                yield chunk
        else:
            for chunk in all_chunks:
                yield chunk

    def _build_block_payload(
        self, context: _ScanContext, verdict: _CiscoVerdict
    ) -> Dict[str, Any]:
        """Canonical block payload used across all four block paths.

        Same dict is the ``HTTPException.detail`` for chat / MCP request
        and chat response blocks, the ``error`` value in the streaming
        SSE event, and (JSON-encoded) the text content of the synthetic
        MCP response object. Keeps the customer-facing format identical
        regardless of which transport carries the block.
        """
        return {
            "error": "Blocked by Cisco AI Defense Guardrail",
            "message": "Blocked by Cisco AI Defense Guardrail",
            "provider": self._PROVIDER_NAME,
            "guardrail": self.guardrail_name,
            "surface": context.surface,
            "direction": context.direction,
            "action": "block",
            "classifications": list(verdict.classifications),
            "severity": verdict.severity,
            "rules": [r.get("rule_name") for r in verdict.rules if isinstance(r, dict)],
            "explanation": verdict.explanation,
            "event_id": verdict.event_id,
        }

    def _http_exception_to_error_obj(self, exc: HTTPException) -> Dict[str, Any]:
        """Wrap an ``HTTPException`` detail into the SSE ``error`` payload.

        For Cisco's own blocks the detail is already the canonical block
        payload, so this is a near-passthrough that just adds ``code``
        / ``guardrail`` defaults for non-Cisco / unstructured details.
        """
        error_obj: Dict[str, Any] = (
            dict(exc.detail)
            if isinstance(exc.detail, dict)
            else {"message": str(exc.detail)}
        )
        error_obj.setdefault("message", error_obj.get("error", "Guardrail block"))
        error_obj.setdefault("code", exc.status_code)
        error_obj.setdefault("guardrail", self.guardrail_name)
        return error_obj

    @classmethod
    def _streaming_content_was_modified(
        cls, original_chunks: List[Any], assembled: ModelResponse
    ) -> bool:
        """Decide whether redact changed content or tool/function arguments."""
        original_text = cls._extract_streaming_chunk_scan_text(original_chunks)
        assembled_text = " ".join(
            m.get("content", "") for m in cls._extract_response_messages(assembled)
        )
        return original_text != assembled_text

    @classmethod
    def _extract_streaming_chunk_scan_text(cls, chunks: List[Any]) -> str:
        original_text = ""
        argument_text = ""
        for chunk in chunks:
            choices = getattr(chunk, "choices", None) or []
            for c in choices:
                delta = getattr(c, "delta", None)
                if delta is None:
                    continue
                text = getattr(delta, "content", None)
                if isinstance(text, str):
                    original_text += text
                reasoning_text = " ".join(cls._extract_message_reasoning_parts(delta))
                if reasoning_text:
                    original_text += reasoning_text
                for tc in getattr(delta, "tool_calls", None) or []:
                    args = cls._extract_tool_call_arguments(tc)
                    if args:
                        argument_text += args
                fc = getattr(delta, "function_call", None)
                if fc is not None:
                    args = cls._extract_function_call_arguments(fc)
                    if args:
                        argument_text += args
        return " ".join(part for part in (original_text, argument_text) if part)

    # ------------------------------------------------------------------
    # MCP post-tool-call hook lives on ``_CiscoAIDefenseMcpMixin`` in
    # ``cisco_ai_defense_mcp.py``. The mixin's methods are inherited via
    # the class declaration above (multiple-inheritance with
    # ``_CiscoAIDefenseMcpMixin`` placed first).
    # ------------------------------------------------------------------

    def _surface_matches(self, is_mcp_traffic: bool) -> bool:
        """Return True when the traffic surface matches the configured type."""
        if self.inspection_type == "mcp":
            return is_mcp_traffic
        return not is_mcp_traffic

    @staticmethod
    def _normalize_event_hooks(event_hook: object) -> set:
        """Coerce a ``mode`` arg (str, enum, or list of either) to a set of values."""

        def _norm(hook: object) -> Optional[str]:
            value = getattr(hook, "value", None)
            if isinstance(value, str):
                return value
            if isinstance(hook, str):
                return hook
            return None

        if event_hook is None:
            return set()
        if isinstance(event_hook, list):
            values = {_norm(h) for h in event_hook}
        else:
            values = {_norm(event_hook)}
        values.discard(None)
        return values

    @staticmethod
    def _infer_inspection_type_from_mode(event_hook: object, current: str) -> str:
        """Return ``mcp`` when ``event_hook`` is exclusively MCP-typed.

        ``pre_mcp_call`` and ``during_mcp_call`` only fire for MCP traffic,
        so a user who picks them clearly wants MCP inspection — auto-flip
        the surface so they don't also have to toggle ``inspection_type``.
        """
        configured = CiscoAIDefenseGuardrail._normalize_event_hooks(event_hook)
        if not configured:
            return current
        mcp_hooks = {"pre_mcp_call", "during_mcp_call"}
        chat_hooks = {"pre_call", "during_call", "post_call"}
        has_mcp = bool(configured & mcp_hooks)
        has_chat = bool(configured & chat_hooks)
        # Exclusively MCP → mcp; exclusively chat → chat; mixed → keep
        # current so the user retains control over the dual-surface case.
        if has_mcp and not has_chat:
            return "mcp"
        if has_chat and not has_mcp:
            return "chat"
        return current

    def _log_decision(
        self,
        context: _ScanContext,
        verdict: _CiscoVerdict,
        duration_ms: float,
        request_data: dict,
    ) -> None:
        """Emit a single visible log line per scan.

        Mirrors the reference plugin's ``AI_DEFENSE_DECISION`` line so
        operators can observe scans without bumping log levels. INFO for
        allow, WARNING for intervened/redacted, ERROR is left for
        upstream API failures.
        """
        fields: Dict[str, Any] = {
            "guardrail": self.guardrail_name,
            "surface": context.surface,
            "direction": context.direction,
            "action": verdict.action,
            "is_safe": verdict.is_safe,
            "severity": verdict.severity,
            "classifications": (
                list(verdict.classifications) if verdict.classifications else []
            ),
            "rule_violations": sorted(
                {
                    rule.get("rule_name")
                    for rule in verdict.rules
                    if isinstance(rule, dict)
                    and rule.get("rule_name")
                    and rule.get("classification") not in (None, "NONE_VIOLATION")
                }
            ),
            "event_id": verdict.event_id,
            "duration_ms": round(duration_ms, 1),
        }
        # Best-effort request context — useful when correlating with model
        # / MCP-tool calls. None values are dropped for log-line brevity.
        for source_key, target_key in (
            ("model", "model"),
            ("litellm_call_id", "call_id"),
            ("mcp_tool_name", "mcp_tool"),
            ("mcp_server_name", "mcp_server"),
        ):
            value = request_data.get(source_key)
            if value:
                fields[target_key] = value

        payload = {k: v for k, v in fields.items() if v not in (None, [], "")}
        line = "CISCO_AI_DEFENSE_DECISION " + json.dumps(
            payload, default=str, sort_keys=True, separators=(",", ":")
        )

        if verdict.action == _ACTION_ALLOW:
            verbose_proxy_logger.info(line)
        else:
            verbose_proxy_logger.warning(line)

    def _warn_if_mode_surface_mismatch(self, event_hook: object) -> None:
        """Log a warning only when ``mode`` mixes both surfaces.

        Auto-inference in ``_infer_inspection_type_from_mode`` handles the
        "exclusively MCP" and "exclusively chat" cases, so this warning
        fires only for genuinely mixed configurations where we can't tell
        which surface the user wants and have to honour their explicit
        ``inspection_type``.
        """
        configured = self._normalize_event_hooks(event_hook)
        mcp_hooks = configured & {"pre_mcp_call", "during_mcp_call"}
        chat_hooks = configured & {"pre_call", "during_call", "post_call"}
        if not (mcp_hooks and chat_hooks):
            return

        unused_hooks = mcp_hooks if self.inspection_type == "chat" else chat_hooks
        verbose_proxy_logger.warning(
            "Cisco AI Defense guardrail '%s' (inspection_type=%s) has mixed "
            "mode %s — the %s event hooks won't fire because this guardrail "
            "only inspects %s traffic. Configure two guardrails (one per "
            "surface) for full coverage, or drop the cross-surface modes.",
            self.guardrail_name,
            self.inspection_type,
            sorted(configured),
            sorted(unused_hooks),
            self.inspection_type,
        )

    # ------------------------------------------------------------------
    # Chat inspection
    # ------------------------------------------------------------------

    async def _inspect_chat(
        self,
        messages: List[Dict[str, str]],
        request_data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        direction: str = "input",
        response_obj: object = None,
    ) -> Dict[str, Any]:
        url = f"{self.api_base}{self.inspect_path}"
        payload = self._build_chat_payload(messages, request_data, user_api_key_dict)
        start_time = datetime.now()
        try:
            inspect_response = await self._post_inspection(
                url=url, payload=payload, surface="chat"
            )
        except HTTPException:
            # Re-raise; _post_inspection only raises CiscoAIDefenseGuardrailAPIError,
            # but be defensive in case downstream evolves.
            raise
        except Exception as exc:
            return self._handle_api_error(
                exc,
                request_data=request_data,
                start_time=start_time,
                surface="chat",
                direction=direction,
            )

        return self._finalize_inspection(
            inspect_response=inspect_response,
            request_data=request_data,
            context=_ScanContext(surface="chat", direction=direction),
            start_time=start_time,
            response_obj=response_obj,
        )

    def _build_chat_payload(
        self,
        messages: List[Dict[str, str]],
        request_data: dict,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> Dict[str, Any]:
        return {
            "messages": messages,
            "metadata": self._build_metadata(request_data, user_api_key_dict),
            "config": self._build_config(),
        }

    # ------------------------------------------------------------------
    # Shared HTTP / metadata helpers
    # ------------------------------------------------------------------

    async def _post_inspection(
        self,
        url: str,
        payload: Dict[str, Any],
        surface: str,
    ) -> Dict[str, Any]:
        headers = self._build_headers()
        verbose_proxy_logger.debug(
            "Cisco AI Defense guardrail: posting %s inspection to %s",
            surface,
            url,
        )
        try:
            request = self.async_handler.client.build_request(
                "POST",
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response = await self.async_handler.client.send(
                request,
                follow_redirects=False,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code if exc.response is not None else 0
            body_snippet = ""
            try:
                body_snippet = exc.response.text[:500] if exc.response else ""
            except Exception:
                body_snippet = ""
            raise CiscoAIDefenseGuardrailAPIError(
                f"Cisco AI Defense {surface} API returned HTTP {status_code}: "
                f"{body_snippet}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise CiscoAIDefenseGuardrailAPIError(
                f"Cisco AI Defense {surface} API call timed out after "
                f"{self.timeout}s"
            ) from exc
        except httpx.RequestError as exc:
            raise CiscoAIDefenseGuardrailAPIError(
                f"Cisco AI Defense {surface} API request failed: {exc}"
            ) from exc

        try:
            return response.json()
        except ValueError as exc:
            raise CiscoAIDefenseGuardrailAPIError(
                f"Cisco AI Defense {surface} API returned a non-JSON response"
            ) from exc

    def _build_headers(self) -> Dict[str, str]:
        return {
            CISCO_API_KEY_HEADER: self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"litellm/{litellm_version}",
        }

    def _build_metadata(
        self,
        request_data: dict,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {}

        user = request_data.get("user") or getattr(user_api_key_dict, "user_id", None)
        if user:
            metadata["user"] = str(user)

        litellm_call_id = request_data.get("litellm_call_id")
        if litellm_call_id:
            metadata["client_transaction_id"] = str(litellm_call_id)

        request_metadata = request_data.get("metadata") or {}
        if isinstance(request_metadata, dict):
            for src_key in (
                "src_app",
                "dst_app",
                "src_ip",
                "dst_ip",
                "dst_host",
                "sni",
                "user_agent",
            ):
                value = request_metadata.get(src_key)
                if value:
                    metadata[src_key] = str(value)

        return metadata

    def _build_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        if self.enabled_rules:
            config["enabled_rules"] = self.enabled_rules
        if self.integration_profile_id:
            config["integration_profile_id"] = self.integration_profile_id
        if self.integration_profile_version:
            config["integration_profile_version"] = self.integration_profile_version
        if self.integration_tenant_id:
            config["integration_tenant_id"] = self.integration_tenant_id
        if self.integration_type:
            config["integration_type"] = self.integration_type
        return config

    @staticmethod
    def _normalize_rule(rule: object) -> Dict[str, Any]:
        """Coerce a user-supplied rule into the wire-shape dict Cisco expects.

        Accepts ``str``, ``dict``, and Pydantic model inputs.
        """
        if isinstance(rule, str):
            return {"rule_name": rule}

        if not isinstance(rule, dict):
            # Pydantic BaseModel (CiscoAIDefenseRule and friends): dump
            # to a dict and re-enter the dict branch. Anything else
            # falls through to the explicit raise so misconfig still
            # surfaces clearly at startup instead of mid-request.
            model_dump = getattr(rule, "model_dump", None)
            if callable(model_dump):
                try:
                    dumped = model_dump(exclude_none=True)
                except TypeError:
                    dumped = model_dump()
                if isinstance(dumped, dict):
                    rule = dumped

        if isinstance(rule, dict):
            normalized: Dict[str, Any] = {}
            rule_name = rule.get("rule_name")
            if rule_name:
                normalized["rule_name"] = rule_name
            entity_types = rule.get("entity_types")
            if entity_types:
                normalized["entity_types"] = list(entity_types)
            rule_id = rule.get("rule_id")
            if rule_id is not None:
                normalized["rule_id"] = rule_id
            classification = rule.get("classification")
            if classification:
                normalized["classification"] = classification
            return normalized

        raise ValueError(
            f"Cisco AI Defense guardrail: invalid rule definition: {rule!r}"
        )

    # ------------------------------------------------------------------
    # Response processing
    # ------------------------------------------------------------------

    def _finalize_inspection(
        self,
        inspect_response: Dict[str, Any],
        request_data: dict,
        context: _ScanContext,
        start_time: datetime,
        response_obj: object = None,
    ) -> Dict[str, Any]:
        """Parse, log, and (optionally) raise/redact on the Cisco verdict.

        ``context.direction`` is ``"input"`` for request scans and ``"output"``
        for response scans (used for metadata namespacing and response headers).
        ``response_obj`` is the LiteLLM response object (or MCP tool-call
        response) used when applying a ``redact`` action to outputs.

        Cisco AI Defense returns two different envelope shapes depending on
        the endpoint:

        * ``/api/v1/inspect/chat`` — top-level verdict
          ``{"is_safe": ..., "classifications": [...], "action": ..., ...}``
        * ``/api/v1/inspect/mcp`` — JSON-RPC wrapper
          ``{"jsonrpc": "2.0", "id": ..., "result": {<same verdict>}}``

        We unwrap the JSON-RPC ``result`` so both endpoints feed the same
        downstream code path. The error envelope detection below already
        handles ``error`` at either level.
        """
        # Surface JSON-RPC error envelopes (HTTP 200 + Cisco-side error) the
        # same way as transport errors: fail-open or fail-closed.
        jsonrpc_error = self._extract_jsonrpc_error(inspect_response)
        if jsonrpc_error is not None:
            verbose_proxy_logger.warning(
                "Cisco AI Defense guardrail: API returned JSON-RPC error "
                "envelope (code=%s message=%s)",
                jsonrpc_error.get("code"),
                jsonrpc_error.get("message"),
            )
            return self._handle_api_error(
                CiscoAIDefenseGuardrailAPIError(
                    f"AI Defense error code={jsonrpc_error.get('code')} "
                    f"message={jsonrpc_error.get('message')}"
                ),
                request_data=request_data,
                start_time=start_time,
                surface=context.surface,
                direction=context.direction,
            )

        # Unwrap the JSON-RPC ``result`` envelope used by the MCP inspect
        # endpoint. The chat endpoint returns the verdict at the top
        # level and isn't wrapped, so this is a no-op there.
        verdict_dict = self._unwrap_verdict_envelope(inspect_response)

        # OpenAPI spec lists `classification` as required (singular) but
        # examples & SDK return `classifications` (plural). Accept both.
        classifications = (
            verdict_dict.get("classifications")
            or (
                [verdict_dict["classification"]]
                if verdict_dict.get("classification")
                else []
            )
            or []
        )
        verdict = _CiscoVerdict(
            is_safe=verdict_dict.get("is_safe"),
            classifications=classifications,
            severity=verdict_dict.get("severity"),
            rules=verdict_dict.get("rules") or [],
            explanation=verdict_dict.get("explanation"),
            event_id=verdict_dict.get("event_id"),
            sanitized_text=self._extract_sanitized_text(verdict_dict),
            sanitized_messages=self._extract_sanitized_messages(verdict_dict),
            sanitized_mcp_arguments=self._extract_sanitized_mcp_arguments(verdict_dict),
        )

        action_raw = verdict_dict.get("action")
        if isinstance(action_raw, str) and action_raw.strip():
            action = self._normalize_action(action_raw)
        else:
            action = _ACTION_ALLOW
        verdict = replace(verdict, action=action)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        if context.surface == "mcp":
            logging_event_type = (
                GuardrailEventHooks.during_mcp_call
                if context.direction == "output"
                else GuardrailEventHooks.pre_mcp_call
            )
        else:
            logging_event_type = (
                GuardrailEventHooks.post_call
                if context.direction == "output"
                else GuardrailEventHooks.pre_call
            )

        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_provider=self._PROVIDER_NAME,
            guardrail_json_response=self._sanitize_response_for_logging(
                inspect_response, surface=context.surface, action=action
            ),
            request_data=request_data,
            guardrail_status=(
                "guardrail_intervened"
                if action in (_ACTION_BLOCK, _ACTION_REDACT)
                else "success"
            ),
            start_time=start_time.timestamp(),
            end_time=end_time.timestamp(),
            duration=duration,
            masked_entity_count=self._extract_masked_entity_count(verdict.rules),
            event_type=logging_event_type,
        )

        self._stash_verdict_on_request(request_data, context, verdict)

        self._log_decision(context, verdict, duration * 1000, request_data)

        if action == _ACTION_ALLOW:
            return inspect_response

        if action == _ACTION_REDACT:
            redacted = self._apply_redaction(
                request_data, response_obj, context, verdict
            )
            if redacted:
                verbose_proxy_logger.info(
                    "Cisco AI Defense guardrail (%s): redaction applied "
                    "(event_id=%s)",
                    context.surface,
                    verdict.event_id,
                )
                return inspect_response
            verbose_proxy_logger.warning(
                "Cisco AI Defense guardrail (%s): redact requested but no "
                "rewritable surface found — falling through to "
                "on_flagged_action=%s",
                context.surface,
                self.on_flagged_action,
            )

        if self.on_flagged_action == "block":
            raise HTTPException(
                status_code=400,
                detail=self._build_block_payload(context, verdict),
            )

        verbose_proxy_logger.info(
            "Cisco AI Defense guardrail (%s): violation in monitor mode — "
            "request allowed to proceed (event_id=%s)",
            context.surface,
            verdict.event_id,
        )
        return inspect_response

    @staticmethod
    def _stash_verdict_on_request(
        request_data: dict, context: _ScanContext, verdict: _CiscoVerdict
    ) -> None:
        """Surface the Cisco verdict on the request metadata for observability."""
        metadata_store = request_data.setdefault("metadata", {})
        if not isinstance(metadata_store, dict):
            return
        prefix = f"cisco_ai_defense_{context.surface}_{context.direction}"
        metadata_store[f"{prefix}_is_safe"] = verdict.is_safe
        if verdict.action:
            metadata_store[f"{prefix}_action"] = verdict.action
        if verdict.classifications:
            metadata_store[f"{prefix}_classifications"] = list(verdict.classifications)
        if verdict.severity:
            metadata_store[f"{prefix}_severity"] = verdict.severity
        if verdict.rules:
            metadata_store[f"{prefix}_rules"] = [
                rule.get("rule_name")
                for rule in verdict.rules
                if isinstance(rule, dict)
            ]
        if verdict.event_id:
            metadata_store[f"{prefix}_event_id"] = verdict.event_id

    _REDACTED_LOG_KEYS = frozenset(
        {
            "raw_request",
            "sanitized_payload",
            "sanitizedPayload",
            "modified_payload",
            "modifiedPayload",
        }
    )

    @classmethod
    def _sanitize_response_for_logging(
        cls,
        inspect_response: Dict[str, Any],
        surface: str,
        action: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Drop bulky / privacy-sensitive fields, recursing into nested dicts.

        MCP verdicts are commonly nested under ``result``, so a
        top-level-only strip would leave ``result.raw_request`` or
        ``result.sanitized_payload`` in the logging metadata.
        """
        if not isinstance(inspect_response, dict):
            return {"surface": surface, **({"action": action} if action else {})}
        sanitized = cls._strip_sensitive_keys(inspect_response)
        sanitized["surface"] = surface
        if action:
            sanitized["action"] = action
        return sanitized

    @classmethod
    def _strip_sensitive_keys(cls, d: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively strip privacy-sensitive keys from a verdict dict."""
        out: Dict[str, Any] = {}
        for key, value in d.items():
            if key.startswith("_") or key in cls._REDACTED_LOG_KEYS:
                continue
            if isinstance(value, dict):
                out[key] = cls._strip_sensitive_keys(value)
            else:
                out[key] = value
        return out

    # ------------------------------------------------------------------
    # Verdict extraction helpers (sanitized content + JSON-RPC errors)
    # ------------------------------------------------------------------

    _DECISION_FIELDS: Tuple[str, ...] = (
        "action",
        "allowed",
        "blocked",
        "safe",
        "is_safe",
        "decision",
        "verdict",
        "status",
        "score",
        "risk_score",
        "confidence",
        "categories",
        "classifications",
        "violations",
        "threats",
        "policies",
        "reason",
        "rules",
        "sanitized_text",
        "sanitizedText",
        "sanitized_payload",
    )

    @classmethod
    def _has_decision_fields(cls, payload: object) -> bool:
        if not isinstance(payload, dict):
            return False
        return any(key in payload for key in cls._DECISION_FIELDS)

    @classmethod
    def _unwrap_verdict_envelope(
        cls, inspect_response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Return the dict that actually holds is_safe / action / rules.

        Cisco AI Defense returns the verdict at different nesting depths
        depending on the endpoint and SDK version:

        * ``/api/v1/inspect/chat`` — verdict is at the top level.
        * ``/api/v1/inspect/mcp`` — JSON-RPC envelope wraps the verdict
          under ``result``.
        * Some SDKs nest under ``data`` / ``inspection`` / ``ai_defense``.

        Mirrors the reference plugin's ``_decision_payload`` so the
        handler tolerates every shape Cisco's own tested integration
        already supports.
        """
        if not isinstance(inspect_response, dict):
            return {}

        if cls._has_decision_fields(inspect_response):
            return inspect_response

        for key in ("result", "data", "inspection", "ai_defense", "aiDefense"):
            value = inspect_response.get(key)
            if cls._has_decision_fields(value):
                return value  # type: ignore[return-value]

        result = inspect_response.get("result")
        if isinstance(result, dict):
            for key in ("data", "inspection", "ai_defense", "aiDefense"):
                value = result.get(key)
                if cls._has_decision_fields(value):
                    return value  # type: ignore[return-value]

        return inspect_response

    @staticmethod
    def _extract_jsonrpc_error(
        inspect_response: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Detect a JSON-RPC error envelope inside an HTTP 200 response.

        The Cisco Inspect API can return ``{"error": {...}}`` (or nest one
        under ``"result"``) inside a 200. We treat that the same as a
        transport error so the configured ``fallback_on_error`` policy
        applies.
        """
        if not isinstance(inspect_response, dict):
            return None
        error = inspect_response.get("error")
        if isinstance(error, dict):
            return error
        result = inspect_response.get("result")
        if isinstance(result, dict):
            inner = result.get("error")
            if isinstance(inner, dict):
                return inner
        return None

    @staticmethod
    def _normalize_action(raw_action: str) -> str:
        """Map Cisco/reference-plugin action vocabulary to ours."""
        normalized = raw_action.strip().lower()
        if normalized in {
            "deny",
            "denied",
            "block",
            "blocked",
            "reject",
            "rejected",
            "unsafe",
            "malicious",
        }:
            return _ACTION_BLOCK
        if normalized in {"redact", "redacted", "sanitize", "sanitized", "mask"}:
            return _ACTION_REDACT
        if normalized in {"allow", "allowed", "safe", "ok"}:
            return _ACTION_ALLOW
        verbose_proxy_logger.warning(
            "Cisco AI Defense guardrail: unrecognized action %r treated as block",
            raw_action,
        )
        return _ACTION_BLOCK

    @staticmethod
    def _extract_sanitized_text(
        inspect_response: Dict[str, Any],
    ) -> Optional[str]:
        """Pull ``sanitized_text`` (or camelCase variant) off the verdict."""
        for key in ("sanitized_text", "sanitizedText"):
            value = inspect_response.get(key)
            if isinstance(value, str) and value:
                return value
        result = inspect_response.get("result")
        if isinstance(result, dict):
            for key in ("sanitized_text", "sanitizedText"):
                value = result.get(key)
                if isinstance(value, str) and value:
                    return value
        return None

    @staticmethod
    def _extract_sanitized_messages(
        inspect_response: Dict[str, Any],
    ) -> Optional[List[Dict[str, Any]]]:
        """Pull a sanitized OpenAI-format messages array off the verdict.

        Cisco can return the rewrite under several keys; we accept any of
        the common variants and stop at the first non-empty match.
        """
        containers = [inspect_response]
        for container_key in ("result", "data"):
            container = inspect_response.get(container_key)
            if isinstance(container, dict):
                containers.append(container)

        for container in containers:
            for key in (
                "sanitized_messages",
                "sanitizedMessages",
                "modified_messages",
                "modifiedMessages",
            ):
                value = container.get(key)
                if isinstance(value, list) and value:
                    return [m for m in value if isinstance(m, dict)]
            for key in (
                "sanitized_payload",
                "sanitizedPayload",
                "modified_payload",
                "modifiedPayload",
            ):
                payload = container.get(key)
                if isinstance(payload, dict):
                    messages = payload.get("messages")
                    if isinstance(messages, list) and messages:
                        return [m for m in messages if isinstance(m, dict)]
        return None

    def _apply_redaction(
        self,
        request_data: dict,
        response_obj: object,
        context: _ScanContext,
        verdict: _CiscoVerdict,
    ) -> bool:
        """Apply a Cisco-supplied rewrite to the request/response in place.

        Returns True when a rewrite was applied; False when there was no
        suitable surface to rewrite (caller then falls back to
        ``on_flagged_action``).
        """
        if context.surface == "mcp" and context.direction == "input":
            return self._redact_mcp_input(
                request_data, verdict.sanitized_text, verdict.sanitized_mcp_arguments
            )
        if context.surface == "mcp" and context.direction == "output":
            if response_obj is None:
                return False
            if verdict.sanitized_text:
                return self._set_mcp_tool_response_text(
                    response_obj, verdict.sanitized_text
                )
            return False
        if context.surface == "chat" and context.direction == "input":
            return self._redact_chat_input(
                request_data, verdict.sanitized_text, verdict.sanitized_messages
            )
        if context.surface == "chat" and context.direction == "output":
            return self._redact_chat_output(
                response_obj, verdict.sanitized_text, verdict.sanitized_messages
            )
        return False

    @staticmethod
    def _redact_mcp_input(
        request_data: dict,
        sanitized_text: Optional[str],
        sanitized_mcp_arguments: Optional[Dict[str, Any]],
    ) -> bool:
        """Rewrite MCP request arguments in all locations the proxy reads."""
        if sanitized_mcp_arguments is not None:
            request_data["mcp_arguments"] = sanitized_mcp_arguments
            request_data["modified_arguments"] = sanitized_mcp_arguments
            params = request_data.get("params")
            if isinstance(params, dict):
                params["arguments"] = sanitized_mcp_arguments
            if isinstance(request_data.get("arguments"), dict):
                request_data["arguments"] = sanitized_mcp_arguments
            return True
        if sanitized_text:
            applied = False
            for args_path in (
                request_data.get("mcp_arguments"),
                request_data.get("arguments"),
                (request_data.get("params") or {}).get("arguments"),
            ):
                if not isinstance(args_path, dict):
                    continue
                string_keys = [
                    key for key, value in args_path.items() if isinstance(value, str)
                ]
                if len(string_keys) != 1:
                    continue
                args_path[string_keys[0]] = sanitized_text
                request_data["modified_arguments"] = args_path
                applied = True
            return applied
        return False

    def _redact_chat_input(
        self,
        request_data: dict,
        sanitized_text: Optional[str],
        sanitized_messages: Optional[List[Dict[str, Any]]],
    ) -> bool:
        """Rewrite chat request input (``messages`` or ``input``)."""
        if sanitized_messages and self._extract_tool_definition_text(request_data):
            # We append one synthetic message carrying the tool/function
            # definitions for inspection; Cisco echoes it back in
            # ``sanitized_messages``, but it maps to no structured request
            # field, so drop it before rewriting the real conversation.
            sanitized_messages = sanitized_messages[:-1] or None
        uses_input = "input" in request_data and "messages" not in request_data
        has_instructions = request_data.get("instructions") is not None
        instructions_redacted = False
        if has_instructions:
            instructions_redacted = self._redact_responses_instructions(
                request_data, sanitized_text, sanitized_messages
            )
            sanitized_messages = self._non_instruction_messages(sanitized_messages)
            if not sanitized_messages:
                return instructions_redacted
        if sanitized_messages:
            if uses_input:
                rewritten = self._sanitized_messages_to_responses_input(
                    sanitized_messages
                )
                if rewritten is not None:
                    request_data["input"] = rewritten
                    return True
                return False
            request_data["messages"] = sanitized_messages
            return True
        if sanitized_text:
            if uses_input:
                rewritten_input = self._rewrite_responses_input_text(
                    request_data.get("input"), sanitized_text
                )
                if rewritten_input is not None:
                    request_data["input"] = rewritten_input
                    return True
                return False
            redacted_arguments = self._clear_chat_input_tool_arguments(request_data)
            messages = request_data.get("messages")
            redacted_content = False
            if isinstance(messages, list) and messages:
                for message in reversed(messages):
                    if (
                        isinstance(message, dict)
                        and message.get("role") == "user"
                        and isinstance(message.get("content"), str)
                    ):
                        message["content"] = sanitized_text
                        redacted_content = True
                        break
            return redacted_content or redacted_arguments
        return False

    @classmethod
    def _redact_responses_instructions(
        cls,
        request_data: dict,
        sanitized_text: Optional[str],
        sanitized_messages: Optional[List[Dict[str, Any]]],
    ) -> bool:
        if sanitized_messages:
            instruction_text = cls._instruction_text_from_messages(sanitized_messages)
            if instruction_text:
                request_data["instructions"] = instruction_text
                return True
        if sanitized_text and not any(
            key in request_data for key in ("input", "messages", "prompt")
        ):
            request_data["instructions"] = sanitized_text
            return True
        return False

    @classmethod
    def _instruction_text_from_messages(
        cls, messages: List[Dict[str, Any]]
    ) -> Optional[str]:
        for message in messages:
            if not isinstance(message, dict):
                continue
            if cls._is_instruction_role(message.get("role")):
                text = cls._normalize_message_content(message.get("content"))
                if text:
                    return text
        return None

    @classmethod
    def _non_instruction_messages(
        cls, messages: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        if messages is None:
            return None
        return [
            message
            for message in messages
            if not (
                isinstance(message, dict)
                and cls._is_instruction_role(message.get("role"))
            )
        ]

    @staticmethod
    def _is_instruction_role(role: object) -> bool:
        return isinstance(role, str) and role.lower() in {"system", "developer"}

    @classmethod
    def _clear_chat_input_tool_arguments(cls, request_data: dict) -> bool:
        messages = request_data.get("messages")
        if not isinstance(messages, list):
            return False
        applied = False
        for message in messages:
            if not isinstance(message, dict):
                continue
            if cls._extract_message_tool_argument_parts(message):
                cls._clear_tool_call_arguments(message)
                applied = True
        return applied

    def _redact_chat_output(
        self,
        response_obj: object,
        sanitized_text: Optional[str],
        sanitized_messages: Optional[List[Dict[str, Any]]],
    ) -> bool:
        """Rewrite chat response (``ModelResponse`` or ``ResponsesAPIResponse``)."""
        if response_obj is None:
            return False

        if isinstance(response_obj, TextCompletionResponse):
            return self._redact_text_completion_choices(
                getattr(response_obj, "choices", None) or [],
                sanitized_text,
                sanitized_messages,
            )

        choices = getattr(response_obj, "choices", None)
        if isinstance(choices, list):
            return self._redact_model_response_choices(
                choices, sanitized_text, sanitized_messages
            )

        output_items = getattr(response_obj, "output", None)
        if isinstance(output_items, list):
            return self._redact_responses_api_output(
                output_items, sanitized_text, sanitized_messages
            )

        return False

    @staticmethod
    def _redact_model_response_choices(
        choices: list,
        sanitized_text: Optional[str],
        sanitized_messages: Optional[List[Dict[str, Any]]],
    ) -> bool:
        """Redact every returned choice, including tool-call/reasoning fields."""
        if sanitized_messages:
            applied = False
            msg_iter = iter(sanitized_messages)
            for choice in choices:
                if not isinstance(choice, Choices):
                    continue
                replacement = next(msg_iter, None)
                replacement_text = sanitized_text or "[REDACTED]"
                if replacement is not None:
                    text = CiscoAIDefenseGuardrail._normalize_message_content(
                        replacement.get("content")
                    )
                    if text:
                        replacement_text = text
                        choice.message.content = text
                        applied = True
                else:
                    if getattr(choice.message, "content", None):
                        choice.message.content = replacement_text
                        applied = True
                if CiscoAIDefenseGuardrail._redact_message_reasoning_fields(
                    choice.message, replacement_text
                ):
                    applied = True
                CiscoAIDefenseGuardrail._clear_tool_call_arguments(choice.message)
            return applied
        if sanitized_text:
            applied = False
            for choice in choices:
                if not isinstance(choice, Choices):
                    continue
                msg = choice.message
                if getattr(msg, "content", None):
                    msg.content = sanitized_text
                    applied = True
                if CiscoAIDefenseGuardrail._redact_message_reasoning_fields(
                    msg, sanitized_text
                ):
                    applied = True
                CiscoAIDefenseGuardrail._clear_tool_call_arguments(msg)
            return applied
        return False

    @staticmethod
    def _redact_text_completion_choices(
        choices: list,
        sanitized_text: Optional[str],
        sanitized_messages: Optional[List[Dict[str, Any]]],
    ) -> bool:
        """Rewrite ``/v1/completions`` text choices after Cisco redaction."""
        replacement = sanitized_text
        if not replacement and sanitized_messages:
            for message in sanitized_messages:
                if not isinstance(message, dict):
                    continue
                text = CiscoAIDefenseGuardrail._normalize_message_content(
                    message.get("content")
                )
                if text:
                    replacement = text
                    break
        if not replacement:
            return False
        applied = False
        for choice in choices:
            if getattr(choice, "text", None):
                choice.text = replacement
                applied = True
        return applied

    @classmethod
    def _redact_message_reasoning_fields(
        cls, message: object, replacement_text: str
    ) -> bool:
        """Remove preserved reasoning fields and expose the sanitized text."""
        if not cls._extract_message_reasoning_parts(message):
            return False
        setattr(message, "content", replacement_text)
        for key in ("reasoning_content", "thinking_blocks", "reasoning_items"):
            if not hasattr(message, key):
                continue
            try:
                delattr(message, key)
            except (AttributeError, TypeError, ValueError):
                try:
                    setattr(message, key, None)
                except (AttributeError, TypeError, ValueError):
                    pass
        return True

    @staticmethod
    def _clear_arguments_field(obj: object) -> None:
        """Set ``obj.arguments`` (or ``obj["arguments"]``) to ``"{}"``."""
        if obj is None:
            return
        if isinstance(obj, dict):
            obj["arguments"] = "{}"
            return
        try:
            setattr(obj, "arguments", "{}")
        except (AttributeError, TypeError, ValueError):
            pass

    @classmethod
    def _clear_tool_call_arguments(cls, message: object) -> None:
        """Clear tool-call / function-call arguments after Cisco redaction."""
        tool_calls = (
            message.get("tool_calls")
            if isinstance(message, dict)
            else getattr(message, "tool_calls", None)
        )
        for tc in tool_calls or []:
            fn = (
                tc.get("function")
                if isinstance(tc, dict)
                else getattr(tc, "function", None)
            )
            cls._clear_arguments_field(fn)
        function_call = (
            message.get("function_call")
            if isinstance(message, dict)
            else getattr(message, "function_call", None)
        )
        cls._clear_arguments_field(function_call)

    def _redact_responses_api_output(
        self,
        output_items: list,
        sanitized_text: Optional[str],
        sanitized_messages: Optional[List[Dict[str, Any]]],
    ) -> bool:
        replacement_text: Optional[str] = sanitized_text
        if not replacement_text and sanitized_messages:
            replacement_text = " ".join(
                self._normalize_message_content(m.get("content"))
                for m in sanitized_messages
                if isinstance(m, dict)
            ).strip()
        if not replacement_text:
            return False
        applied = False
        for item in output_items:
            content = getattr(item, "content", None) or (
                item.get("content") if isinstance(item, dict) else None
            )
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") in self._TEXT_PART_TYPES:
                            part["text"] = replacement_text
                            applied = True
                    else:
                        ptype = getattr(part, "type", None)
                        if ptype in self._TEXT_PART_TYPES:
                            try:
                                setattr(part, "text", replacement_text)
                                applied = True
                            except (AttributeError, TypeError, ValueError):
                                continue
            args = (
                item.get("arguments")
                if isinstance(item, dict)
                else getattr(item, "arguments", None)
            )
            if isinstance(args, str) and args:
                self._clear_arguments_field(item)
                applied = True
        return applied

    @staticmethod
    def _sanitized_messages_to_responses_input(
        sanitized_messages: List[Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        """Convert chat-shape sanitized_messages to Responses API ``input``.

        Returns ``None`` if nothing usable could be converted, so the
        caller falls back to ``on_flagged_action``.
        """
        out: List[Dict[str, Any]] = []
        for m in sanitized_messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role") or "user"
            content = m.get("content")
            if isinstance(content, str):
                ptype = "output_text" if role == "assistant" else "input_text"
                out.append(
                    {"role": role, "content": [{"type": ptype, "text": content}]}
                )
            elif isinstance(content, list):
                out.append({"role": role, "content": content})
        return out or None

    @staticmethod
    def _rewrite_responses_input_text(
        original_input: object, sanitized_text: str
    ) -> Optional[object]:
        """Apply ``sanitized_text`` to a Responses API ``input`` value.

        Handles plain string, list of message items (rewrites the last
        user item's first text part), and flat list of content parts.
        Returns ``None`` if no text part could be rewritten.
        """
        if isinstance(original_input, str):
            return sanitized_text
        if not isinstance(original_input, list):
            return None

        text_types = CiscoAIDefenseGuardrail._TEXT_PART_TYPES
        has_messages = any(isinstance(i, dict) and "role" in i for i in original_input)

        if has_messages:
            rewritten = list(original_input)
            for idx in range(len(rewritten) - 1, -1, -1):
                item = rewritten[idx]
                if not (isinstance(item, dict) and item.get("role") == "user"):
                    continue
                content = item.get("content")
                if isinstance(content, str):
                    rewritten[idx] = {**item, "content": sanitized_text}
                    return rewritten
                if isinstance(content, list):
                    new_content = list(content)
                    for j, part in enumerate(new_content):
                        if isinstance(part, dict) and part.get("type") in text_types:
                            new_content[j] = {**part, "text": sanitized_text}
                            rewritten[idx] = {**item, "content": new_content}
                            return rewritten
            return None

        rewritten_parts = list(original_input)
        for j, part in enumerate(rewritten_parts):
            if isinstance(part, dict) and part.get("type") in text_types:
                rewritten_parts[j] = {**part, "text": sanitized_text}
                return rewritten_parts
        return None

    @staticmethod
    def _extract_masked_entity_count(
        rules: List[Dict[str, Any]],
    ) -> Optional[Dict[str, int]]:
        """Count entity-type detections per Cisco rule for the logging payload."""
        if not rules:
            return None
        counts: Dict[str, int] = {}
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            entity_types = rule.get("entity_types") or []
            for entity_type in entity_types:
                if not isinstance(entity_type, str):
                    continue
                counts[entity_type] = counts.get(entity_type, 0) + 1
        return counts or None

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def _handle_api_error(
        self,
        error: Exception,
        *,
        request_data: Optional[dict] = None,
        start_time: Optional[datetime] = None,
        surface: str = "chat",
        direction: str = "input",
    ) -> Dict[str, Any]:
        verbose_proxy_logger.error(
            "Cisco AI Defense guardrail (%s): API communication failed: %s",
            surface,
            error,
        )

        if request_data is not None and start_time is not None:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            if surface == "mcp":
                evt = (
                    GuardrailEventHooks.during_mcp_call
                    if direction == "output"
                    else GuardrailEventHooks.pre_mcp_call
                )
            else:
                evt = (
                    GuardrailEventHooks.post_call
                    if direction == "output"
                    else GuardrailEventHooks.pre_call
                )
            self.add_standard_logging_guardrail_information_to_request_data(
                guardrail_provider=self._PROVIDER_NAME,
                guardrail_json_response={
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "surface": surface,
                },
                request_data=request_data,
                guardrail_status="guardrail_failed_to_respond",
                start_time=start_time.timestamp(),
                end_time=end_time.timestamp(),
                duration=duration,
                event_type=evt,
            )

        if self.fallback_on_error == "allow":
            verbose_proxy_logger.warning(
                "Cisco AI Defense guardrail: API unavailable, proceeding "
                "without scanning (fallback_on_error='allow')"
            )
            return {
                "is_safe": True,
                "classifications": [],
                "_unscanned": True,
            }

        raise HTTPException(
            status_code=503,
            detail={
                "error": "Cisco AI Defense guardrail unavailable",
                "message": (
                    "Cisco AI Defense scanning service is temporarily "
                    "unavailable and fallback_on_error='block'"
                ),
                "error_type": type(error).__name__,
            },
        )

    # ------------------------------------------------------------------
    # Message extraction helpers
    # ------------------------------------------------------------------

    # Content-part ``type`` values that should be flattened to text by
    # ``_normalize_message_content``. Covers both Chat Completions
    # (``text``) and the Responses API (``input_text`` for caller-side
    # parts, ``output_text`` for assistant turns, ``summary_text`` /
    # ``reasoning_text`` for reasoning summaries that may appear in
    # conversation history).
    _TEXT_PART_TYPES = frozenset(
        {"text", "input_text", "output_text", "summary_text", "reasoning_text"}
    )

    @staticmethod
    def _extract_inspect_messages_from_request(
        data: dict,
    ) -> List[Dict[str, str]]:
        """Build {role, content} messages for the Cisco AI Defense chat API."""
        messages: List[Dict[str, str]] = []

        instructions_text = CiscoAIDefenseGuardrail._normalize_message_content(
            data.get("instructions")
        )
        if instructions_text:
            messages.append({"role": "system", "content": instructions_text})

        raw_messages = data.get("messages") or []
        for message in raw_messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if not role:
                continue
            parts: List[str] = []
            text = CiscoAIDefenseGuardrail._normalize_message_content(
                message.get("content")
            )
            if text:
                parts.append(text)
            parts.extend(
                CiscoAIDefenseGuardrail._extract_message_tool_argument_parts(message)
            )
            if parts:
                messages.append({"role": role, "content": " ".join(parts)})

        if "input" in data:
            # Responses API ``input`` can be: a plain string, a list of
            # message-shaped dicts (with role + nested content array), or
            # a flat list of content-part dicts. Flatten properly so the
            # scan sees every text segment, not just the top-level ones.
            messages.extend(
                CiscoAIDefenseGuardrail._flatten_responses_input(data.get("input"))
            )

        if not messages and data.get("prompt") is not None:
            prompt_text = CiscoAIDefenseGuardrail._normalize_message_content(
                data.get("prompt")
            )
            if prompt_text:
                messages.append({"role": "user", "content": prompt_text})

        tool_text = CiscoAIDefenseGuardrail._extract_tool_definition_text(data)
        if tool_text:
            messages.append({"role": "system", "content": tool_text})

        return messages

    @staticmethod
    def _extract_tool_definition_text(data: dict) -> str:
        """Flatten request-side tool/function definitions into scannable text.

        Tool definitions (names, descriptions, nested JSON-schema docs) are
        forwarded to the model, so attacker-controlled text placed there must
        be inspected too; otherwise it bypasses the guardrail by hiding in
        ``tools[].function.description`` and similar metadata.
        """
        parts: List[str] = []
        for key in ("tools", "functions"):
            CiscoAIDefenseGuardrail._collect_strings(data.get(key), parts)
        return " ".join(parts)

    @staticmethod
    def _collect_strings(value: object, out: List[str]) -> None:
        if isinstance(value, str):
            if value:
                out.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                CiscoAIDefenseGuardrail._collect_strings(item, out)
        elif isinstance(value, list):
            for item in value:
                CiscoAIDefenseGuardrail._collect_strings(item, out)

    @staticmethod
    def _flatten_responses_input(input_value: object) -> List[Dict[str, str]]:
        """Flatten the OpenAI Responses API ``input`` into chat-message form.

        Recognized shapes:

        1. Plain string -> one user message.
        2. List of message-shaped dicts
           ``{"role": "...", "content": [<content parts>]}`` -> one
           message per item, with the role preserved.
        3. Flat list of content-part dicts
           ``{"type": "input_text", "text": "..."}`` -> single user
           message containing the concatenated text.

        """
        if input_value is None:
            return []
        if isinstance(input_value, str):
            return [{"role": "user", "content": input_value}]
        if not isinstance(input_value, list):
            text = str(input_value)
            return [{"role": "user", "content": text}] if text else []

        if any(isinstance(item, dict) and "role" in item for item in input_value):
            result: List[Dict[str, str]] = []
            for item in input_value:
                if not isinstance(item, dict):
                    continue
                role = item.get("role") or "user"
                text = CiscoAIDefenseGuardrail._normalize_message_content([item])
                if text:
                    result.append({"role": role, "content": text})
            return result

        text = CiscoAIDefenseGuardrail._normalize_message_content(input_value)
        return [{"role": "user", "content": text}] if text else []

    @staticmethod
    def _normalize_message_content(content: object) -> str:
        """Coerce OpenAI multi-modal content into a plain text string.

        Supports:

        * Plain string.
        * List of content-part dicts where ``type`` is one of
          ``text`` (Chat Completions), ``input_text`` / ``output_text`` /
          ``summary_text`` (Responses API).
        * List of message-shaped dicts with a nested ``content`` list —
          recurses into the nested content so a Responses API ``input``
          item like ``{"role":"user","content":[{"type":"input_text",...}]}``
          gets flattened correctly.
        """
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type in CiscoAIDefenseGuardrail._TEXT_PART_TYPES and part.get(
                    "text"
                ):
                    parts.append(str(part["text"]))
                    continue
                nested = part.get("content")
                if nested is not None:
                    nested_text = CiscoAIDefenseGuardrail._normalize_message_content(
                        nested
                    )
                    if nested_text:
                        parts.append(nested_text)
                for key in ("arguments", "output"):
                    value = part.get(key)
                    if value:
                        parts.append(
                            CiscoAIDefenseGuardrail._normalize_message_content(value)
                        )
            return " ".join(parts)
        return str(content)

    @staticmethod
    def _extract_response_messages(response: object) -> List[Dict[str, str]]:
        """Extract scannable assistant text from a chat response.

        Handles both ``ModelResponse`` (Chat Completions) and
        ``ResponsesAPIResponse`` (``/v1/responses``). On both shapes
        tool-call / function-call argument strings and reasoning fields
        are included alongside the main text so a model can't bypass the
        scan by placing content there.
        """
        if isinstance(response, ModelResponse):
            result: List[Dict[str, str]] = []
            for choice in getattr(response, "choices", None) or []:
                if not isinstance(choice, Choices):
                    continue
                parts: List[str] = []
                content = CiscoAIDefenseGuardrail._normalize_message_content(
                    getattr(choice.message, "content", None)
                )
                if content:
                    parts.append(content)
                parts.extend(
                    CiscoAIDefenseGuardrail._extract_message_tool_argument_parts(
                        choice.message
                    )
                )
                parts.extend(
                    CiscoAIDefenseGuardrail._extract_message_reasoning_parts(
                        choice.message
                    )
                )
                if parts:
                    result.append({"role": "assistant", "content": " ".join(parts)})
            return result

        if isinstance(response, TextCompletionResponse):
            text_parts: List[str] = []
            for choice in getattr(response, "choices", None) or []:
                text = getattr(choice, "text", None)
                if isinstance(text, str) and text:
                    text_parts.append(text)
            joined = " ".join(text_parts)
            return [{"role": "assistant", "content": joined}] if joined else []

        output_items = getattr(response, "output", None)
        if not isinstance(output_items, list):
            return []
        output_parts: List[str] = []
        for item in output_items:
            get = (
                item.get
                if isinstance(item, dict)
                else (lambda k: getattr(item, k, None))
            )
            for part in get("content") or []:
                pget = (
                    part.get
                    if isinstance(part, dict)
                    else (lambda k: getattr(part, k, None))
                )
                for key in ("text", "reasoning", "thinking"):
                    value = pget(key)
                    if isinstance(value, str) and value:
                        output_parts.append(value)
            args = get("arguments")
            if isinstance(args, str) and args:
                output_parts.append(args)
            direct = get("text")
            if isinstance(direct, str) and direct:
                output_parts.append(direct)
        joined = " ".join(output_parts)
        return [{"role": "assistant", "content": joined}] if joined else []

    @classmethod
    def _extract_message_reasoning_parts(cls, message: object) -> List[str]:
        """Extract inspectable reasoning fields from a message/delta object."""
        parts: List[str] = []
        reasoning_content = cls._field(message, "reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content:
            parts.append(reasoning_content)
        for block in cls._field_list(message, "thinking_blocks"):
            # Do not forward redacted_thinking.data; it is opaque provider
            # metadata rather than scannable plaintext.
            for key in ("thinking", "reasoning", "text"):
                value = cls._field(block, key)
                if isinstance(value, str) and value:
                    parts.append(value)
        for item in cls._field_list(message, "reasoning_items"):
            for block in cls._field_list(item, "summary"):
                text = cls._field(block, "text")
                if isinstance(text, str) and text:
                    parts.append(text)
            for key in ("text", "reasoning", "reasoning_content"):
                value = cls._field(item, key)
                if isinstance(value, str) and value:
                    parts.append(value)
        return parts

    @staticmethod
    def _field(obj: object, key: str) -> object:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    @classmethod
    def _field_list(cls, obj: object, key: str) -> List[Any]:
        value = cls._field(obj, key)
        return value if isinstance(value, list) else []

    @classmethod
    def _extract_message_tool_argument_parts(cls, message: object) -> List[str]:
        parts: List[str] = []
        tool_calls = (
            message.get("tool_calls")
            if isinstance(message, dict)
            else getattr(message, "tool_calls", None)
        )
        for tool_call in tool_calls or []:
            args = cls._extract_tool_call_arguments(tool_call)
            if args:
                parts.append(args)
        function_call = (
            message.get("function_call")
            if isinstance(message, dict)
            else getattr(message, "function_call", None)
        )
        if function_call is not None:
            args = cls._extract_function_call_arguments(function_call)
            if args:
                parts.append(args)
        return parts

    @staticmethod
    def _extract_tool_call_arguments(tool_call: object) -> Optional[str]:
        """Pull ``function.arguments`` off a tool_calls entry (dict or model)."""
        if tool_call is None:
            return None
        function = (
            tool_call.get("function")
            if isinstance(tool_call, dict)
            else getattr(tool_call, "function", None)
        )
        return CiscoAIDefenseGuardrail._extract_function_call_arguments(function)

    @staticmethod
    def _extract_function_call_arguments(function_call: object) -> Optional[str]:
        """Pull ``arguments`` off a function_call entry (dict or model)."""
        if function_call is None:
            return None
        args = (
            function_call.get("arguments")
            if isinstance(function_call, dict)
            else getattr(function_call, "arguments", None)
        )
        if args is None:
            return None
        return str(args)

    # ------------------------------------------------------------------
    # Config model surface
    # ------------------------------------------------------------------

    @staticmethod
    def get_config_model() -> Optional[Type["GuardrailConfigModel"]]:
        from litellm.types.proxy.guardrails.guardrail_hooks.cisco_ai_defense import (
            CiscoAIDefenseGuardrailConfigModel,
        )

        return CiscoAIDefenseGuardrailConfigModel
