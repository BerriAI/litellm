# +-------------------------------------------------------------+
#
#               Cisco AI Defense Inspection Guardrail
#       https://developer.cisco.com/docs/ai-defense-inspection/
#
# +-------------------------------------------------------------+
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
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
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

# Action vocabulary Cisco AI Defense can return / our heuristic produces.
_ACTION_BLOCK = "block"
_ACTION_REDACT = "redact"
_ACTION_ALLOW = "allow"


class CiscoAIDefenseGuardrailMissingSecrets(Exception):
    """Raised when the Cisco AI Defense API key is missing."""


class CiscoAIDefenseGuardrailAPIError(Exception):
    """Raised when there is an error talking to the Cisco AI Defense API."""


class CiscoAIDefenseGuardrail(_CiscoAIDefenseMcpMixin, CustomGuardrail):
    """
    Cisco AI Defense guardrail integration.

    Each instance scans exactly one inspection surface (``chat`` or ``mcp``)
    via the corresponding Cisco AI Defense Inspection API endpoint.

    MCP-specific hooks (``async_post_mcp_tool_call_hook``,
    ``_inspect_mcp_request`` / ``_inspect_mcp_response``, JSON-RPC payload
    builders, redact helper) live on ``_CiscoAIDefenseMcpMixin`` in
    ``cisco_ai_defense_mcp.py`` to keep this module focused on the chat
    surface. The mixin's methods call back into shared infrastructure on
    ``self`` (``self._post_inspection``, ``self._finalize_inspection``,
    ``self._handle_api_error``, etc.); Python's late binding resolves
    those at call time so the split is purely organizational — public
    behaviour and imports are unchanged.
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

        # MCP-only event hooks (``pre_mcp_call`` / ``during_mcp_call``) are
        # unambiguous about the surface the user wants to scan — auto-infer
        # ``inspection_type=mcp`` so the dashboard form only requires picking
        # one field. The reverse direction is the default already (chat
        # inspection + chat hooks).
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

        # Derive the inspection endpoint path from the surface, with an
        # explicit override available for non-standard deployments.
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

        self.enabled_rules = enabled_rules or None
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

        # Advertise every event hook the proxy can dispatch so the
        # dashboard's "create guardrail" form accepts any combination of
        # ``mode`` + ``inspection_type``. At runtime ``_surface_matches()``
        # is the source of truth: chat-mode guardrails are no-ops for MCP
        # traffic, and mcp-mode guardrails are no-ops for chat traffic.
        # Validating compatibility here would just re-create the
        # construction-time error users see when they pick e.g.
        # mode=pre_mcp_call but forget to flip inspection_type=mcp; the
        # PANW Prisma AIRS handler takes the same all-hooks approach for
        # exactly this reason.
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

        # Best-effort warning when the configured ``mode`` doesn't line up
        # with ``inspection_type`` — the guardrail will register cleanly
        # but won't actually scan any traffic because the surface filter
        # will reject everything. Runs after super() so self.guardrail_name
        # is populated.
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
    def _coerce_timeout(value: Any) -> Optional[float]:
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
    # Surface decision rule
    # ------------------------------------------------------------------
    # The pre-call, moderation, and post-call hooks deliberately do NOT
    # sniff the request/response body shape to decide whether traffic is
    # chat or MCP. The proxy-asserted ``call_type`` is the authoritative
    # signal; the body is caller-controlled and was previously used as
    # an OR-clause that let a chat-completion caller bypass a chat-mode
    # guardrail by sprinkling ``mcp_tool_name`` / ``mcp_arguments`` /
    # ``"jsonrpc": "2.0"`` into the payload (Veria AI security review on
    # PR #28249, High severity). Helpers like ``_is_mcp_request_shape``
    # / ``_is_mcp_response_shape`` were removed so they cannot be
    # re-introduced into the surface decision by accident.

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
        # SECURITY: do NOT OR in ``_is_mcp_request_shape(data)`` here. The
        # request body is caller-controlled — a chat completion that adds
        # spoofed ``mcp_tool_name`` / ``mcp_arguments`` / ``jsonrpc`` fields
        # would have flipped ``is_mcp`` to True, and a chat-mode guardrail
        # would then have skipped scanning via ``_surface_matches``. The
        # proxy is the authoritative source of the call surface; treat
        # ``call_type`` as the only signal. See Veria AI security review
        # on PR #28249 ("chat guardrail bypass via user-controlled MCP
        # shape", High severity).
        is_mcp = self._is_mcp_call_type(call_type)

        if not self._surface_matches(is_mcp):
            verbose_proxy_logger.debug(
                "Cisco AI Defense guardrail: call_type=%s does not match "
                "configured inspection_type=%s, skipping",
                call_type,
                self.inspection_type,
            )
            return data

        # Honor explicit event_hook config; for MCP-mode guardrails the
        # framework dispatches via pre_mcp_call.
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
        # SECURITY: see ``async_pre_call_hook`` — surface is decided by
        # the proxy-asserted ``call_type``, never by caller-controlled
        # request-body shape, to prevent chat-guardrail bypass.
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
        # Chat-mode only — MCP responses come through
        # async_post_mcp_tool_call_hook, which the proxy dispatches as a
        # CustomLogger hook (post_call is intentionally not in our event
        # hook list for mcp-mode guardrails).
        if self.inspection_type != "chat":
            return response

        # SECURITY: do NOT early-return based on payload-shape sniffing
        # of either the request body or the model response. Both are
        # caller / model controlled — a chat request containing spoofed
        # ``mcp_tool_name`` fields, or a model emitting a JSON-RPC-shaped
        # text payload, would otherwise bypass the chat post-call scan.
        # The proxy only dispatches this hook on actual chat-completion
        # paths (MCP tool responses flow through
        # ``async_post_mcp_tool_call_hook``), so the inspection-type guard
        # above is the right surface gate. See Veria AI security review
        # on PR #28249 ("chat guardrail bypass via user-controlled MCP
        # shape", High severity).

        if (
            self.should_run_guardrail(
                data=data, event_type=GuardrailEventHooks.post_call
            )
            is not True
        ):
            return response

        # ``_extract_response_messages`` handles both ``ModelResponse``
        # (Chat Completions) and ``ResponsesAPIResponse`` (``/v1/responses``)
        # — do NOT gate on ``isinstance(response, ModelResponse)`` here;
        # that silently bypassed Responses API output (Veria AI High).
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
        response: Any,
        request_data: dict,
    ):
        """Scan the assembled streaming chat response BEFORE it's delivered.

        Without this hook the proxy lets streamed chunks reach the client
        first and runs ``async_post_call_success_hook`` only after the
        stream is closed — by which point any violations have already
        been delivered. A caller can therefore set ``stream: true`` to
        bypass chat-mode response scanning. (Veria AI security review on
        PR #28249, High severity: "Streaming output bypass".)

        Pattern matches the other partner output guardrails (Pillar,
        PANW Prisma AIRS, Lasso): buffer every chunk, assemble into a
        ``ModelResponse`` via ``stream_chunk_builder``, run the chat
        inspection on the assembled text, then either re-yield the
        original chunks (allow), emit an SSE error event (block), or
        emit modified chunks via ``MockResponseIterator`` (redact).

        Important error-surfacing rule: raising ``HTTPException`` from a
        Python generator does not propagate as an HTTP error — the proxy
        treats it as a generic 500. We therefore catch the exception and
        ``yield`` a structured SSE error event whose code matches the
        exception's ``status_code``; ``create_response`` recognises that
        shape and returns the correct JSON error.
        """
        from litellm.llms.base_llm.base_model_iterator import MockResponseIterator
        from litellm.main import stream_chunk_builder

        # MCP-mode guardrails do not own the chat post-call surface — MCP
        # tool responses flow through ``async_post_mcp_tool_call_hook``.
        if self.inspection_type != "chat":
            async for chunk in response:
                yield chunk
            return

        if not self.should_run_guardrail(
            data=request_data, event_type=GuardrailEventHooks.post_call
        ):
            async for chunk in response:
                yield chunk
            return

        verbose_proxy_logger.debug(
            "Cisco AI Defense guardrail (%s): scanning streaming chat response.",
            self.guardrail_name,
        )

        # Buffer everything before we let any byte hit the wire. We must
        # not yield incrementally before the inspection completes — that
        # is the bypass the security review flagged.
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

        # If there are no chunks (or none are ``ModelResponse(Stream)``
        # shaped), assembly will fail or be meaningless. Be conservative:
        # if we can't assemble a scannable payload, allow the original
        # chunks through unchanged (we still recorded any earlier
        # failures via verbose_proxy_logger).
        if not all_chunks:
            return

        if not isinstance(all_chunks[0], (ModelResponse, ModelResponseStream)):
            # Non-OpenAI streaming shapes (Anthropic SSE bytes,
            # /v1/responses pydantic events, etc.). We can't parse or
            # assemble these today. Fail closed: emit an SSE error event
            # rather than delivering content we couldn't inspect.
            verbose_proxy_logger.warning(
                "Cisco AI Defense guardrail (%s): unsupported streaming "
                "chunk shape (%s) — failing closed.",
                self.guardrail_name,
                type(all_chunks[0]).__name__,
            )
            yield f'data: {json.dumps({"error": {"message": "Cisco AI Defense: unsupported streaming format — response withheld for safety", "type": "guardrail_unsupported_stream", "code": 400, "guardrail": self.guardrail_name}})}\n\n'
            return

        assembled = stream_chunk_builder(chunks=all_chunks)
        if not isinstance(assembled, ModelResponse):
            for chunk in all_chunks:
                yield chunk
            return

        # ``_inspect_chat`` shares its block/redact/allow vocabulary with
        # the non-streaming path. ``response_obj=assembled`` lets the
        # redact path rewrite the assistant message in place.
        response_messages = self._extract_response_messages(assembled)
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
            # See docstring: raising would be reported as a generic 500.
            # Format as a structured SSE error event.
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
            # Fail closed by default: a scan error must not let the
            # buffered chunks leak. The ``fallback_on_error`` policy is
            # already enforced inside ``_inspect_chat`` /
            # ``_handle_api_error`` for the non-streaming path; if we
            # reach here it means an unexpected exception bubbled out,
            # which the dispatcher would otherwise log as non-blocking
            # and deliver the original chunks. Emit a structured error
            # so the caller sees an explicit failure.
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

        # Allow / redact path. If ``_inspect_chat`` rewrote the assistant
        # content (redact action), the assembled ModelResponse now
        # contains the sanitized text and we need to deliver THAT, not
        # the buffered original chunks.
        if self._streaming_content_was_modified(all_chunks, assembled):
            mock_iterator = MockResponseIterator(model_response=assembled)
            async for chunk in mock_iterator:
                yield chunk
        else:
            for chunk in all_chunks:
                yield chunk

    def _build_block_payload(
        self,
        *,
        surface: str,
        direction: str,
        classifications: List[str],
        severity: Optional[str],
        rules: List[Dict[str, Any]],
        explanation: Optional[str],
        event_id: Optional[str],
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
            "surface": surface,
            "direction": direction,
            "action": "block",
            "classifications": list(classifications),
            "severity": severity,
            "rules": [r.get("rule_name") for r in rules if isinstance(r, dict)],
            "explanation": explanation,
            "event_id": event_id,
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

    @staticmethod
    def _streaming_content_was_modified(
        original_chunks: List[Any], assembled: ModelResponse
    ) -> bool:
        """Decide whether redact rewrote the assembled assistant content.

        We rebuild the original assistant text from the chunk stream and
        compare against the assembled response's content. Any drift means
        ``_inspect_chat`` applied a sanitized rewrite — we must yield the
        modified content instead of the original chunks.
        """
        original_text = ""
        for chunk in original_chunks:
            choices = getattr(chunk, "choices", None) or []
            for c in choices:
                delta = getattr(c, "delta", None)
                if delta is None:
                    continue
                text = getattr(delta, "content", None)
                if isinstance(text, str):
                    original_text += text

        assembled_choices = getattr(assembled, "choices", None) or []
        assembled_text = ""
        for c in assembled_choices:
            message = getattr(c, "message", None)
            if message is None:
                continue
            text = getattr(message, "content", None)
            if isinstance(text, str):
                assembled_text += text

        return original_text != assembled_text

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
    def _normalize_event_hooks(event_hook: Any) -> set:
        """Coerce a ``mode`` arg (str, enum, or list of either) to a set of values."""

        def _norm(hook: Any) -> Optional[str]:
            if hasattr(hook, "value"):
                return hook.value
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
    def _infer_inspection_type_from_mode(event_hook: Any, current: str) -> str:
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
        *,
        surface: str,
        direction: str,
        action: str,
        is_safe: Optional[bool],
        classifications: List[str],
        severity: Optional[str],
        rules: List[Dict[str, Any]],
        event_id: Optional[str],
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
            "surface": surface,
            "direction": direction,
            "action": action,
            "is_safe": is_safe,
            "severity": severity,
            "classifications": list(classifications) if classifications else [],
            "rule_violations": sorted(
                {
                    rule.get("rule_name")
                    for rule in rules
                    if isinstance(rule, dict)
                    and rule.get("rule_name")
                    and rule.get("classification") not in (None, "NONE_VIOLATION")
                }
            ),
            "event_id": event_id,
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

        if action == _ACTION_ALLOW:
            verbose_proxy_logger.info(line)
        else:
            verbose_proxy_logger.warning(line)

    def _warn_if_mode_surface_mismatch(self, event_hook: Any) -> None:
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
        response_obj: Any = None,
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
            surface="chat",
            start_time=start_time,
            direction=direction,
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
            response = await self.async_handler.post(
                url=url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
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
            config["enabled_rules"] = [
                self._normalize_rule(rule) for rule in self.enabled_rules
            ]
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
    def _normalize_rule(rule: Any) -> Dict[str, Any]:
        """Coerce a user-supplied rule into the wire-shape dict Cisco expects.

        Accepts three input shapes:

        * ``str`` — shorthand for ``{"rule_name": "<value>"}``.
        * ``dict`` — copied as-is, picking up known fields.
        * Pydantic ``CiscoAIDefenseRule`` (or any object exposing a
          ``model_dump`` method) — produced when ``enabled_rules``
          travels through the typed
          ``CiscoAIDefenseGuardrailConfigModelOptionalParams`` path
          (YAML config → ``LitellmParams`` validation →
          ``_get_optional_value``). Without this branch the helper
          raised ``ValueError`` for the Pydantic shape, and because the
          call site (``_build_chat_payload``) sits BEFORE the try/except
          in ``_inspect_chat``, that exception escaped uncaught and any
          user who configured ``enabled_rules`` in YAML got a 500 on
          every request. (Greptile P1 review on PR #28249.)
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
        surface: str,
        start_time: datetime,
        direction: str = "input",
        response_obj: Any = None,
    ) -> Dict[str, Any]:
        """Parse, log, and (optionally) raise/redact on the Cisco verdict.

        ``direction`` is ``"input"`` for request scans and ``"output"`` for
        response scans (used for metadata namespacing and response headers).
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
                surface=surface,
                direction=direction,
            )

        # Unwrap the JSON-RPC ``result`` envelope used by the MCP inspect
        # endpoint. The chat endpoint returns the verdict at the top
        # level and isn't wrapped, so this is a no-op there.
        verdict = self._unwrap_verdict_envelope(inspect_response)

        is_safe = verdict.get("is_safe")
        # OpenAPI spec lists `classification` as required (singular) but
        # examples & SDK return `classifications` (plural). Accept both.
        classifications = (
            verdict.get("classifications")
            or ([verdict["classification"]] if verdict.get("classification") else [])
            or []
        )
        severity = verdict.get("severity")
        rules = verdict.get("rules") or []
        explanation = verdict.get("explanation")
        event_id = verdict.get("event_id")
        # Did Cisco return sanitized content to rewrite the request/response
        # with? (Maps the reference plugin's `redact` action.)
        sanitized_text = self._extract_sanitized_text(verdict)
        sanitized_messages = self._extract_sanitized_messages(verdict)
        sanitized_mcp_arguments = self._extract_sanitized_mcp_arguments(verdict)
        sanitized_payload_present = bool(
            sanitized_text or sanitized_messages or sanitized_mcp_arguments
        )

        self._stash_verdict_on_request(
            request_data=request_data,
            surface=surface,
            direction=direction,
            is_safe=is_safe,
            classifications=classifications,
            severity=severity,
            rules=rules,
            event_id=event_id,
        )

        flagged = self._is_flagged(is_safe, classifications)

        # Decide the effective action. Explicit "action" wins; otherwise
        # infer from is_safe + presence of sanitized content. This matches
        # the reference plugin's `_decision_action` heuristic.
        action_raw = verdict.get("action")
        if isinstance(action_raw, str) and action_raw.strip():
            action = self._normalize_action(action_raw)
        elif flagged and sanitized_payload_present:
            action = _ACTION_REDACT
        elif flagged:
            action = _ACTION_BLOCK
        else:
            action = _ACTION_ALLOW

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Record structured guardrail info for spend logs, Datadog, Langfuse,
        # OTEL, and any other StandardLoggingPayload-aware logger. We pass
        # the full Cisco verdict so dashboards can group by surface,
        # severity, classifications, and rules.
        #
        # Pick the event type so observability tools can tell request scans
        # apart from response scans. We do NOT collapse every scan to
        # ``pre_*`` because dashboards built on ``standard_logging_payload``
        # rely on the event type to bucket input vs output violations.
        #
        # Mapping (matches the framework's existing ``pre_*`` / ``post_*``
        # / ``during_*`` conventions):
        #   chat  + input  -> pre_call
        #   chat  + output -> post_call
        #   mcp   + input  -> pre_mcp_call
        #   mcp   + output -> during_mcp_call  (there is no ``post_mcp_call``;
        #                     ``during_mcp_call`` is the framework's
        #                     designated MCP response-phase event.)
        if surface == "mcp":
            logging_event_type = (
                GuardrailEventHooks.during_mcp_call
                if direction == "output"
                else GuardrailEventHooks.pre_mcp_call
            )
        else:
            logging_event_type = (
                GuardrailEventHooks.post_call
                if direction == "output"
                else GuardrailEventHooks.pre_call
            )

        self.add_standard_logging_guardrail_information_to_request_data(
            guardrail_provider=self._PROVIDER_NAME,
            guardrail_json_response=self._sanitize_response_for_logging(
                inspect_response, surface=surface, action=action
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
            masked_entity_count=self._extract_masked_entity_count(rules),
            event_type=logging_event_type,
        )

        # Refresh stash with the resolved action so response headers reflect
        # block/redact/allow correctly.
        self._stash_verdict_on_request(
            request_data=request_data,
            surface=surface,
            direction=direction,
            is_safe=is_safe,
            classifications=classifications,
            severity=severity,
            rules=rules,
            event_id=event_id,
            action=action,
        )

        # Always emit one visible log line per scan so operators can see
        # the guardrail running without bumping the global proxy log
        # level to DEBUG. Allowed → INFO, intervened/redacted → WARNING.
        self._log_decision(
            surface=surface,
            direction=direction,
            action=action,
            is_safe=is_safe,
            classifications=classifications,
            severity=severity,
            rules=rules,
            event_id=event_id,
            duration_ms=duration * 1000,
            request_data=request_data,
        )

        if action == _ACTION_ALLOW:
            return inspect_response

        if action == _ACTION_REDACT:
            redacted = self._apply_redaction(
                request_data=request_data,
                response_obj=response_obj,
                surface=surface,
                direction=direction,
                sanitized_text=sanitized_text,
                sanitized_messages=sanitized_messages,
                sanitized_mcp_arguments=sanitized_mcp_arguments,
            )
            if redacted:
                verbose_proxy_logger.info(
                    "Cisco AI Defense guardrail (%s): redaction applied "
                    "(event_id=%s)",
                    surface,
                    event_id,
                )
                return inspect_response
            # Cisco asked for redact but we couldn't apply it — fall through
            # to the block decision instead of silently letting unsafe
            # content through.
            verbose_proxy_logger.warning(
                "Cisco AI Defense guardrail (%s): redact requested but no "
                "rewritable surface found — falling through to "
                "on_flagged_action=%s",
                surface,
                self.on_flagged_action,
            )

        if self.on_flagged_action == "block":
            raise HTTPException(
                status_code=400,
                detail=self._build_block_payload(
                    surface=surface,
                    direction=direction,
                    classifications=classifications,
                    severity=severity,
                    rules=rules,
                    explanation=explanation,
                    event_id=event_id,
                ),
            )

        verbose_proxy_logger.info(
            "Cisco AI Defense guardrail (%s): violation in monitor mode — "
            "request allowed to proceed (event_id=%s)",
            surface,
            event_id,
        )
        return inspect_response

    @staticmethod
    def _stash_verdict_on_request(
        request_data: dict,
        surface: str,
        direction: str,
        is_safe: Optional[bool],
        classifications: List[str],
        severity: Optional[str],
        rules: List[Dict[str, Any]],
        event_id: Optional[str],
        action: Optional[str] = None,
    ) -> None:
        """Surface the Cisco verdict on the request metadata for observability."""
        metadata_store = request_data.setdefault("metadata", {})
        if not isinstance(metadata_store, dict):
            return
        # Namespace by surface + direction (input/output) so chat & MCP plus
        # request- and response-side verdicts coexist on the same request.
        prefix = f"cisco_ai_defense_{surface}_{direction}"
        metadata_store[f"{prefix}_is_safe"] = is_safe
        if action:
            metadata_store[f"{prefix}_action"] = action
        if classifications:
            metadata_store[f"{prefix}_classifications"] = list(classifications)
        if severity:
            metadata_store[f"{prefix}_severity"] = severity
        if rules:
            metadata_store[f"{prefix}_rules"] = [
                rule.get("rule_name") for rule in rules if isinstance(rule, dict)
            ]
        if event_id:
            metadata_store[f"{prefix}_event_id"] = event_id

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

    # Keys whose presence indicates a dict actually contains a Cisco
    # decision/verdict (as opposed to being an envelope around it).
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
    def _has_decision_fields(cls, payload: Any) -> bool:
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

        # Double-nested case: ``result.{data,inspection,ai_defense,...}``
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
        return normalized

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
        response_obj: Any,
        surface: str,
        direction: str,
        sanitized_text: Optional[str],
        sanitized_messages: Optional[List[Dict[str, Any]]],
        sanitized_mcp_arguments: Optional[Dict[str, Any]],
    ) -> bool:
        """Apply a Cisco-supplied rewrite to the request/response in place.

        Returns True when a rewrite was applied; False when there was no
        suitable surface to rewrite (caller then falls back to
        ``on_flagged_action``).
        """
        if surface == "mcp" and direction == "input":
            return self._redact_mcp_input(
                request_data, sanitized_text, sanitized_mcp_arguments
            )
        if surface == "mcp" and direction == "output":
            if response_obj is None:
                return False
            if sanitized_text:
                return self._set_mcp_tool_response_text(response_obj, sanitized_text)
            return False
        if surface == "chat" and direction == "input":
            return self._redact_chat_input(
                request_data, sanitized_text, sanitized_messages
            )
        if surface == "chat" and direction == "output":
            return self._redact_chat_output(
                response_obj, sanitized_text, sanitized_messages
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
            params = request_data.get("params")
            if isinstance(params, dict):
                params["arguments"] = sanitized_mcp_arguments
            if isinstance(request_data.get("arguments"), dict):
                request_data["arguments"] = sanitized_mcp_arguments
            return True
        if sanitized_text:
            for args_path in (
                request_data.get("mcp_arguments"),
                request_data.get("arguments"),
                (request_data.get("params") or {}).get("arguments"),
            ):
                if isinstance(args_path, dict):
                    for key, value in list(args_path.items()):
                        if isinstance(value, str):
                            args_path[key] = sanitized_text
                            return True
        return False

    def _redact_chat_input(
        self,
        request_data: dict,
        sanitized_text: Optional[str],
        sanitized_messages: Optional[List[Dict[str, Any]]],
    ) -> bool:
        """Rewrite chat request input (``messages`` or ``input``)."""
        uses_input = "input" in request_data and "messages" not in request_data
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
            messages = request_data.get("messages")
            if isinstance(messages, list) and messages:
                for message in reversed(messages):
                    if (
                        isinstance(message, dict)
                        and message.get("role") == "user"
                        and isinstance(message.get("content"), str)
                    ):
                        message["content"] = sanitized_text
                        return True
        return False

    def _redact_chat_output(
        self,
        response_obj: Any,
        sanitized_text: Optional[str],
        sanitized_messages: Optional[List[Dict[str, Any]]],
    ) -> bool:
        """Rewrite chat response (``ModelResponse`` or ``ResponsesAPIResponse``)."""
        if response_obj is None:
            return False

        # Chat Completions shape.
        choices = getattr(response_obj, "choices", None)
        if isinstance(choices, list):
            return self._redact_model_response_choices(
                choices, sanitized_text, sanitized_messages
            )

        # Responses API shape.
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
        """Redact all choices. For n > 1 completions every choice was
        scanned together, so a redact verdict must apply to every choice
        — returning after the first leaks the rest (Codex P1).
        """
        if sanitized_messages:
            applied = False
            msg_iter = iter(sanitized_messages)
            for choice in choices:
                if not isinstance(choice, Choices):
                    continue
                replacement = next(msg_iter, None)
                if replacement is not None:
                    # Cisco may send ``content`` as a plain string OR as
                    # the OpenAI structured shape
                    # ``[{"type":"output_text","text":"..."}]``. Coerce
                    # via ``_normalize_message_content`` so structured
                    # payloads don't silently no-op (Codex P1).
                    text = CiscoAIDefenseGuardrail._normalize_message_content(
                        replacement.get("content")
                    )
                    if text:
                        choice.message.content = text
                        applied = True
                else:
                    # Cisco returned fewer replacements than choices.
                    # Scrub remaining choices to a safe default rather
                    # than leaving the original (unsafe) content alive.
                    fallback = sanitized_text or "[REDACTED]"
                    if getattr(choice.message, "content", None):
                        choice.message.content = fallback
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
                CiscoAIDefenseGuardrail._clear_tool_call_arguments(msg)
            return applied
        return False

    @staticmethod
    def _clear_arguments_field(obj: Any) -> None:
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
    def _clear_tool_call_arguments(cls, message: Any) -> None:
        """Scrub tool-call / function-call arguments after redact — the
        scanner included them in the text sent to Cisco, so they must
        not survive alongside the sanitized content.
        """
        for tc in getattr(message, "tool_calls", None) or []:
            fn = (
                tc.get("function")
                if isinstance(tc, dict)
                else getattr(tc, "function", None)
            )
            cls._clear_arguments_field(fn)
        cls._clear_arguments_field(getattr(message, "function_call", None))

    def _redact_responses_api_output(
        self,
        output_items: list,
        sanitized_text: Optional[str],
        sanitized_messages: Optional[List[Dict[str, Any]]],
    ) -> bool:
        replacement_text: Optional[str] = sanitized_text
        if not replacement_text and sanitized_messages:
            # Coerce content via ``_normalize_message_content`` so the
            # OpenAI structured shape ``[{"type":"output_text",...}]``
            # is handled alongside plain strings (Codex P1).
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
            # Clear function-call arguments on output items — same
            # reasoning as ``_clear_tool_call_arguments``.
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
        original_input: Any, sanitized_text: str
    ) -> Optional[Any]:
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

        # Flat list of content parts.
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

    @staticmethod
    def _is_flagged(is_safe: Optional[bool], classifications: List[str]) -> bool:
        if is_safe is False:
            return True
        if is_safe is True:
            return False
        return bool(classifications)

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
            # Pick the event type from (surface, direction) — same
            # mapping as ``_finalize_inspection``. Without this, output-
            # side API failures were recorded as pre_call/pre_mcp_call
            # and skewed dashboards (Codex P2).
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
                "original_error": str(error),
            },
        )

    # ------------------------------------------------------------------
    # Message extraction helpers
    # ------------------------------------------------------------------

    # Content-part ``type`` values that should be flattened to text by
    # ``_normalize_message_content``. Covers both Chat Completions
    # (``text``) and the Responses API (``input_text`` for caller-side
    # parts, ``output_text`` for assistant turns, ``summary_text`` for
    # reasoning summaries that may appear in conversation history).
    _TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text", "summary_text"})

    @staticmethod
    def _extract_inspect_messages_from_request(
        data: dict,
    ) -> List[Dict[str, str]]:
        """Build {role, content} messages for the Cisco AI Defense chat API."""
        messages: List[Dict[str, str]] = []

        raw_messages = data.get("messages") or []
        for message in raw_messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if not role:
                continue
            text = CiscoAIDefenseGuardrail._normalize_message_content(
                message.get("content")
            )
            if text:
                messages.append({"role": role, "content": text})

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

        return messages

    @staticmethod
    def _flatten_responses_input(input_value: Any) -> List[Dict[str, str]]:
        """Flatten the OpenAI Responses API ``input`` into chat-message form.

        Recognized shapes:

        1. Plain string -> one user message.
        2. List of message-shaped dicts
           ``{"role": "...", "content": [<content parts>]}`` -> one
           message per item, with the role preserved.
        3. Flat list of content-part dicts
           ``{"type": "input_text", "text": "..."}`` -> single user
           message containing the concatenated text.

        Without (2) and the ``input_text`` part type, structured
        Responses API requests slipped past the pre-call scan entirely
        (Veria AI High: "Responses API input bypass").
        """
        if input_value is None:
            return []
        if isinstance(input_value, str):
            return [{"role": "user", "content": input_value}]
        if not isinstance(input_value, list):
            text = str(input_value)
            return [{"role": "user", "content": text}] if text else []

        # Shape 2: list of message-shaped dicts (has ``role``).
        if any(isinstance(item, dict) and "role" in item for item in input_value):
            result: List[Dict[str, str]] = []
            for item in input_value:
                if not isinstance(item, dict):
                    continue
                role = item.get("role") or "user"
                text = CiscoAIDefenseGuardrail._normalize_message_content(
                    item.get("content")
                )
                if text:
                    result.append({"role": role, "content": text})
            return result

        # Shape 3: flat list of content parts.
        text = CiscoAIDefenseGuardrail._normalize_message_content(input_value)
        return [{"role": "user", "content": text}] if text else []

    @staticmethod
    def _normalize_message_content(content: Any) -> str:
        """Coerce OpenAI multi-modal content into a plain text string.

        Supports:

        * Plain string.
        * List of content-part dicts where ``type`` is one of
          ``text`` (Chat Completions), ``input_text`` / ``output_text`` /
          ``summary_text`` (Responses API).
        * List of message-shaped dicts with a nested ``content`` list —
          recurses into the nested content so a Responses API ``input``
          item like ``{"role":"user","content":[{"type":"input_text",...}]}``
          gets flattened correctly. (Without this branch the bypass
          surface Veria AI flagged was open.)
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
                # Recognised text content part.
                if part_type in CiscoAIDefenseGuardrail._TEXT_PART_TYPES and part.get(
                    "text"
                ):
                    parts.append(str(part["text"]))
                    continue
                # Nested message-shaped item (Responses API input).
                nested = part.get("content")
                if nested is not None:
                    nested_text = CiscoAIDefenseGuardrail._normalize_message_content(
                        nested
                    )
                    if nested_text:
                        parts.append(nested_text)
            return " ".join(parts)
        return str(content)

    @staticmethod
    def _extract_response_messages(response: Any) -> List[Dict[str, str]]:
        """Extract scannable assistant text from a chat response.

        Handles both ``ModelResponse`` (Chat Completions) and
        ``ResponsesAPIResponse`` (``/v1/responses``). On both shapes
        tool-call / function-call argument strings are included
        alongside the main text so a model can't bypass the scan by
        placing content there.
        """
        # Chat Completions: choices[*].message.{content, tool_calls, function_call}.
        if isinstance(response, ModelResponse):
            result: List[Dict[str, str]] = []
            for choice in getattr(response, "choices", None) or []:
                if not isinstance(choice, Choices):
                    continue
                parts: List[str] = []
                content = getattr(choice.message, "content", None)
                if content:
                    parts.append(str(content))
                for tc in getattr(choice.message, "tool_calls", None) or []:
                    args = CiscoAIDefenseGuardrail._extract_tool_call_arguments(tc)
                    if args:
                        parts.append(args)
                fc = getattr(choice.message, "function_call", None)
                if fc is not None:
                    args = CiscoAIDefenseGuardrail._extract_function_call_arguments(fc)
                    if args:
                        parts.append(args)
                if parts:
                    result.append({"role": "assistant", "content": " ".join(parts)})
            return result

        # Responses API: output[*].content[*].text and output[*].arguments
        # (and a generic .text fallback for other item types). Detected
        # via the ``output`` attribute rather than isinstance so the SDK
        # can swap response classes without breaking us.
        output_items = getattr(response, "output", None)
        if not isinstance(output_items, list):
            return []
        text_parts: List[str] = []
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
                t = pget("text")
                if isinstance(t, str) and t:
                    text_parts.append(t)
            args = get("arguments")
            if isinstance(args, str) and args:
                text_parts.append(args)
            direct = get("text")
            if isinstance(direct, str) and direct:
                text_parts.append(direct)
        joined = " ".join(text_parts)
        return [{"role": "assistant", "content": joined}] if joined else []

    @staticmethod
    def _extract_tool_call_arguments(tool_call: Any) -> Optional[str]:
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
    def _extract_function_call_arguments(function_call: Any) -> Optional[str]:
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
        # OpenAI emits ``arguments`` as a JSON string. Keep it as-is —
        # Cisco's text inspection works on the raw string regardless of
        # whether it parses as JSON, and round-tripping through
        # json.loads/dumps would lose data on malformed arguments.
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


# ``_serialize_mcp_content_item`` lives in ``cisco_ai_defense_mcp.py``
# alongside the other MCP-specific helpers (it's only invoked from
# ``_normalize_mcp_response`` which now also lives there).
