# +-------------------------------------------------------------+
#
#           Use AgentGuards Guardrails for your LLM calls
#                   https://agentguards.co/
#
# +-------------------------------------------------------------+

import os
from typing import (
    TYPE_CHECKING,
    Literal,
)

from fastapi import HTTPException

import litellm
from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails._content_utils import build_inspection_messages
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import (
    AnthropicMessagesResponse,
    LLMResponseTypes,
    ResponsesAPIResponse,
)

if TYPE_CHECKING:
    from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel

# AgentGuards decisions that must stop the request/response.
BLOCKING_INPUT_DECISIONS = {"block", "escalate"}
BLOCKING_OUTPUT_DECISIONS = {"reject", "escalate"}


class AgentGuardsGuardrailAPIError(Exception):
    """Raised when the AgentGuards API call fails and fail_closed is enabled."""


class AgentGuardsGuardrail(CustomGuardrail):
    """
    AgentGuards guardrail integration for LiteLLM.

    Screens prompts (input) via `/v1/guardrails/evaluate-input` and model
    responses (output) via `/v1/outputs/validate`, blocking jailbreaks,
    prompt injection, and data-exfiltration according to AgentGuards policy.
    """

    @classmethod
    def get_supported_event_hooks(cls) -> list[GuardrailEventHooks]:
        return [
            GuardrailEventHooks.pre_call,
            GuardrailEventHooks.post_call,
        ]

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        use_case: str | None = "check",
        tenant_id: str | None = None,
        fail_closed: bool | None = False,
        **kwargs: object,
    ) -> None:
        kwargs.setdefault("supported_event_hooks", list(self.get_supported_event_hooks()))
        self.async_handler = get_async_httpx_client(llm_provider=httpxSpecialProvider.GuardrailCallback)
        self.api_key = api_key or os.environ.get("AGENTGUARDS_API_KEY")
        self.api_base = (api_base or os.environ.get("AGENTGUARDS_API_BASE") or "https://prod.agentguards.co").rstrip(
            "/"
        )
        self.use_case = use_case or "check"
        self.tenant_id = tenant_id or os.environ.get("AGENTGUARDS_TENANT_ID")
        self.fail_closed = bool(fail_closed)

        verbose_proxy_logger.debug(
            "AgentGuards guardrail initialized: %s, api_base: %s, fail_closed: %s",
            kwargs.get("guardrail_name", "unknown"),
            self.api_base,
            self.fail_closed,
        )

        super().__init__(**kwargs)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        elif self.tenant_id:
            # Local / self-hosted dev AgentGuards accepts a tenant id instead of a key.
            headers["X-Tenant-ID"] = self.tenant_id
        return headers

    @staticmethod
    def _inspection_messages(data: dict) -> list[dict[str, str]]:
        """All inspectable text from the request — every message, multimodal
        text parts, and Responses-API ``input`` — flattened to plain-string
        ``{role, content}`` entries. Screening the whole conversation (not just
        the last turn) stops injections hidden in system / earlier / tool
        messages from bypassing the guardrail."""
        return build_inspection_messages(data)

    @staticmethod
    def _join_messages(messages: list[dict[str, str]]) -> str:
        return "\n\n".join(m["content"] for m in messages if m.get("content"))

    async def _post(self, path: str, payload: dict[str, object]) -> dict | None:
        """POST to AgentGuards. Returns the JSON body, or None when the call
        fails and fail_closed is disabled (fail-open)."""
        url = f"{self.api_base}{path}"
        try:
            response = await self.async_handler.post(
                url=url,
                headers=self._headers(),
                json=payload,
                timeout=10.0,
            )
            response.raise_for_status()
            return response.json()
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001 - intentional fail-open/closed on any backend error
            verbose_proxy_logger.error("AgentGuards API call to %s failed: %s", url, str(e))
            if self.fail_closed:
                raise AgentGuardsGuardrailAPIError(f"AgentGuards unreachable and fail_closed is enabled: {e!s}")
            return None

    @staticmethod
    def _block(decision: str, response: dict) -> None:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Blocked by AgentGuards",
                "decision": decision,
                "message": response.get("message"),
                "agentguards_response": response,
            },
        )

    # ------------------------------------------------------------------ #
    # Input: pre_call
    # ------------------------------------------------------------------ #
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
    ) -> Exception | str | dict | None:
        """Screen the prompt before it reaches the model."""
        if self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.pre_call) is not True:
            return data

        messages = self._inspection_messages(data)
        text = self._join_messages(messages)
        if not text:
            return data

        result = await self._post(
            "/v1/guardrails/evaluate-input",
            {
                "text": text,
                "use_case": self.use_case,
                "channel": "api",
                "metadata": {},
            },
        )
        if result is None:  # fail-open
            return data

        decision = result.get("decision")
        if decision in BLOCKING_INPUT_DECISIONS:
            self._block(decision, result)

        if decision == "redact" and result.get("redacted_text"):
            self._apply_redaction(data, result["redacted_text"], single_source=len(messages) == 1)

        return data

    @staticmethod
    def _apply_redaction(data: dict, redacted_text: str, single_source: bool) -> None:
        """Apply AgentGuards redaction in place, covering chat ``messages`` and
        Responses-API ``input``. A single redacted string can only be mapped back
        when the request has one inspectable text source; with multiple messages
        we skip in-place redaction (the request is still fully screened/blocked)
        rather than write one message's text over another."""
        if not single_source:
            verbose_proxy_logger.warning("AgentGuards redaction spans multiple messages; skipping in-place redaction.")
            return
        messages = data.get("messages") or []
        for message in reversed(messages):
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = redacted_text
                return
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        part["text"] = redacted_text
                        return
        # Responses-API string input
        if isinstance(data.get("input"), str):
            data["input"] = redacted_text

    # ------------------------------------------------------------------ #
    # Output: post_call
    # ------------------------------------------------------------------ #
    @log_guardrail_information
    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: LLMResponseTypes,
    ) -> LLMResponseTypes:
        """Validate the model's response after the LLM call."""
        if self.should_run_guardrail(data=data, event_type=GuardrailEventHooks.post_call) is not True:
            return response

        output_text = self._extract_response_text(response)
        if not output_text:
            return response

        result = await self._post(
            "/v1/outputs/validate",
            {
                "output_text": output_text,
                "context_text": self._join_messages(self._inspection_messages(data)),
                "metadata": {},
            },
        )
        if result is None:  # fail-open
            return response

        decision = result.get("decision")
        if decision in BLOCKING_OUTPUT_DECISIONS:
            self._block(decision, result)

        return response

    @staticmethod
    def _extract_response_text(response: LLMResponseTypes) -> str:
        """Extract all model-authored text from a response so exfiltration in
        tool-call arguments or non-chat response formats is validated too.
        Non-text responses (embeddings, images, files) yield an empty string
        and are passed through."""
        parts: list[str] = []
        if isinstance(response, litellm.ModelResponse):
            for choice in response.choices:
                # Text-completion responses carry the text on `choice.text`.
                choice_text = getattr(choice, "text", None)
                if isinstance(choice_text, str) and choice_text:
                    parts.append(choice_text)
                message = getattr(choice, "message", None)
                if message is None:
                    continue
                content = getattr(message, "content", None)
                if isinstance(content, str) and content:
                    parts.append(content)
                for call in getattr(message, "tool_calls", None) or []:
                    fn = call.get("function") if isinstance(call, dict) else getattr(call, "function", None)
                    if fn is None:
                        continue
                    name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
                    args = fn.get("arguments") if isinstance(fn, dict) else getattr(fn, "arguments", None)
                    if name:
                        parts.append(f"{name}({args or ''})")
        elif isinstance(response, ResponsesAPIResponse):
            output_text = getattr(response, "output_text", None)
            if isinstance(output_text, str) and output_text:
                parts.append(output_text)
        elif isinstance(response, AnthropicMessagesResponse):
            for block in getattr(response, "content", None) or []:
                block_text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
                if isinstance(block_text, str) and block_text:
                    parts.append(block_text)
        return "\n".join(parts)

    @staticmethod
    def get_config_model() -> type["GuardrailConfigModel"] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.agentguards import (
            AgentGuardsGuardrailConfigModel,
        )

        return AgentGuardsGuardrailConfigModel
