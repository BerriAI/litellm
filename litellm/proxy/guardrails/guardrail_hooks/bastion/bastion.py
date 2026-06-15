"""Bastion Prompt Protection guardrail for the LiteLLM proxy.

Local, ONNX-based prompt-injection / jailbreak detection (~5 ms warm on CPU; no
network calls). The detection engine ships in the optional
``bastion-prompt-protection`` package and is imported lazily, so litellm has no
hard dependency on it. Install it where you run the proxy::

    pip install bastion-prompt-protection

The free ``tiny`` model is AGPL-3.0; a commercial multilingual model (and an
AGPL exemption) is available at https://bastionsoft.com.
"""

import asyncio
import json
from typing import (
    TYPE_CHECKING,
    Literal,
    Optional,
    Union,
)

from fastapi import HTTPException

from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.utils import GenericGuardrailAPIInputs

if TYPE_CHECKING:
    from datetime import datetime

    from bastion_prompt_protection import Guard, GuardResult

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.mcp import MCPPostCallResponseObject


DEFAULT_VIOLATION_MESSAGE = (
    "I can't help with that request: it was flagged as a potential "
    "prompt-injection attempt and blocked."
)


def _function_texts(fn: object, fields: tuple[str, ...]) -> list[str]:
    """Pull screenable strings out of a tool / tool-call ``function`` payload."""
    out: list[str] = []
    for field in fields:
        value = fn.get(field) if isinstance(fn, dict) else getattr(fn, field, None)
        if isinstance(value, str):
            if value:
                out.append(value)
        elif value is not None:
            out.append(json.dumps(value, default=str))
    return out


def _collect_screenable_texts(
    inputs: GenericGuardrailAPIInputs, request_data: dict
) -> list[str]:
    """Everything to screen: message text **plus** tool-call arguments, tool
    definitions, and the legacy OpenAI ``functions`` field. Injection can hide in
    a tool-call's ``arguments``, a tool's ``description`` / ``parameters``, or a
    legacy function definition — not just message content — so those are
    serialized into the screened set rather than passing through unchecked.
    """
    texts: list[str] = [t for t in (inputs.get("texts") or []) if isinstance(t, str)]
    for tool in inputs.get("tools") or []:
        fn = (
            tool.get("function")
            if isinstance(tool, dict)
            else getattr(tool, "function", None)
        )
        if fn is not None:
            texts.extend(_function_texts(fn, ("name", "description", "parameters")))
    for tool_call in inputs.get("tool_calls") or []:
        call_fn = (
            tool_call.get("function")
            if isinstance(tool_call, dict)
            else getattr(tool_call, "function", None)
        )
        if call_fn is not None:
            texts.extend(_function_texts(call_fn, ("name", "arguments")))
    # Legacy OpenAI `functions` request field (deprecated but still forwarded).
    legacy_functions = (
        request_data.get("functions") if isinstance(request_data, dict) else None
    )
    for func in legacy_functions or []:
        texts.extend(_function_texts(func, ("name", "description", "parameters")))
    # MCP tool-call request fields (present when screening an MCP call pre-invocation).
    if isinstance(request_data, dict):
        for key in ("tool_name", "mcp_tool_name"):
            name = request_data.get(key)
            if isinstance(name, str) and name:
                texts.append(name)
        mcp_args = request_data.get("mcp_arguments")
        if mcp_args is None:
            mcp_args = request_data.get("arguments")
        if isinstance(mcp_args, str):
            if mcp_args:
                texts.append(mcp_args)
        elif mcp_args is not None:
            texts.append(json.dumps(mcp_args, default=str))
    return texts


def _mcp_content_list(response: object) -> list[object]:
    """Unwrap an MCP tool-call response to its list of content items.

    LiteLLM may hand us the content list directly, or a wrapped ``CallToolResult``
    whose text items live under an inner ``content`` field. The wrapper shows up as
    a dict (``{"content": [...]}`` / ``{"result": {"content": [...]}}``), an MCP SDK
    object (``.content`` / ``.result.content``), or a Pydantic-coerced
    ``[(field, value), ...]`` list. Dig the content list out of any of these so a
    poisoned text item is never silently skipped (the screening-bypass the bot
    flagged). Returns ``[]`` when no content list is found.
    """
    # Pydantic-coerced [(field, value), ...] -> treat as a dict wrapper.
    if (
        isinstance(response, list)
        and response
        and all(
            isinstance(it, tuple) and len(it) == 2 and isinstance(it[0], str)
            for it in response
        )
    ):
        response = dict(response)
    # Already a plain content list.
    if isinstance(response, list):
        return response
    # dict wrapper: {"content": [...]} or {"result": {"content": [...]}}.
    if isinstance(response, dict):
        content = response.get("content")
        if isinstance(content, list):
            return content
        result = response.get("result")
        if isinstance(result, dict) and isinstance(result.get("content"), list):
            return result["content"]
        return []
    # object wrapper (MCP SDK model / CallToolResult): .content or .result.content.
    content = getattr(response, "content", None)
    if isinstance(content, list):
        return content
    inner_content = getattr(getattr(response, "result", None), "content", None)
    if isinstance(inner_content, list):
        return inner_content
    return []


def _mcp_text_items(content: object) -> list[object]:
    """Return the text-bearing items from an MCP tool-call response list."""
    items: list[object] = []
    if not isinstance(content, list):
        return items
    for item in content:
        is_text = (
            item.get("type") == "text"
            if isinstance(item, dict)
            else getattr(item, "type", None) == "text"
        )
        if is_text:
            items.append(item)
    return items


def _mcp_item_text(item: object) -> str:
    """Read the text off an MCP text-content item (dict or object)."""
    text = item.get("text") if isinstance(item, dict) else getattr(item, "text", None)
    return text if isinstance(text, str) else ""


def _set_mcp_item_text(item: object, text: str) -> None:
    """Replace the text on an MCP text-content item (dict or object)."""
    if isinstance(item, dict):
        item["text"] = text
    else:
        try:
            setattr(item, "text", text)
        except (AttributeError, TypeError):
            pass


class BastionGuardrail(CustomGuardrail):
    """Screen text for prompt injection / jailbreak using Bastion Prompt Protection."""

    def __init__(
        self,
        guardrail_name: Optional[str] = None,
        preset: str = "tiny",
        threshold: Optional[float] = None,
        violation_message: str = DEFAULT_VIOLATION_MESSAGE,
        event_hook: Optional[Union[str, list[str]]] = None,
        default_on: bool = False,
    ) -> None:
        _event_hook: Optional[Union[GuardrailEventHooks, list[GuardrailEventHooks]]] = (
            None
        )
        if event_hook is not None:
            if isinstance(event_hook, list):
                _event_hook = [
                    GuardrailEventHooks(h) if isinstance(h, str) else h
                    for h in event_hook
                ]
            else:
                _event_hook = GuardrailEventHooks(event_hook)
        super().__init__(
            guardrail_name=guardrail_name or "bastion",
            supported_event_hooks=[
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
                GuardrailEventHooks.during_call,
                GuardrailEventHooks.pre_mcp_call,
                GuardrailEventHooks.during_mcp_call,
            ],
            event_hook=_event_hook or [GuardrailEventHooks.pre_call],
            default_on=default_on,
        )
        self.preset = preset
        self.threshold = threshold
        self.violation_message = violation_message
        self._guard: Optional["Guard"] = None  # lazily constructed (loads the model)

    def _get_guard(self) -> "Guard":
        if self._guard is None:
            try:
                from bastion_prompt_protection import Guard
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "The 'bastion-prompt-protection' package is required for the "
                    "Bastion guardrail. Install it with: "
                    "pip install bastion-prompt-protection"
                ) from e
            self._guard = Guard(preset=self.preset)
        return self._guard

    def _is_attack(self, result: "GuardResult") -> bool:
        if self.threshold is not None:
            return bool(result.risk >= self.threshold)
        return bool(result.is_attack)

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict,
        input_type: Literal["request", "response"],
        logging_obj: Optional["LiteLLMLoggingObj"] = None,
    ) -> GenericGuardrailAPIInputs:
        texts = _collect_screenable_texts(inputs, request_data)
        if not texts:
            return inputs

        guard = self._get_guard()
        for text in texts:
            if not text:
                continue
            # Bastion inference is synchronous and CPU-bound (~5 ms); offload to a
            # thread so it never blocks the proxy's event loop under concurrency.
            result = await asyncio.to_thread(guard.protect, text)
            if self._is_attack(result):
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": self.violation_message,
                        "bastion_guardrail": {
                            "risk": float(result.risk),
                            "stage": result.stage_reached,
                            "input_type": input_type,
                        },
                    },
                )
        return inputs

    async def async_post_mcp_tool_call_hook(
        self,
        kwargs: dict,
        response_obj: "MCPPostCallResponseObject",
        start_time: "datetime",
        end_time: "datetime",
    ) -> Optional["MCPPostCallResponseObject"]:
        """Screen an MCP tool *result* for indirect prompt injection.

        Tool results carry untrusted content (web pages, issues, documents) that
        can smuggle instructions back into the model — the MCP analog of indirect
        injection in RAG. On a flagged result the offending text is replaced with a
        refusal so the poisoned content never reaches the LLM.

        This runs on the proxy's logging path (where exceptions are swallowed), so
        blocking is done by returning a modified response, not by raising.
        """
        data = kwargs if isinstance(kwargs, dict) else {}
        # Only screen results when this guardrail is configured for MCP.
        if not (
            self.should_run_guardrail(
                data=data, event_type=GuardrailEventHooks.pre_mcp_call
            )
            or self.should_run_guardrail(
                data=data, event_type=GuardrailEventHooks.during_mcp_call
            )
        ):
            return None

        content = _mcp_content_list(getattr(response_obj, "mcp_tool_call_response", None))
        items = _mcp_text_items(content)
        if not items:
            return None

        guard = self._get_guard()
        flagged = False
        for item in items:
            text = _mcp_item_text(item)
            if not text:
                continue
            result = await asyncio.to_thread(guard.protect, text)
            if self._is_attack(result):
                flagged = True
                break

        if not flagged:
            return None

        # Block by replacing the result text with a refusal — the poisoned content
        # never reaches the model.
        for item in items:
            _set_mcp_item_text(item, self.violation_message)
        return response_obj
