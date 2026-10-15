import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import anthropic
import httpx
import openai
import pytest
from integration._support.client import Gateway, eventually, list_value, object_value, string_value
from integration._support.openai_wire import openai_error
from integration._support.wire import Request, wire_server
from integration.routing._reasoning_effort_levels_support import (
    AZURE_API_VERSION,
    CLAUDE,
    CLAUDE_LEVELS,
    ECHOED_KEY_MODEL,
    FIREWORKS_KIMI,
    GEMINI,
    GLM,
    GPT,
    GPT_LEVELS,
    KIMI,
    KIMI_LEVELS,
    OPENROUTER_KIMI,
    ZAI_GLM,
    answer,
    body_of,
    chat_messages,
    chat_request,
    deployment,
    marker_of,
    messages_request,
    new_marker,
    prompt,
    provider_reply,
    responses_request,
    sent_effort,
)
from pydantic import JsonValue

CHAT: Final = "/v1/chat/completions"
MESSAGES: Final = "/v1/messages"
RESPONSES: Final = "/v1/responses"
_BUDGET: Final[dict[str, JsonValue]] = {"type": "enabled", "budget_tokens": 2048}
_ADAPTIVE: Final[dict[str, JsonValue]] = {"type": "adaptive"}
_STRING_BUDGET: Final[dict[str, JsonValue]] = {"type": "enabled", "budget_tokens": "2048"}
_XHIGH_BUDGET: Final[dict[str, JsonValue]] = {"type": "enabled", "budget_tokens": 9000}
_BOOL_BUDGET: Final[dict[str, JsonValue]] = {"type": "enabled", "budget_tokens": True}
_NULL_BUDGET: Final[dict[str, JsonValue]] = {"type": "enabled", "budget_tokens": None}
_LIST_BUDGET: Final[dict[str, JsonValue]] = {"type": "enabled", "budget_tokens": [2048]}
_EMPTY_BUDGET: Final[dict[str, JsonValue]] = {"type": "enabled", "budget_tokens": ""}
_LONG_BUDGET: Final[dict[str, JsonValue]] = {"type": "enabled", "budget_tokens": "9" * 5000}
_MISSING_BUDGET: Final[dict[str, JsonValue]] = {"type": "enabled"}
_DISABLED: Final[dict[str, JsonValue]] = {"type": "disabled"}
_KIMI_INFO: Final[Mapping[str, JsonValue]] = MappingProxyType({"reasoning_effort_levels": list(KIMI_LEVELS)})
_CLAUDE_INFO: Final[Mapping[str, JsonValue]] = MappingProxyType({"reasoning_effort_levels": list(CLAUDE_LEVELS)})
_GPT_INFO: Final[Mapping[str, JsonValue]] = MappingProxyType({"reasoning_effort_levels": list(GPT_LEVELS)})
_UNDECLARED: Final[Mapping[str, JsonValue]] = MappingProxyType({})
_ECHOED_MAP_KEY: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"key": "azure/chat-latest", "litellm_provider": "azure"}
)
_LONG_LEVEL: Final = "x" * 5000
_JSON_SCHEMA_FORMAT: Final[dict[str, JsonValue]] = {
    "type": "json_schema",
    "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
}

Expected = Callable[[str], dict[str, JsonValue]]


def _openai_compatible(effort: JsonValue, *, model: str = KIMI, stream: bool = False) -> Expected:
    def expected(marker: str) -> dict[str, JsonValue]:
        usage: Final[dict[str, JsonValue]] = {"stream_options": {"include_usage": True}} if stream else {}
        sent: Final[dict[str, JsonValue]] = {
            "model": model,
            "messages": chat_messages(marker),
            "stream": stream,
            **usage,
        }
        return sent if effort is None else {**sent, "reasoning_effort": effort}

    return expected


def _openai_chat(effort: str) -> Expected:
    return lambda marker: {"model": GPT, "messages": chat_messages(marker), "reasoning_effort": effort}


def _azure_chat(effort: str) -> Expected:
    return lambda marker: {"model": GPT, "messages": chat_messages(marker), "reasoning_effort": effort, "stream": False}


def _anthropic_chat(effort: str, thinking: Mapping[str, JsonValue]) -> Expected:
    return lambda marker: {
        "model": CLAUDE,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt(marker)}]}],
        "thinking": dict(thinking),
        "output_config": {"effort": effort},
        "max_tokens": 128000,
    }


def _anthropic_chat_without_effort(thinking: Mapping[str, JsonValue]) -> Expected:
    return lambda marker: {
        "model": CLAUDE,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt(marker)}]}],
        "thinking": dict(thinking),
        "max_tokens": 128000,
    }


def _with_output_config(expected: Expected, output_config: JsonValue) -> Expected:
    return lambda marker: {**expected(marker), "output_config": output_config}


def _converse_chat(effort: str, thinking: Mapping[str, JsonValue]) -> Expected:
    return lambda marker: {
        "messages": [{"role": "user", "content": [{"text": prompt(marker)}]}],
        "inferenceConfig": {},
        "additionalModelRequestFields": {"thinking": dict(thinking), "output_config": {"effort": effort}},
        "additionalModelResponseFieldPaths": ["/usage/output_tokens_details"],
    }


def _native_messages(
    output_config: Mapping[str, JsonValue] | None, thinking: Mapping[str, JsonValue] | None
) -> Expected:
    def expected(marker: str) -> dict[str, JsonValue]:
        sent: Final[dict[str, JsonValue]] = {
            "messages": chat_messages(marker),
            "max_tokens": 4096,
            "model": CLAUDE,
            "stream": False,
        }
        with_thinking: Final = sent if thinking is None else {**sent, "thinking": dict(thinking)}
        return with_thinking if output_config is None else {**with_thinking, "output_config": dict(output_config)}

    return expected


def _invoke_messages(effort: str) -> Expected:
    return lambda marker: {
        "messages": chat_messages(marker),
        "max_tokens": 4096,
        "thinking": dict(_ADAPTIVE),
        "output_config": {"effort": effort},
        "anthropic_version": "bedrock-2023-05-31",
    }


def _bridged_messages(effort: str | None, *, model: str = KIMI) -> Expected:
    def expected(marker: str) -> dict[str, JsonValue]:
        sent: Final[dict[str, JsonValue]] = {"model": model, "messages": chat_messages(marker), "max_tokens": 4096}
        return sent if effort is None else {**sent, "reasoning_effort": effort}

    return expected


def _openrouter_messages(effort: str | None) -> Expected:
    def expected(marker: str) -> dict[str, JsonValue]:
        return {**_bridged_messages(effort, model=OPENROUTER_KIMI)(marker), "usage": {"include": True}}

    return expected


def _bridged_stream_messages(effort: str | None) -> Expected:
    def expected(marker: str) -> dict[str, JsonValue]:
        sent: Final[dict[str, JsonValue]] = {
            "model": KIMI,
            "messages": chat_messages(marker),
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": 4096,
        }
        return sent if effort is None else {**sent, "reasoning_effort": effort}

    return expected


def _azure_messages(effort: str | None) -> Expected:
    def expected(marker: str) -> dict[str, JsonValue]:
        sent: Final[dict[str, JsonValue]] = {
            "model": GPT,
            "messages": chat_messages(marker),
            "max_completion_tokens": 4096,
        }
        return sent if effort is None else {**sent, "reasoning_effort": effort}

    return expected


def _gemini_messages(level: str | None) -> Expected:
    def expected(marker: str) -> dict[str, JsonValue]:
        thinking: Final[dict[str, JsonValue]] = (
            {} if level is None else {"thinkingConfig": {"thinkingLevel": level, "includeThoughts": True}}
        )
        return {
            "contents": [{"role": "user", "parts": [{"text": prompt(marker)}]}],
            "generationConfig": {"max_output_tokens": 4096, **thinking},
        }

    return expected


def _responses_bridged_messages_without_effort() -> Expected:
    return lambda marker: {
        "model": GPT,
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt(marker)}]}],
        "include": ["reasoning.encrypted_content"],
        "max_output_tokens": 4096,
    }


def _responses_bridged_messages(effort: str) -> Expected:
    return lambda marker: {
        "model": GPT,
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt(marker)}]}],
        "include": ["reasoning.encrypted_content"],
        "max_output_tokens": 4096,
        "reasoning": {"effort": effort},
    }


def _converse_messages(effort: str | None) -> Expected:
    def expected(marker: str) -> dict[str, JsonValue]:
        effort_fields: Final[dict[str, JsonValue]] = {} if effort is None else {"output_config": {"effort": effort}}
        return {
            "messages": [{"role": "user", "content": [{"text": prompt(marker)}]}],
            "inferenceConfig": {"maxTokens": 4096},
            "additionalModelRequestFields": {"thinking": dict(_ADAPTIVE), **effort_fields},
            "additionalModelResponseFieldPaths": ["/usage/output_tokens_details"],
        }

    return expected


def _native_responses(reasoning: Mapping[str, JsonValue] | None, *, model: str = KIMI) -> Expected:
    def expected(marker: str) -> dict[str, JsonValue]:
        sent: Final[dict[str, JsonValue]] = {"model": model, "input": prompt(marker)}
        return sent if reasoning is None else {**sent, "reasoning": dict(reasoning)}

    return expected


def _bridged_responses(effort: str) -> Expected:
    return lambda marker: {"model": KIMI, "messages": chat_messages(marker), "reasoning_effort": effort}


@dataclass(frozen=True, slots=True)
class Cell:
    route: str
    kind: str
    model_info: Mapping[str, JsonValue]
    fields: Mapping[str, JsonValue]
    expected: Expected | None
    target: str
    status: int = 200
    error: str = ""
    stream: bool = False


_VLLM_CHAT: Final = "/v1/chat/completions"
_AZURE_AI_CHAT: Final = "/chat/completions"
_OPENAI_CHAT: Final = "/v1/chat/completions"
_AZURE_CHAT: Final = f"/openai/deployments/{GPT}/chat/completions?api-version={AZURE_API_VERSION}"
_ANTHROPIC: Final = "/v1/messages"
_CONVERSE: Final = f"/model/us.anthropic.{CLAUDE}/converse"
_CONVERSE_STREAM: Final = f"/model/us.anthropic.{CLAUDE}/converse-stream"
_INVOKE: Final = f"/model/us.anthropic.{CLAUDE}/invoke"
_AZURE_AI_ANTHROPIC: Final = "/anthropic/v1/messages"
_VLLM_RESPONSES: Final = "/v1/responses"
_OPENAI_RESPONSES: Final = "/v1/responses"
_AZURE_RESPONSES: Final = f"/openai/responses?api-version={AZURE_API_VERSION}"
_UNSUPPORTED: Final = "does not support parameters"
_GEMINI: Final = f"/models/{GEMINI}:generateContent"

CELLS: Final[Mapping[str, Cell]] = MappingProxyType(
    {
        "chat-azure_ai-medium-to-high": Cell(
            CHAT, "azure_ai", _KIMI_INFO, {"reasoning_effort": "medium"}, _openai_compatible("high"), _AZURE_AI_CHAT
        ),
        "chat-azure_ai-xhigh-to-max": Cell(
            CHAT, "azure_ai", _KIMI_INFO, {"reasoning_effort": "xhigh"}, _openai_compatible("max"), _AZURE_AI_CHAT
        ),
        "chat-azure_ai-stream-medium-to-high": Cell(
            CHAT,
            "azure_ai",
            _KIMI_INFO,
            {"reasoning_effort": "medium", "stream": True},
            _openai_compatible("high", stream=True),
            _AZURE_AI_CHAT,
            stream=True,
        ),
        "chat-azure_ai-budget-to-high": Cell(
            CHAT, "azure_ai", _KIMI_INFO, {"thinking": _BUDGET}, _openai_compatible("high"), _AZURE_AI_CHAT
        ),
        "chat-azure_ai-none-undeclared-passes": Cell(
            CHAT,
            "azure_ai",
            {"reasoning_effort_levels": ["low", "high"]},
            {"reasoning_effort": "none"},
            _openai_compatible("none"),
            _AZURE_AI_CHAT,
        ),
        "chat-azure_ai-undeclared-control-400": Cell(
            CHAT,
            "azure_ai",
            _UNDECLARED,
            {"reasoning_effort": "medium"},
            None,
            _AZURE_AI_CHAT,
            status=400,
            error=_UNSUPPORTED,
        ),
        "chat-azure_ai-empty-declaration-400": Cell(
            CHAT,
            "azure_ai",
            {"reasoning_effort_levels": []},
            {"reasoning_effort": "medium"},
            None,
            _AZURE_AI_CHAT,
            status=400,
            error=_UNSUPPORTED,
        ),
        "chat-azure_ai-int-effort-400": Cell(
            CHAT, "azure_ai", _KIMI_INFO, {"reasoning_effort": 5}, None, _AZURE_AI_CHAT, status=400, error=_UNSUPPORTED
        ),
        "chat-azure_ai-explicit-effort-wins-over-budget-thinking": Cell(
            CHAT,
            "azure_ai",
            _KIMI_INFO,
            {"reasoning_effort": "xhigh", "thinking": _BUDGET},
            _openai_compatible("max"),
            _AZURE_AI_CHAT,
        ),
        "chat-azure_ai-explicit-effort-wins-over-string-budget-thinking": Cell(
            CHAT,
            "azure_ai",
            _KIMI_INFO,
            {"reasoning_effort": "xhigh", "thinking": _STRING_BUDGET},
            _openai_compatible("max"),
            _AZURE_AI_CHAT,
        ),
        "chat-azure_ai-explicit-effort-wins-over-adaptive-thinking": Cell(
            CHAT,
            "azure_ai",
            _KIMI_INFO,
            {"reasoning_effort": "xhigh", "thinking": _ADAPTIVE},
            _openai_compatible("max"),
            _AZURE_AI_CHAT,
        ),
        "chat-azure_ai-stream-explicit-effort-wins-over-budget-thinking": Cell(
            CHAT,
            "azure_ai",
            _KIMI_INFO,
            {"reasoning_effort": "xhigh", "thinking": _BUDGET, "stream": True},
            _openai_compatible("max", stream=True),
            _AZURE_AI_CHAT,
            stream=True,
        ),
        "chat-azure_ai-undeclared-explicit-effort-beside-budget-400": Cell(
            CHAT,
            "azure_ai",
            _UNDECLARED,
            {"reasoning_effort": "xhigh", "thinking": _BUDGET},
            None,
            _AZURE_AI_CHAT,
            status=400,
            error=_UNSUPPORTED,
        ),
        "chat-vllm-medium-to-high": Cell(
            CHAT, "hosted_vllm", _KIMI_INFO, {"reasoning_effort": "medium"}, _openai_compatible("high"), _VLLM_CHAT
        ),
        "chat-vllm-minimal-to-low": Cell(
            CHAT, "hosted_vllm", _KIMI_INFO, {"reasoning_effort": "minimal"}, _openai_compatible("low"), _VLLM_CHAT
        ),
        "chat-vllm-stream-xhigh-to-max": Cell(
            CHAT,
            "hosted_vllm",
            _KIMI_INFO,
            {"reasoning_effort": "xhigh", "stream": True},
            _openai_compatible("max", stream=True),
            _VLLM_CHAT,
            stream=True,
        ),
        "chat-vllm-none-kept": Cell(
            CHAT, "hosted_vllm", _KIMI_INFO, {"reasoning_effort": "none"}, _openai_compatible("none"), _VLLM_CHAT
        ),
        "chat-vllm-default-sends-nothing": Cell(
            CHAT, "hosted_vllm", _KIMI_INFO, {"reasoning_effort": "default"}, _openai_compatible(None), _VLLM_CHAT
        ),
        "chat-vllm-unknown-level-passes": Cell(
            CHAT, "hosted_vllm", _KIMI_INFO, {"reasoning_effort": "turbo"}, _openai_compatible("turbo"), _VLLM_CHAT
        ),
        "chat-vllm-unknown-level-drop-params": Cell(
            CHAT,
            "hosted_vllm",
            _KIMI_INFO,
            {"reasoning_effort": "turbo", "drop_params": True},
            _openai_compatible(None),
            _VLLM_CHAT,
        ),
        "chat-vllm-undeclared-none-drop-params": Cell(
            CHAT,
            "hosted_vllm",
            {"reasoning_effort_levels": ["low", "high", "max"]},
            {"reasoning_effort": "none", "drop_params": True},
            _openai_compatible(None),
            _VLLM_CHAT,
        ),
        "chat-vllm-declared-none-drop-params": Cell(
            CHAT,
            "hosted_vllm",
            _KIMI_INFO,
            {"reasoning_effort": "none", "drop_params": True},
            _openai_compatible("none"),
            _VLLM_CHAT,
        ),
        "chat-vllm-budget-to-high": Cell(
            CHAT, "hosted_vllm", _KIMI_INFO, {"thinking": _BUDGET}, _openai_compatible("high"), _VLLM_CHAT
        ),
        "chat-vllm-budget-undeclared-control": Cell(
            CHAT, "hosted_vllm", _UNDECLARED, {"thinking": _BUDGET}, _openai_compatible("medium"), _VLLM_CHAT
        ),
        "chat-vllm-budget-only-none-drop-params": Cell(
            CHAT,
            "hosted_vllm",
            {"reasoning_effort_levels": ["none"]},
            {"thinking": _BUDGET, "drop_params": True},
            _openai_compatible(None),
            _VLLM_CHAT,
        ),
        "chat-vllm-float-budget-to-high": Cell(
            CHAT,
            "hosted_vllm",
            _KIMI_INFO,
            {"thinking": {"type": "enabled", "budget_tokens": 2048.5}},
            _openai_compatible("high"),
            _VLLM_CHAT,
        ),
        "chat-vllm-string-budget-400": Cell(
            CHAT,
            "hosted_vllm",
            _KIMI_INFO,
            {"thinking": {"type": "enabled", "budget_tokens": "2048"}},
            None,
            _VLLM_CHAT,
            status=400,
            error="budget_tokens",
        ),
        "chat-vllm-string-budget-drop-params": Cell(
            CHAT,
            "hosted_vllm",
            _KIMI_INFO,
            {"thinking": {"type": "enabled", "budget_tokens": "2048"}, "drop_params": True},
            _openai_compatible(None),
            _VLLM_CHAT,
        ),
        "chat-vllm-disabled-thinking": Cell(
            CHAT, "hosted_vllm", _KIMI_INFO, {"thinking": {"type": "disabled"}}, _openai_compatible(None), _VLLM_CHAT
        ),
        "chat-vllm-explicit-effort-beats-budget": Cell(
            CHAT,
            "hosted_vllm",
            _KIMI_INFO,
            {"reasoning_effort": "minimal", "thinking": _BUDGET},
            _openai_compatible("low"),
            _VLLM_CHAT,
        ),
        "chat-vllm-deployment-rounds-down": Cell(
            CHAT,
            "hosted_vllm",
            {**_KIMI_INFO, "reasoning_effort_rounding": "down"},
            {"reasoning_effort": "medium"},
            _openai_compatible("low"),
            _VLLM_CHAT,
        ),
        "chat-vllm-rounding-down-floor": Cell(
            CHAT,
            "hosted_vllm",
            {"reasoning_effort_levels": ["high", "max"], "reasoning_effort_rounding": "down"},
            {"reasoning_effort": "minimal"},
            _openai_compatible("high"),
            _VLLM_CHAT,
        ),
        "chat-vllm-rounding-up-ceiling": Cell(
            CHAT,
            "hosted_vllm",
            {"reasoning_effort_levels": ["minimal", "low"]},
            {"reasoning_effort": "max"},
            _openai_compatible("low"),
            _VLLM_CHAT,
        ),
        "chat-vllm-rounding-mixed-case": Cell(
            CHAT,
            "hosted_vllm",
            {**_KIMI_INFO, "reasoning_effort_rounding": " DOWN "},
            {"reasoning_effort": "medium"},
            _openai_compatible("low"),
            _VLLM_CHAT,
        ),
        "chat-vllm-rounding-unknown-falls-back-up": Cell(
            CHAT,
            "hosted_vllm",
            {**_KIMI_INFO, "reasoning_effort_rounding": "sideways"},
            {"reasoning_effort": "medium"},
            _openai_compatible("high"),
            _VLLM_CHAT,
        ),
        "chat-vllm-rounding-int-falls-back-up": Cell(
            CHAT,
            "hosted_vllm",
            {**_KIMI_INFO, "reasoning_effort_rounding": 5},
            {"reasoning_effort": "medium"},
            _openai_compatible("high"),
            _VLLM_CHAT,
        ),
        "chat-vllm-uppercase-levels-ignored": Cell(
            CHAT,
            "hosted_vllm",
            {"reasoning_effort_levels": ["LOW", "HIGH"]},
            {"reasoning_effort": "medium"},
            _openai_compatible("medium"),
            _VLLM_CHAT,
        ),
        "chat-vllm-string-levels-ignored": Cell(
            CHAT,
            "hosted_vllm",
            {"reasoning_effort_levels": "high"},
            {"reasoning_effort": "medium"},
            _openai_compatible("medium"),
            _VLLM_CHAT,
        ),
        "chat-vllm-null-levels-ignored": Cell(
            CHAT,
            "hosted_vllm",
            {"reasoning_effort_levels": None},
            {"reasoning_effort": "medium"},
            _openai_compatible("medium"),
            _VLLM_CHAT,
        ),
        "chat-vllm-mixed-junk-levels": Cell(
            CHAT,
            "hosted_vllm",
            {"reasoning_effort_levels": ["high", 7, None, "turbo"]},
            {"reasoning_effort": "low"},
            _openai_compatible("high"),
            _VLLM_CHAT,
        ),
        "chat-vllm-undeclared-control": Cell(
            CHAT, "hosted_vllm", _UNDECLARED, {"reasoning_effort": "medium"}, _openai_compatible("medium"), _VLLM_CHAT
        ),
        "chat-vllm-int-effort-passes": Cell(
            CHAT, "hosted_vllm", _KIMI_INFO, {"reasoning_effort": 5}, _openai_compatible(5), _VLLM_CHAT
        ),
        "chat-vllm-list-effort-passes": Cell(
            CHAT, "hosted_vllm", _KIMI_INFO, {"reasoning_effort": ["high"]}, _openai_compatible(["high"]), _VLLM_CHAT
        ),
        "chat-vllm-empty-effort-passes": Cell(
            CHAT, "hosted_vllm", _KIMI_INFO, {"reasoning_effort": ""}, _openai_compatible(""), _VLLM_CHAT
        ),
        "chat-vllm-empty-effort-drop-params": Cell(
            CHAT,
            "hosted_vllm",
            _KIMI_INFO,
            {"reasoning_effort": "", "drop_params": True},
            _openai_compatible(None),
            _VLLM_CHAT,
        ),
        "chat-vllm-5kb-effort-passes": Cell(
            CHAT,
            "hosted_vllm",
            _KIMI_INFO,
            {"reasoning_effort": _LONG_LEVEL},
            _openai_compatible(_LONG_LEVEL),
            _VLLM_CHAT,
        ),
        "chat-vllm-5kb-effort-drop-params": Cell(
            CHAT,
            "hosted_vllm",
            _KIMI_INFO,
            {"reasoning_effort": _LONG_LEVEL, "drop_params": True},
            _openai_compatible(None),
            _VLLM_CHAT,
        ),
        "chat-vllm-claude-name-undeclared-budget-unchanged": Cell(
            CHAT,
            "hosted_vllm_claude",
            _UNDECLARED,
            {"thinking": {"type": "enabled", "budget_tokens": 9000}},
            _openai_compatible("high", model=CLAUDE),
            _VLLM_CHAT,
        ),
        "messages-vllm-claude-name-undeclared-budget-unchanged": Cell(
            MESSAGES,
            "hosted_vllm_claude",
            _UNDECLARED,
            {"thinking": {"type": "enabled", "budget_tokens": 9000}},
            _bridged_messages("high", model=CLAUDE),
            _VLLM_CHAT,
        ),
        "chat-vllm-echoed-map-key-undeclared-unchanged": Cell(
            CHAT,
            "hosted_vllm_echoed_key",
            _ECHOED_MAP_KEY,
            {"reasoning_effort": "high"},
            _openai_compatible("high", model=ECHOED_KEY_MODEL),
            _VLLM_CHAT,
        ),
        "chat-vllm-undeclared-string-budget-400": Cell(
            CHAT,
            "hosted_vllm",
            _UNDECLARED,
            {"thinking": {"type": "enabled", "budget_tokens": "2048"}},
            None,
            _VLLM_CHAT,
            status=400,
            error="budget_tokens",
        ),
        "chat-mistral-declared-level-is-sent": Cell(
            CHAT,
            "mistral",
            {"reasoning_effort_levels": ["low", "medium"]},
            {"reasoning_effort": "medium"},
            lambda marker: {"model": GLM, "messages": chat_messages(marker), "reasoning_effort": "medium"},
            _OPENAI_CHAT,
        ),
        "chat-mistral-deployment-rounds-down": Cell(
            CHAT,
            "mistral",
            {"reasoning_effort_levels": ["low", "high"], "reasoning_effort_rounding": "down"},
            {"reasoning_effort": "medium"},
            lambda marker: {"model": GLM, "messages": chat_messages(marker), "reasoning_effort": "low"},
            _OPENAI_CHAT,
        ),
        "chat-openai-minimal-to-low": Cell(
            CHAT, "openai", _GPT_INFO, {"reasoning_effort": "minimal"}, _openai_chat("low"), _OPENAI_CHAT
        ),
        "chat-openai-max-to-xhigh": Cell(
            CHAT, "openai", _GPT_INFO, {"reasoning_effort": "max"}, _openai_chat("xhigh"), _OPENAI_CHAT
        ),
        "chat-openai-undeclared-minimal-control-400": Cell(
            CHAT,
            "openai",
            _UNDECLARED,
            {"reasoning_effort": "minimal"},
            None,
            _OPENAI_CHAT,
            status=400,
            error="reasoning_effort=minimal is not supported",
        ),
        "chat-azure-medium-to-high": Cell(
            CHAT,
            "azure",
            {"reasoning_effort_levels": ["low", "high"]},
            {"reasoning_effort": "medium"},
            _azure_chat("high"),
            _AZURE_CHAT,
        ),
        "chat-anthropic-medium-to-high": Cell(
            CHAT,
            "anthropic",
            _CLAUDE_INFO,
            {"reasoning_effort": "medium"},
            _anthropic_chat("high", {"type": "adaptive", "display": "summarized"}),
            _ANTHROPIC,
        ),
        "chat-anthropic-budget-to-high": Cell(
            CHAT, "anthropic", _CLAUDE_INFO, {"thinking": _BUDGET}, _anthropic_chat("high", _ADAPTIVE), _ANTHROPIC
        ),
        "chat-anthropic-budget-keeps-display": Cell(
            CHAT,
            "anthropic",
            _CLAUDE_INFO,
            {"thinking": {**_BUDGET, "display": "omitted"}},
            _anthropic_chat("high", {"type": "adaptive", "display": "omitted"}),
            _ANTHROPIC,
        ),
        "chat-anthropic-undeclared-budget-control": Cell(
            CHAT, "anthropic", _UNDECLARED, {"thinking": _BUDGET}, _anthropic_chat("medium", _ADAPTIVE), _ANTHROPIC
        ),
        "chat-converse-medium-to-high": Cell(
            CHAT,
            "bedrock_converse",
            _CLAUDE_INFO,
            {"reasoning_effort": "medium"},
            _converse_chat("high", {"type": "adaptive", "display": "summarized"}),
            _CONVERSE,
        ),
        "chat-converse-budget-to-high": Cell(
            CHAT, "bedrock_converse", _CLAUDE_INFO, {"thinking": _BUDGET}, _converse_chat("high", _ADAPTIVE), _CONVERSE
        ),
        "chat-converse-undeclared-control": Cell(
            CHAT,
            "bedrock_converse",
            _UNDECLARED,
            {"reasoning_effort": "medium"},
            _converse_chat("medium", {"type": "adaptive", "display": "summarized"}),
            _CONVERSE,
        ),
        "messages-anthropic-claude-code-medium-to-high": Cell(
            MESSAGES,
            "anthropic",
            _CLAUDE_INFO,
            {"thinking": _ADAPTIVE, "output_config": {"effort": "medium"}},
            _native_messages({"effort": "high"}, _ADAPTIVE),
            _ANTHROPIC,
        ),
        "messages-anthropic-budget-to-high": Cell(
            MESSAGES,
            "anthropic",
            _CLAUDE_INFO,
            {"thinking": _BUDGET},
            _native_messages({"effort": "high"}, _ADAPTIVE),
            _ANTHROPIC,
        ),
        "messages-anthropic-budget-keeps-display": Cell(
            MESSAGES,
            "anthropic",
            _CLAUDE_INFO,
            {"thinking": {**_BUDGET, "display": "omitted"}},
            _native_messages({"effort": "high"}, {"type": "adaptive", "display": "omitted"}),
            _ANTHROPIC,
        ),
        "messages-anthropic-output-effort-only": Cell(
            MESSAGES,
            "anthropic",
            _CLAUDE_INFO,
            {"output_config": {"effort": "medium"}},
            _native_messages({"effort": "high"}, None),
            _ANTHROPIC,
        ),
        "messages-anthropic-drop-params-keeps-format": Cell(
            MESSAGES,
            "anthropic",
            _CLAUDE_INFO,
            {"output_config": {"effort": "turbo", "format": _JSON_SCHEMA_FORMAT}, "drop_params": True},
            _native_messages({"format": _JSON_SCHEMA_FORMAT}, None),
            _ANTHROPIC,
        ),
        "messages-anthropic-default-sends-no-effort": Cell(
            MESSAGES,
            "anthropic",
            _CLAUDE_INFO,
            {"thinking": _ADAPTIVE, "output_config": {"effort": "default"}},
            _native_messages(None, _ADAPTIVE),
            _ANTHROPIC,
        ),
        "messages-anthropic-undeclared-control": Cell(
            MESSAGES,
            "anthropic",
            _UNDECLARED,
            {"thinking": _ADAPTIVE, "output_config": {"effort": "medium"}},
            _native_messages({"effort": "medium"}, _ADAPTIVE),
            _ANTHROPIC,
        ),
        "messages-azure_ai-claude-code-medium-to-high": Cell(
            MESSAGES,
            "azure_ai",
            _KIMI_INFO,
            {"thinking": _ADAPTIVE, "output_config": {"effort": "medium"}},
            _bridged_messages("high"),
            _AZURE_AI_CHAT,
        ),
        "messages-azure_ai-budget-to-high": Cell(
            MESSAGES, "azure_ai", _KIMI_INFO, {"thinking": _BUDGET}, _bridged_messages("high"), _AZURE_AI_CHAT
        ),
        "messages-vllm-claude-code-xhigh-to-max": Cell(
            MESSAGES,
            "hosted_vllm",
            _KIMI_INFO,
            {"thinking": _ADAPTIVE, "output_config": {"effort": "xhigh"}},
            _bridged_messages("max"),
            _VLLM_CHAT,
        ),
        "messages-vllm-budget-to-high": Cell(
            MESSAGES, "hosted_vllm", _KIMI_INFO, {"thinking": _BUDGET}, _bridged_messages("high"), _VLLM_CHAT
        ),
        "messages-vllm-undeclared-budget-control": Cell(
            MESSAGES, "hosted_vllm", _UNDECLARED, {"thinking": _BUDGET}, _bridged_messages("medium"), _VLLM_CHAT
        ),
        "messages-openai-claude-code-medium-to-high": Cell(
            MESSAGES,
            "openai",
            {"reasoning_effort_levels": ["low", "high"]},
            {"thinking": _ADAPTIVE, "output_config": {"effort": "medium"}},
            _responses_bridged_messages("high"),
            _OPENAI_RESPONSES,
        ),
        "messages-converse-budget-to-high": Cell(
            MESSAGES, "bedrock_converse", _CLAUDE_INFO, {"thinking": _BUDGET}, _converse_messages("high"), _CONVERSE
        ),
        "messages-converse-claude-code-medium-to-high": Cell(
            MESSAGES,
            "bedrock_converse",
            _CLAUDE_INFO,
            {"thinking": _ADAPTIVE, "output_config": {"effort": "medium"}},
            _converse_messages("high"),
            _CONVERSE,
        ),
        "messages-invoke-claude-code-medium-to-high": Cell(
            MESSAGES,
            "bedrock_invoke",
            _CLAUDE_INFO,
            {"thinking": _ADAPTIVE, "output_config": {"effort": "medium"}},
            _invoke_messages("high"),
            _INVOKE,
        ),
        "messages-azure_ai-claude-claude-code-medium-to-high": Cell(
            MESSAGES,
            "azure_ai_claude",
            _CLAUDE_INFO,
            {"thinking": _ADAPTIVE, "output_config": {"effort": "medium"}},
            _native_messages({"effort": "high"}, _ADAPTIVE),
            _AZURE_AI_ANTHROPIC,
        ),
        "responses-vllm-minimal-to-low": Cell(
            RESPONSES,
            "hosted_vllm",
            _KIMI_INFO,
            {"reasoning": {"effort": "minimal"}},
            _native_responses({"effort": "low"}),
            _VLLM_RESPONSES,
        ),
        "responses-vllm-keeps-summary": Cell(
            RESPONSES,
            "hosted_vllm",
            _KIMI_INFO,
            {"reasoning": {"effort": "medium", "summary": "auto"}},
            _native_responses({"effort": "high", "summary": "auto"}),
            _VLLM_RESPONSES,
        ),
        "responses-vllm-default-keeps-summary": Cell(
            RESPONSES,
            "hosted_vllm",
            _KIMI_INFO,
            {"reasoning": {"effort": "default", "summary": "auto"}},
            _native_responses({"summary": "auto"}),
            _VLLM_RESPONSES,
        ),
        "responses-vllm-default-alone-sends-nothing": Cell(
            RESPONSES,
            "hosted_vllm",
            _KIMI_INFO,
            {"reasoning": {"effort": "default"}},
            _native_responses(None),
            _VLLM_RESPONSES,
        ),
        "responses-vllm-unknown-drop-params": Cell(
            RESPONSES,
            "hosted_vllm",
            _KIMI_INFO,
            {"reasoning": {"effort": "turbo"}, "drop_params": True},
            _native_responses(None),
            _VLLM_RESPONSES,
        ),
        "responses-vllm-stream-medium-to-high": Cell(
            RESPONSES,
            "hosted_vllm",
            _KIMI_INFO,
            {"reasoning": {"effort": "medium"}, "stream": True},
            lambda marker: {"model": KIMI, "input": prompt(marker), "reasoning": {"effort": "high"}, "stream": True},
            _VLLM_RESPONSES,
            stream=True,
        ),
        "responses-vllm-deployment-rounds-down": Cell(
            RESPONSES,
            "hosted_vllm",
            {**_KIMI_INFO, "reasoning_effort_rounding": "down"},
            {"reasoning": {"effort": "medium"}},
            _native_responses({"effort": "low"}),
            _VLLM_RESPONSES,
        ),
        "responses-vllm-undeclared-control": Cell(
            RESPONSES,
            "hosted_vllm",
            _UNDECLARED,
            {"reasoning": {"effort": "medium"}},
            _native_responses({"effort": "medium"}),
            _VLLM_RESPONSES,
        ),
        "responses-azure_ai-minimal-to-low": Cell(
            RESPONSES,
            "azure_ai",
            _KIMI_INFO,
            {"reasoning": {"effort": "minimal"}},
            _bridged_responses("low"),
            _AZURE_AI_CHAT,
        ),
        "responses-openai-minimal-to-low": Cell(
            RESPONSES,
            "openai",
            {"reasoning_effort_levels": ["low", "high"]},
            {"reasoning": {"effort": "minimal", "summary": "auto"}},
            _native_responses({"effort": "low", "summary": "auto"}, model=GPT),
            _OPENAI_RESPONSES,
        ),
        "responses-azure-medium-to-high": Cell(
            RESPONSES,
            "azure",
            {"reasoning_effort_levels": ["low", "high"]},
            {"reasoning": {"effort": "medium"}},
            _native_responses({"effort": "high"}, model=GPT),
            _AZURE_RESPONSES,
        ),
        "chat-anthropic-output-config-medium-to-high": Cell(
            CHAT,
            "anthropic",
            _CLAUDE_INFO,
            {"thinking": _ADAPTIVE, "output_config": {"effort": "medium"}},
            _anthropic_chat("high", _ADAPTIVE),
            _ANTHROPIC,
        ),
        "chat-anthropic-output-config-undeclared-control": Cell(
            CHAT,
            "anthropic",
            _UNDECLARED,
            {"thinking": _ADAPTIVE, "output_config": {"effort": "medium"}},
            _anthropic_chat("medium", _ADAPTIVE),
            _ANTHROPIC,
        ),
        "chat-anthropic-output-config-default-sends-no-effort": Cell(
            CHAT,
            "anthropic",
            _CLAUDE_INFO,
            {"thinking": _ADAPTIVE, "output_config": {"effort": "default"}},
            _anthropic_chat_without_effort(_ADAPTIVE),
            _ANTHROPIC,
        ),
        "chat-anthropic-output-config-unknown-drop-params": Cell(
            CHAT,
            "anthropic",
            _CLAUDE_INFO,
            {"thinking": _ADAPTIVE, "output_config": {"effort": "turbo"}, "drop_params": True},
            _anthropic_chat_without_effort(_ADAPTIVE),
            _ANTHROPIC,
        ),
        "chat-converse-output-config-medium-to-high": Cell(
            CHAT,
            "bedrock_converse",
            _CLAUDE_INFO,
            {"thinking": _ADAPTIVE, "output_config": {"effort": "medium"}},
            _converse_chat("high", _ADAPTIVE),
            _CONVERSE,
        ),
        "chat-converse-output-config-undeclared-control": Cell(
            CHAT,
            "bedrock_converse",
            _UNDECLARED,
            {"thinking": _ADAPTIVE, "output_config": {"effort": "medium"}},
            _converse_chat("medium", _ADAPTIVE),
            _CONVERSE,
        ),
        "chat-vllm-output-config-medium-to-high": Cell(
            CHAT,
            "hosted_vllm",
            _KIMI_INFO,
            {"output_config": {"effort": "medium"}},
            _with_output_config(_openai_compatible(None), {"effort": "high"}),
            _VLLM_CHAT,
        ),
        "chat-vllm-output-config-keeps-other-keys": Cell(
            CHAT,
            "hosted_vllm",
            _KIMI_INFO,
            {"output_config": {"effort": "minimal", "format": _JSON_SCHEMA_FORMAT}},
            _with_output_config(_openai_compatible(None), {"effort": "low", "format": _JSON_SCHEMA_FORMAT}),
            _VLLM_CHAT,
        ),
        "chat-vllm-output-config-string-passes": Cell(
            CHAT,
            "hosted_vllm",
            _KIMI_INFO,
            {"output_config": "medium"},
            _with_output_config(_openai_compatible(None), "medium"),
            _VLLM_CHAT,
        ),
        "chat-vllm-output-config-int-effort-passes": Cell(
            CHAT,
            "hosted_vllm",
            _KIMI_INFO,
            {"output_config": {"effort": 5}},
            _with_output_config(_openai_compatible(None), {"effort": 5}),
            _VLLM_CHAT,
        ),
        "chat-vllm-output-config-5kb-effort-drop-params": Cell(
            CHAT,
            "hosted_vllm",
            _KIMI_INFO,
            {"output_config": {"effort": _LONG_LEVEL}, "drop_params": True},
            _openai_compatible(None),
            _VLLM_CHAT,
        ),
        "chat-vllm-output-config-undeclared-control": Cell(
            CHAT,
            "hosted_vllm",
            _UNDECLARED,
            {"output_config": {"effort": "medium"}},
            _with_output_config(_openai_compatible(None), {"effort": "medium"}),
            _VLLM_CHAT,
        ),
        "chat-openai-output-config-minimal-to-low": Cell(
            CHAT,
            "openai",
            _GPT_INFO,
            {"output_config": {"effort": "minimal"}},
            lambda marker: {"model": GPT, "messages": chat_messages(marker), "output_config": {"effort": "low"}},
            _OPENAI_CHAT,
        ),
        "chat-openai-output-config-undeclared-control": Cell(
            CHAT,
            "openai",
            _UNDECLARED,
            {"output_config": {"effort": "minimal"}},
            lambda marker: {"model": GPT, "messages": chat_messages(marker), "output_config": {"effort": "minimal"}},
            _OPENAI_CHAT,
        ),
        "chat-azure-output-config-medium-to-high": Cell(
            CHAT,
            "azure",
            {"reasoning_effort_levels": ["low", "high"]},
            {"output_config": {"effort": "medium"}},
            lambda marker: {
                "model": GPT,
                "messages": chat_messages(marker),
                "stream": False,
                "output_config": {"effort": "high"},
            },
            _AZURE_CHAT,
        ),
        "chat-azure_ai-output-config-medium-to-high": Cell(
            CHAT,
            "azure_ai",
            _KIMI_INFO,
            {"output_config": {"effort": "medium"}},
            _with_output_config(_openai_compatible(None), {"effort": "high"}),
            _AZURE_AI_CHAT,
        ),
        "messages-converse-claude-code-undeclared-control": Cell(
            MESSAGES,
            "bedrock_converse",
            _UNDECLARED,
            {"thinking": _ADAPTIVE, "output_config": {"effort": "medium"}},
            _converse_messages("medium"),
            _CONVERSE,
        ),
        "messages-converse-claude-code-unknown-drop-params": Cell(
            MESSAGES,
            "bedrock_converse",
            _CLAUDE_INFO,
            {"thinking": _ADAPTIVE, "output_config": {"effort": "turbo"}, "drop_params": True},
            _converse_messages(None),
            _CONVERSE,
        ),
        "messages-converse-claude-code-default-sends-no-effort": Cell(
            MESSAGES,
            "bedrock_converse",
            _CLAUDE_INFO,
            {"thinking": _ADAPTIVE, "output_config": {"effort": "default"}},
            _converse_messages(None),
            _CONVERSE,
        ),
        "messages-converse-claude-code-rounds-down": Cell(
            MESSAGES,
            "bedrock_converse",
            {**_CLAUDE_INFO, "reasoning_effort_rounding": "down"},
            {"thinking": _ADAPTIVE, "output_config": {"effort": "medium"}},
            _converse_messages("low"),
            _CONVERSE,
        ),
        "messages-converse-claude-code-stream-medium-to-high": Cell(
            MESSAGES,
            "bedrock_converse",
            _CLAUDE_INFO,
            {"thinking": _ADAPTIVE, "output_config": {"effort": "medium"}, "stream": True},
            _converse_messages("high"),
            _CONVERSE_STREAM,
            stream=True,
        ),
        "chat-vllm-bool-budget-400": Cell(
            CHAT,
            "hosted_vllm",
            _KIMI_INFO,
            {"thinking": {"type": "enabled", "budget_tokens": True}},
            None,
            _VLLM_CHAT,
            status=400,
            error="budget_tokens",
        ),
        "chat-vllm-null-budget-400": Cell(
            CHAT,
            "hosted_vllm",
            _KIMI_INFO,
            {"thinking": {"type": "enabled", "budget_tokens": None}},
            None,
            _VLLM_CHAT,
            status=400,
            error="budget_tokens",
        ),
        "chat-vllm-list-budget-400": Cell(
            CHAT,
            "hosted_vllm",
            _KIMI_INFO,
            {"thinking": {"type": "enabled", "budget_tokens": [2048]}},
            None,
            _VLLM_CHAT,
            status=400,
            error="budget_tokens",
        ),
        "chat-vllm-missing-budget-to-low": Cell(
            CHAT, "hosted_vllm", _KIMI_INFO, {"thinking": {"type": "enabled"}}, _openai_compatible("low"), _VLLM_CHAT
        ),
        "chat-vllm-explicit-effort-wins-over-string-budget": Cell(
            CHAT,
            "hosted_vllm",
            _KIMI_INFO,
            {"reasoning_effort": "medium", "thinking": _STRING_BUDGET},
            _openai_compatible("high"),
            _VLLM_CHAT,
        ),
        "chat-vllm-explicit-effort-wins-over-bool-budget": Cell(
            CHAT,
            "hosted_vllm",
            _KIMI_INFO,
            {"reasoning_effort": "medium", "thinking": {"type": "enabled", "budget_tokens": True}},
            _openai_compatible("high"),
            _VLLM_CHAT,
        ),
        "chat-vllm-explicit-effort-beside-string-budget-drop-params": Cell(
            CHAT,
            "hosted_vllm_drop_params",
            _KIMI_INFO,
            {"reasoning_effort": "medium", "thinking": _STRING_BUDGET},
            _openai_compatible("high"),
            _VLLM_CHAT,
        ),
        "chat-vllm-undeclared-explicit-effort-wins-over-string-budget": Cell(
            CHAT,
            "hosted_vllm",
            _UNDECLARED,
            {"reasoning_effort": "medium", "thinking": _STRING_BUDGET},
            _openai_compatible("medium"),
            _VLLM_CHAT,
        ),
        "chat-vllm-string-budget-deployment-drop-params": Cell(
            CHAT,
            "hosted_vllm_drop_params",
            _KIMI_INFO,
            {"thinking": _STRING_BUDGET},
            _openai_compatible(None),
            _VLLM_CHAT,
        ),
        "chat-vllm-undeclared-string-budget-drop-params": Cell(
            CHAT,
            "hosted_vllm",
            _UNDECLARED,
            {"thinking": _STRING_BUDGET, "drop_params": True},
            _openai_compatible(None),
            _VLLM_CHAT,
        ),
        "chat-vllm-empty-levels-budget-unchanged": Cell(
            CHAT,
            "hosted_vllm",
            {"reasoning_effort_levels": []},
            {"thinking": _BUDGET},
            _openai_compatible("medium"),
            _VLLM_CHAT,
        ),
        "chat-azure_ai-string-budget-400": Cell(
            CHAT,
            "azure_ai",
            _KIMI_INFO,
            {"thinking": _STRING_BUDGET},
            None,
            _AZURE_AI_CHAT,
            status=400,
            error="budget_tokens",
        ),
        "chat-anthropic-numeric-string-budget-to-high": Cell(
            CHAT,
            "anthropic",
            _CLAUDE_INFO,
            {"thinking": _STRING_BUDGET},
            _anthropic_chat("high", _ADAPTIVE),
            _ANTHROPIC,
        ),
        "messages-vllm-string-budget-400": Cell(
            MESSAGES,
            "hosted_vllm",
            _KIMI_INFO,
            {"thinking": _STRING_BUDGET},
            None,
            _VLLM_CHAT,
            status=400,
            error="budget_tokens",
        ),
        "chat-vllm-echoed-map-key-own-levels-win": Cell(
            CHAT,
            "hosted_vllm_echoed_key",
            {**_ECHOED_MAP_KEY, "reasoning_effort_levels": ["low", "high"]},
            {"reasoning_effort": "medium"},
            _openai_compatible("high", model=ECHOED_KEY_MODEL),
            _VLLM_CHAT,
        ),
        "responses-vllm-echoed-map-key-undeclared-unchanged": Cell(
            RESPONSES,
            "hosted_vllm_echoed_key",
            _ECHOED_MAP_KEY,
            {"reasoning": {"effort": "high"}},
            _native_responses({"effort": "high"}, model=ECHOED_KEY_MODEL),
            _VLLM_RESPONSES,
        ),
        "chat-vllm-claude-name-declared-budget-generic-table": Cell(
            CHAT,
            "hosted_vllm_claude",
            {"reasoning_effort_levels": ["low", "high", "max"]},
            {"thinking": _XHIGH_BUDGET},
            _openai_compatible("high", model=CLAUDE),
            _VLLM_CHAT,
        ),
        "messages-vllm-claude-name-declared-budget-generic-table": Cell(
            MESSAGES,
            "hosted_vllm_claude",
            {"reasoning_effort_levels": ["low", "high", "max"]},
            {"thinking": _XHIGH_BUDGET},
            _bridged_messages("high", model=CLAUDE),
            _VLLM_CHAT,
        ),
        "messages-azure_ai-string-budget-400": Cell(
            MESSAGES, "azure_ai", _KIMI_INFO, {"thinking": _STRING_BUDGET}, None, _AZURE_AI_CHAT, 400, "thinking"
        ),
        "messages-azure_ai-undeclared-string-budget-400": Cell(
            MESSAGES, "azure_ai", _UNDECLARED, {"thinking": _STRING_BUDGET}, None, _AZURE_AI_CHAT, 400, "thinking"
        ),
        "messages-azure_ai-string-budget-drop-params": Cell(
            MESSAGES,
            "azure_ai",
            _KIMI_INFO,
            {"thinking": _STRING_BUDGET, "drop_params": True},
            _bridged_messages(None),
            _AZURE_AI_CHAT,
        ),
        "messages-azure_ai-undeclared-string-budget-drop-params": Cell(
            MESSAGES,
            "azure_ai",
            _UNDECLARED,
            {"thinking": _STRING_BUDGET, "drop_params": True},
            _bridged_messages(None),
            _AZURE_AI_CHAT,
        ),
        "messages-azure_ai-string-budget-deployment-drop-params": Cell(
            MESSAGES,
            "azure_ai_drop_params",
            _KIMI_INFO,
            {"thinking": _STRING_BUDGET},
            _bridged_messages(None),
            _AZURE_AI_CHAT,
        ),
        "messages-azure_ai-undeclared-string-budget-deployment-drop-params": Cell(
            MESSAGES,
            "azure_ai_drop_params",
            _UNDECLARED,
            {"thinking": _STRING_BUDGET},
            _bridged_messages(None),
            _AZURE_AI_CHAT,
        ),
        "messages-azure_ai-bool-budget-400": Cell(
            MESSAGES, "azure_ai", _KIMI_INFO, {"thinking": _BOOL_BUDGET}, None, _AZURE_AI_CHAT, 400, "thinking"
        ),
        "messages-azure_ai-null-budget-400": Cell(
            MESSAGES, "azure_ai", _KIMI_INFO, {"thinking": _NULL_BUDGET}, None, _AZURE_AI_CHAT, 400, "thinking"
        ),
        "messages-azure_ai-list-budget-400": Cell(
            MESSAGES, "azure_ai", _KIMI_INFO, {"thinking": _LIST_BUDGET}, None, _AZURE_AI_CHAT, 400, "thinking"
        ),
        "messages-azure_ai-empty-string-budget-400": Cell(
            MESSAGES, "azure_ai", _KIMI_INFO, {"thinking": _EMPTY_BUDGET}, None, _AZURE_AI_CHAT, 400, "thinking"
        ),
        "messages-azure_ai-5kb-budget-400": Cell(
            MESSAGES, "azure_ai", _KIMI_INFO, {"thinking": _LONG_BUDGET}, None, _AZURE_AI_CHAT, 400, "thinking"
        ),
        "messages-azure_ai-stream-string-budget-400": Cell(
            MESSAGES,
            "azure_ai",
            _KIMI_INFO,
            {"thinking": _STRING_BUDGET, "stream": True},
            None,
            _AZURE_AI_CHAT,
            400,
            "thinking",
            stream=True,
        ),
        "messages-azure_ai-stream-string-budget-drop-params": Cell(
            MESSAGES,
            "azure_ai",
            _KIMI_INFO,
            {"thinking": _STRING_BUDGET, "stream": True, "drop_params": True},
            _bridged_stream_messages(None),
            _AZURE_AI_CHAT,
            stream=True,
        ),
        "messages-azure_ai-missing-budget-to-low": Cell(
            MESSAGES, "azure_ai", _KIMI_INFO, {"thinking": _MISSING_BUDGET}, _bridged_messages("low"), _AZURE_AI_CHAT
        ),
        "messages-azure_ai-disabled-thinking-sends-none": Cell(
            MESSAGES, "azure_ai", _KIMI_INFO, {"thinking": _DISABLED}, _bridged_messages("none"), _AZURE_AI_CHAT
        ),
        "messages-azure-budget-control": Cell(
            MESSAGES, "azure", _UNDECLARED, {"thinking": _BUDGET}, _azure_messages("medium"), _AZURE_CHAT
        ),
        "messages-azure-string-budget-400": Cell(
            MESSAGES, "azure", _UNDECLARED, {"thinking": _STRING_BUDGET}, None, _AZURE_CHAT, 400, "thinking"
        ),
        "messages-azure-string-budget-drop-params": Cell(
            MESSAGES,
            "azure",
            _UNDECLARED,
            {"thinking": _STRING_BUDGET, "drop_params": True},
            _azure_messages(None),
            _AZURE_CHAT,
        ),
        "messages-azure-bool-budget-400": Cell(
            MESSAGES, "azure", _UNDECLARED, {"thinking": _BOOL_BUDGET}, None, _AZURE_CHAT, 400, "thinking"
        ),
        "messages-azure-declared-string-budget-400": Cell(
            MESSAGES, "azure", _GPT_INFO, {"thinking": _STRING_BUDGET}, None, _AZURE_CHAT, 400, "budget_tokens"
        ),
        "messages-azure-declared-string-budget-drop-params": Cell(
            MESSAGES,
            "azure",
            _GPT_INFO,
            {"thinking": _STRING_BUDGET, "drop_params": True},
            _azure_messages(None),
            _AZURE_CHAT,
        ),
        "messages-mistral-string-budget-400": Cell(
            MESSAGES, "mistral", _UNDECLARED, {"thinking": _STRING_BUDGET}, None, _VLLM_CHAT, 400, "thinking"
        ),
        "messages-openai-budget-control": Cell(
            MESSAGES,
            "openai",
            _UNDECLARED,
            {"thinking": _BUDGET},
            _responses_bridged_messages("medium"),
            _OPENAI_RESPONSES,
        ),
        "messages-openai-string-budget-400": Cell(
            MESSAGES, "openai", _UNDECLARED, {"thinking": _STRING_BUDGET}, None, _OPENAI_RESPONSES, 400, "thinking"
        ),
        "messages-openai-string-budget-drop-params": Cell(
            MESSAGES,
            "openai",
            _UNDECLARED,
            {"thinking": _STRING_BUDGET, "drop_params": True},
            _responses_bridged_messages_without_effort(),
            _OPENAI_RESPONSES,
        ),
        "messages-openai-bool-budget-400": Cell(
            MESSAGES, "openai", _UNDECLARED, {"thinking": _BOOL_BUDGET}, None, _OPENAI_RESPONSES, 400, "thinking"
        ),
        "messages-openai-string-budget-deployment-drop-params": Cell(
            MESSAGES,
            "openai_drop_params",
            _UNDECLARED,
            {"thinking": _STRING_BUDGET},
            _responses_bridged_messages_without_effort(),
            _OPENAI_RESPONSES,
        ),
        "messages-fireworks-budget-control": Cell(
            MESSAGES,
            "fireworks_ai",
            _UNDECLARED,
            {"thinking": _BUDGET},
            _bridged_messages("medium", model=FIREWORKS_KIMI),
            _VLLM_CHAT,
        ),
        "messages-fireworks-string-budget-400": Cell(
            MESSAGES, "fireworks_ai", _UNDECLARED, {"thinking": _STRING_BUDGET}, None, _VLLM_CHAT, 400, "thinking"
        ),
        "messages-fireworks-string-budget-drop-params": Cell(
            MESSAGES,
            "fireworks_ai",
            _UNDECLARED,
            {"thinking": _STRING_BUDGET, "drop_params": True},
            _bridged_messages(None, model=FIREWORKS_KIMI),
            _VLLM_CHAT,
        ),
        "messages-fireworks-declared-string-budget-400": Cell(
            MESSAGES,
            "fireworks_ai",
            {"reasoning_effort_levels": ["low", "high"]},
            {"thinking": _STRING_BUDGET},
            None,
            _VLLM_CHAT,
            400,
            "thinking",
        ),
        "messages-fireworks-bool-budget-400": Cell(
            MESSAGES, "fireworks_ai", _UNDECLARED, {"thinking": _BOOL_BUDGET}, None, _VLLM_CHAT, 400, "thinking"
        ),
        "messages-openrouter-budget-control": Cell(
            MESSAGES,
            "openrouter",
            _UNDECLARED,
            {"thinking": _BUDGET},
            _openrouter_messages("medium"),
            _VLLM_CHAT,
        ),
        "messages-openrouter-string-budget-drop-params": Cell(
            MESSAGES,
            "openrouter",
            _UNDECLARED,
            {"thinking": _STRING_BUDGET, "drop_params": True},
            _openrouter_messages(None),
            _VLLM_CHAT,
        ),
        "messages-gemini-budget-control": Cell(
            MESSAGES, "gemini", _UNDECLARED, {"thinking": _BUDGET}, _gemini_messages("medium"), _GEMINI
        ),
        "messages-gemini-string-budget-400": Cell(
            MESSAGES, "gemini", _UNDECLARED, {"thinking": _STRING_BUDGET}, None, _GEMINI, 400, "thinking"
        ),
        "messages-gemini-bool-budget-400": Cell(
            MESSAGES, "gemini", _UNDECLARED, {"thinking": _BOOL_BUDGET}, None, _GEMINI, 400, "thinking"
        ),
        "messages-zai-bool-budget-400": Cell(
            MESSAGES, "zai", _UNDECLARED, {"thinking": _BOOL_BUDGET}, None, _VLLM_CHAT, 400
        ),
        "messages-zai-bool-budget-drop-params": Cell(
            MESSAGES,
            "zai",
            _UNDECLARED,
            {"thinking": _BOOL_BUDGET, "drop_params": True},
            _bridged_messages(None, model=ZAI_GLM),
            _VLLM_CHAT,
        ),
        "messages-vllm-undeclared-string-budget-400": Cell(
            MESSAGES, "hosted_vllm", _UNDECLARED, {"thinking": _STRING_BUDGET}, None, _VLLM_CHAT, 400, "budget_tokens"
        ),
        "messages-vllm-string-budget-drop-params": Cell(
            MESSAGES,
            "hosted_vllm",
            _KIMI_INFO,
            {"thinking": _STRING_BUDGET, "drop_params": True},
            _bridged_messages(None),
            _VLLM_CHAT,
        ),
        "messages-vllm-string-budget-deployment-drop-params": Cell(
            MESSAGES,
            "hosted_vllm_drop_params",
            _KIMI_INFO,
            {"thinking": _STRING_BUDGET},
            _bridged_messages(None),
            _VLLM_CHAT,
        ),
        "messages-vllm-list-budget-400": Cell(
            MESSAGES, "hosted_vllm", _KIMI_INFO, {"thinking": _LIST_BUDGET}, None, _VLLM_CHAT, 400, "budget_tokens"
        ),
        "messages-vllm-bool-budget-400": Cell(
            MESSAGES, "hosted_vllm", _KIMI_INFO, {"thinking": _BOOL_BUDGET}, None, _VLLM_CHAT, 400, "budget_tokens"
        ),
        "messages-vllm-claude-name-declared-string-budget-400": Cell(
            MESSAGES,
            "hosted_vllm_claude",
            {"reasoning_effort_levels": ["low", "high", "max"]},
            {"thinking": _STRING_BUDGET},
            None,
            _VLLM_CHAT,
            400,
            "budget_tokens",
        ),
        "chat-vllm-claude-name-declared-adaptive-thinking-sends-no-effort": Cell(
            CHAT,
            "hosted_vllm_claude",
            {"reasoning_effort_levels": ["low", "high", "max"]},
            {"thinking": _ADAPTIVE},
            _openai_compatible(None, model=CLAUDE),
            _VLLM_CHAT,
        ),
        "chat-vllm-explicit-effort-wins-over-null-budget": Cell(
            CHAT,
            "hosted_vllm",
            _KIMI_INFO,
            {"reasoning_effort": "medium", "thinking": _NULL_BUDGET},
            _openai_compatible("high"),
            _VLLM_CHAT,
        ),
        "chat-vllm-explicit-effort-wins-over-list-budget": Cell(
            CHAT,
            "hosted_vllm",
            _KIMI_INFO,
            {"reasoning_effort": "medium", "thinking": _LIST_BUDGET},
            _openai_compatible("high"),
            _VLLM_CHAT,
        ),
        "chat-vllm-explicit-effort-wins-over-5kb-budget": Cell(
            CHAT,
            "hosted_vllm",
            _KIMI_INFO,
            {"reasoning_effort": "medium", "thinking": _LONG_BUDGET},
            _openai_compatible("high"),
            _VLLM_CHAT,
        ),
    }
)


def _request_body(cell: Cell, model: str, marker: str) -> dict[str, JsonValue]:
    match cell.route:
        case "/v1/chat/completions":
            return chat_request(model, marker, **cell.fields)
        case "/v1/messages":
            return messages_request(model, marker, **cell.fields)
        case "/v1/responses":
            return responses_request(model, marker, **cell.fields)
        case _:
            raise AssertionError(cell.route)


def _assert_answered(cell: Cell, response: httpx.Response, marker: str) -> None:
    assert response.status_code == cell.status, response.text[:2000]
    if cell.status != 200:
        assert cell.error in response.text, response.text[:2000]
        return
    assert answer(marker) in response.text, response.text[:2000]
    if cell.stream:
        return
    payload: Final = object_value(response.json())
    if cell.route == CHAT and cell.kind in (
        "hosted_vllm",
        "hosted_vllm_claude",
        "hosted_vllm_echoed_key",
        "hosted_vllm_drop_params",
        "azure_ai",
        "openai",
        "azure",
    ):
        assert payload["id"] == f"chatcmpl-{marker}", payload
    if cell.route == MESSAGES and cell.kind in ("anthropic", "azure_ai_claude", "bedrock_invoke"):
        assert payload["id"] == f"msg_{marker}", payload


@pytest.mark.parametrize("cell", [pytest.param(cell, id=name) for name, cell in CELLS.items()])
def test_reasoning_effort_reaches_a_declaring_deployment_as_one_of_its_levels(gateway: Gateway, cell: Cell) -> None:
    marker: Final = new_marker()
    with wire_server(provider_reply) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, cell.kind, wire, cell.model_info)
        response: Final = gateway.request("POST", cell.route, _request_body(cell, model, marker))
        received: Final = wire.drain()
    _assert_answered(cell, response, marker)
    if cell.expected is None:
        assert received == (), [request.body[:300] for request in received]
        return
    assert [(request.method, request.target) for request in received] == [("POST", cell.target)]
    assert marker_of(received[0]) == marker
    assert body_of(received[0]) == cell.expected(marker)


def _single(received: tuple[Request, ...], marker: str) -> Request:
    matching: Final = tuple(request for request in received if marker_of(request) == marker)
    assert len(matching) == 1, [request.body[:300] for request in received]
    return matching[0]


def _proxy_url(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


async def _streamed_openai_text(base_url: str, key: str, model: str, marker: str) -> str:
    client: Final = openai.AsyncOpenAI(base_url=base_url, api_key=key, max_retries=0)
    stream: Final = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt(marker)}],
        reasoning_effort="minimal",
        stream=True,
    )
    parts: Final = [chunk.choices[0].delta.content or "" async for chunk in stream if chunk.choices]
    return "".join(parts)


def test_openai_sdk_sync_and_async_clients_reach_azure_ai_with_the_declared_level(gateway: Gateway) -> None:
    sync_marker: Final = new_marker()
    stream_marker: Final = new_marker()
    with wire_server(provider_reply) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, "azure_ai", wire, _KIMI_INFO)
        base_url: Final = f"{_proxy_url(gateway)}/v1"
        completion: Final = openai.OpenAI(
            base_url=base_url, api_key=gateway.key, max_retries=0
        ).chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt(sync_marker)}],
            reasoning_effort="medium",
        )
        streamed_text: Final = asyncio.run(_streamed_openai_text(base_url, gateway.key, model, stream_marker))
        received: Final = wire.drain()
    assert completion.id == f"chatcmpl-{sync_marker}"
    assert completion.choices[0].message.content == answer(sync_marker)
    assert streamed_text == answer(stream_marker)
    assert len(received) == 2
    assert sent_effort(_single(received, sync_marker)) == "high"
    assert sent_effort(_single(received, stream_marker)) == "low"


def test_anthropic_sdk_sync_and_streaming_clients_reach_claude_with_the_declared_level(gateway: Gateway) -> None:
    sync_marker: Final = new_marker()
    stream_marker: Final = new_marker()
    with wire_server(provider_reply) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, "anthropic", wire, _CLAUDE_INFO)
        client: Final = anthropic.Anthropic(base_url=_proxy_url(gateway), api_key=gateway.key, max_retries=0)
        message: Final = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt(sync_marker)}],
            extra_body={"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}},
        )
        with client.messages.stream(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt(stream_marker)}],
            thinking={"type": "enabled", "budget_tokens": 2048},
        ) as stream:
            streamed_text: Final = "".join(stream.text_stream)
        received: Final = wire.drain()
    assert message.id == f"msg_{sync_marker}"
    assert [block.text for block in message.content if block.type == "text"] == [answer(sync_marker)]
    assert streamed_text == answer(stream_marker)
    assert len(received) == 2
    assert sent_effort(_single(received, sync_marker)) == "high"
    assert sent_effort(_single(received, stream_marker)) == "high"


def test_a_duplicated_effort_key_rounds_the_value_the_proxy_parsed(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(provider_reply) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, "hosted_vllm", wire, _KIMI_INFO)
        raw: Final = (
            f'{{"model": "{model}", "messages": [{{"role": "user", "content": "{prompt(marker)}"}}],'
            ' "reasoning_effort": "max", "reasoning_effort": "minimal"}'
        )
        response: Final = gateway.client.post(
            CHAT,
            content=raw.encode(),
            headers={"Authorization": f"Bearer {gateway.key}", "Content-Type": "application/json"},
        )
        received: Final = wire.drain()
    assert response.status_code == 200, response.text
    assert object_value(response.json())["id"] == f"chatcmpl-{marker}"
    assert sent_effort(_single(received, marker)) == "low"


def _raw_messages_post(gateway: Gateway, model: str, marker: str, budgets: str) -> httpx.Response:
    raw: Final = (
        f'{{"model": "{model}", "max_tokens": 4096, "messages": [{{"role": "user", "content": "{prompt(marker)}"}}],'
        f' "thinking": {{"type": "enabled", {budgets}}}}}'
    )
    return gateway.client.post(
        MESSAGES,
        content=raw.encode(),
        headers={"Authorization": f"Bearer {gateway.key}", "Content-Type": "application/json"},
    )


def test_a_duplicated_budget_key_on_messages_is_judged_by_the_value_the_proxy_parsed(gateway: Gateway) -> None:
    malformed_last: Final = new_marker()
    numeric_last: Final = new_marker()
    with wire_server(provider_reply) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, "azure_ai", wire, _KIMI_INFO)
        rejected: Final = _raw_messages_post(
            gateway, model, malformed_last, '"budget_tokens": 9000, "budget_tokens": "2048"'
        )
        answered: Final = _raw_messages_post(
            gateway, model, numeric_last, '"budget_tokens": "2048", "budget_tokens": 9000'
        )
        received: Final = wire.drain()
    assert rejected.status_code == 400, rejected.text
    assert "thinking" in rejected.text
    assert answered.status_code == 200, answered.text
    assert answer(numeric_last) in answered.text
    assert [marker_of(request) for request in received] == [numeric_last]
    assert sent_effort(received[0]) == "high"


def test_anthropic_sdk_gets_a_400_for_a_malformed_budget_to_a_non_claude_deployment_and_an_answer_under_drop_params(
    gateway: Gateway,
) -> None:
    rejected_marker: Final = new_marker()
    sync_marker: Final = new_marker()
    stream_marker: Final = new_marker()
    malformed: Final = {"thinking": {"type": "enabled", "budget_tokens": "2048"}}
    with wire_server(provider_reply) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, "azure_ai", wire, _KIMI_INFO)
        client: Final = anthropic.Anthropic(base_url=_proxy_url(gateway), api_key=gateway.key, max_retries=0)
        with pytest.raises(anthropic.BadRequestError) as rejected:
            client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt(rejected_marker)}],
                extra_body=malformed,
            )
        message: Final = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt(sync_marker)}],
            extra_body={**malformed, "drop_params": True},
        )
        with client.messages.stream(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt(stream_marker)}],
            extra_body={**malformed, "drop_params": True},
        ) as stream:
            streamed_text: Final = "".join(stream.text_stream)
        received: Final = wire.drain()
    assert rejected.value.status_code == 400
    assert "thinking" in str(rejected.value)
    assert [block.text for block in message.content if block.type == "text"] == [answer(sync_marker)]
    assert streamed_text == answer(stream_marker)
    assert sorted(marker_of(request) for request in received) == sorted((sync_marker, stream_marker))
    assert [sorted(set(body_of(request)) & {"thinking", "reasoning_effort"}) for request in received] == [[], []]


def _served_by(response: httpx.Response) -> str:
    assert response.status_code == 200, response.text
    return response.headers["x-litellm-model-id"]


def test_each_deployment_of_a_model_group_gets_its_own_declared_level(gateway: Gateway) -> None:
    markers: Final = tuple(new_marker() for _ in range(10))
    with (
        wire_server(provider_reply) as low_wire,
        wire_server(provider_reply) as high_wire,
        gateway.scenario() as scenario,
    ):
        group: Final = deployment(scenario, "hosted_vllm", low_wire, {"reasoning_effort_levels": ["minimal", "low"]})
        sibling: Final = scenario.gateway.post(
            "/model/new",
            {
                "model_name": group,
                "litellm_params": {"model": f"hosted_vllm/{KIMI}", "api_base": f"{high_wire.url}/v1", "api_key": "k"},
                "model_info": {"reasoning_effort_levels": ["high", "max"]},
            },
        )
        sibling_id: Final = string_value(object_value(sibling["model_info"])["id"])
        scenario.cleanups.callback(scenario.delete_model, sibling_id)
        served: Final = tuple(
            _served_by(gateway.request("POST", CHAT, chat_request(group, marker, reasoning_effort="medium")))
            for marker in markers
        )
        low_received: Final = low_wire.drain()
        high_received: Final = high_wire.drain()
    expected_high: Final = frozenset(marker for marker, model_id in zip(markers, served) if model_id == sibling_id)
    assert {marker_of(request) for request in high_received} == expected_high
    assert {marker_of(request) for request in low_received} == frozenset(markers) - expected_high
    assert len(low_received) + len(high_received) == len(markers)
    assert {sent_effort(request) for request in low_received} <= {"low"}
    assert {sent_effort(request) for request in high_received} <= {"high"}


def test_a_fallback_deployment_rounds_the_callers_level_onto_its_own_declaration(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with (
        wire_server(lambda request: openai_error(500)) as failing,
        wire_server(provider_reply) as serving,
        gateway.scenario() as scenario,
    ):
        primary: Final = deployment(scenario, "hosted_vllm", failing, {"reasoning_effort_levels": ["minimal", "high"]})
        fallback: Final = deployment(scenario, "hosted_vllm", serving, {"reasoning_effort_levels": ["low", "max"]})
        response: Final = gateway.request(
            "POST",
            CHAT,
            chat_request(primary, marker, reasoning_effort="medium", fallbacks=[fallback], num_retries=0),
        )
        failed: Final = failing.drain()
        served: Final = serving.drain()
    assert response.status_code == 200, response.text
    assert object_value(response.json())["id"] == f"chatcmpl-{marker}"
    assert failed and {sent_effort(request) for request in failed} == {"high"}
    assert [sent_effort(request) for request in served] == ["max"]


def _deployment_id(gateway: Gateway, model: str) -> str:
    entries: Final = tuple(object_value(entry) for entry in list_value(gateway.get("/model/info", {})["data"]))
    matching: Final = tuple(entry for entry in entries if entry["model_name"] == model)
    assert len(matching) == 1, [entry["model_name"] for entry in entries]
    return string_value(object_value(matching[0]["model_info"])["id"])


def _declared_levels(gateway: Gateway, identity: str) -> JsonValue:
    entries: Final = list_value(gateway.get("/model/info", {"litellm_model_id": identity})["data"])
    return object_value(object_value(entries[0])["model_info"]).get("reasoning_effort_levels")


def test_a_declaration_updated_under_traffic_applies_to_the_next_request(gateway: Gateway) -> None:
    from litellm.constants import PROXY_CONFIG_RELOAD_INTERVAL_SECONDS

    first: Final = new_marker()
    second: Final = new_marker()
    with wire_server(provider_reply) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, "hosted_vllm", wire, {"reasoning_effort_levels": ["low", "high"]})
        identity: Final = _deployment_id(gateway, model)
        before: Final = gateway.request("POST", CHAT, chat_request(model, first, reasoning_effort="medium"))
        updated: Final = gateway.request(
            "PATCH", f"/model/{identity}/update", {"model_info": {"reasoning_effort_levels": ["minimal", "max"]}}
        )
        eventually(
            lambda: _declared_levels(gateway, identity),
            lambda levels: levels == ["minimal", "max"],
            seconds=2 * PROXY_CONFIG_RELOAD_INTERVAL_SECONDS,
        )
        after: Final = gateway.request("POST", CHAT, chat_request(model, second, reasoning_effort="medium"))
        received: Final = wire.drain()
    assert before.status_code == 200, before.text
    assert updated.status_code == 200, updated.text
    assert after.status_code == 200, after.text
    assert sent_effort(_single(received, first)) == "high"
    assert sent_effort(_single(received, second)) == "max"


def test_a_repeated_identical_request_is_served_from_cache_after_one_rounded_upstream_call(gateway: Gateway) -> None:
    marker: Final = new_marker()
    with wire_server(provider_reply) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, "hosted_vllm", wire, _KIMI_INFO)
        body: Final = chat_request(model, marker, reasoning_effort="medium")
        first: Final = gateway.request("POST", CHAT, body)
        second: Final = gateway.request("POST", CHAT, body)
        received: Final = wire.drain()
    assert first.status_code == 200 and second.status_code == 200, (first.text, second.text)
    assert object_value(first.json())["id"] == object_value(second.json())["id"] == f"chatcmpl-{marker}"
    assert [sent_effort(request) for request in received] == ["high"]


def test_model_group_info_advertises_the_declared_levels(gateway: Gateway) -> None:
    with wire_server(provider_reply) as wire, gateway.scenario() as scenario:
        model: Final = deployment(scenario, "hosted_vllm", wire, {**_KIMI_INFO, "supports_reasoning": True})
        info: Final = gateway.get("/model_group/info", {"model_group": model})
    entries: Final = list_value(info["data"])
    assert len(entries) == 1, info
    assert object_value(entries[0])["supported_reasoning_efforts"] == ["none", "low", "high", "max"]


def _create_swapping_prompt(gateway: Gateway, prompt_id: str, model: str, marker: str) -> None:
    created: Final = gateway.request(
        "POST",
        "/prompts",
        {
            "prompt_id": prompt_id,
            "litellm_params": {
                "prompt_id": prompt_id,
                "prompt_integration": "dotprompt",
                "dotprompt_content": f"---\nmodel: {model}\n---\nUser: {prompt(marker)}",
            },
            "prompt_info": {"prompt_type": "db"},
        },
    )
    assert created.status_code == 200, created.text


def test_a_prompt_that_swaps_the_model_is_rounded_onto_the_serving_deployments_levels(gateway: Gateway) -> None:
    marker: Final = new_marker()
    prompt_id: Final = f"prompt-{new_marker()}"
    with (
        wire_server(provider_reply) as routed_wire,
        wire_server(provider_reply) as swapped_wire,
        gateway.scenario() as scenario,
    ):
        routed: Final = deployment(
            scenario, "hosted_vllm", routed_wire, {"reasoning_effort_levels": ["minimal", "max"]}
        )
        swapped: Final = deployment(scenario, "hosted_vllm", swapped_wire, {"reasoning_effort_levels": ["low", "high"]})
        _create_swapping_prompt(gateway, prompt_id, swapped, marker)
        scenario.cleanups.callback(gateway.request, "DELETE", f"/prompts/{prompt_id}")
        response: Final = gateway.request(
            "POST", CHAT, chat_request(routed, marker, reasoning_effort="medium", prompt_id=prompt_id)
        )
        routed_received: Final = routed_wire.drain()
        swapped_received: Final = swapped_wire.drain()
    assert response.status_code == 200, response.text
    assert object_value(response.json())["id"] == f"chatcmpl-{marker}"
    efforts: Final = [("routed", sent_effort(request)) for request in routed_received] + [
        ("swapped", sent_effort(request)) for request in swapped_received
    ]
    assert efforts in ([("routed", "max")], [("swapped", "high")]), efforts
