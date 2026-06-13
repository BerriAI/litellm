"""Parameter gates for the ollama_chat serializer.

v1's supported list (``OllamaChatConfig.get_supported_openai_params``):
max_tokens/max_completion_tokens/stream/top_p/temperature/seed/
frequency_penalty/stop/tools/tool_choice/functions/response_format/
reasoning_effort. ``_check_valid_arg`` RAISES UnsupportedParamsError on
anything else (probed at HEAD: n, logprobs, presence_penalty, thinking).
IR-carried params outside the list are typed fallbacks so v1 serves its own
raise; ``seed``/``frequency_penalty``/``functions`` and the provider-native
passthroughs (mirostat, num_ctx, keep_alive, format, think, function_name)
are parse-level unknowns — v1 SERVES each, the inbound boundary falls back.

``top_k`` is NOT in the openai list but IS in the IR: v1's provider-native
passthrough places it in optional_params and the transform emits it inside
``options`` — SERVED (the cohere/snowflake top-level-passthrough shape).

``user`` is silently dropped upstream (model-list gated, never raised) —
typed fallback, v1 serves its own drop.

Tool-level ``cache_control`` falls back: v1 forwards the caller's tools
VERBATIM (no strip anywhere on this path) while the shared openai tool
assembly drops the marker.
"""

from __future__ import annotations

from ...deps import TranslationDeps
from ...ir import ChatRequest


def unsupported_params(request: ChatRequest, deps: TranslationDeps) -> str | None:
    # deps is unused — the uniform own-module gate signature
    if request.parallel_tool_calls.is_some():
        return (
            "parallel_tool_calls is outside ollama_chat's supported list; "
            "v1's get_optional_params raises UnsupportedParamsError (or "
            "drops it under drop_params)"
        )
    if request.thinking.is_some():
        return (
            "thinking is outside ollama_chat's supported list; v1 raises "
            "UnsupportedParamsError (reasoning_effort is the served knob)"
        )
    if request.user.is_some():
        return (
            "user is model-list gated upstream and silently dropped for "
            "ollama_chat; v1 serves its own drop"
        )
    if any(tool.cache.is_some() for tool in request.tools):
        return (
            "tool-level cache_control: v1 forwards the caller's tools "
            "verbatim onto the ollama wire; the shared tool assembly strips "
            "the marker"
        )
    return None
