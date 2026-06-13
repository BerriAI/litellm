"""Parameter gates for the databricks chat serializer.

``DatabricksConfig.get_supported_openai_params`` (probed in-process at HEAD):
stream/stop/temperature/top_p/max_tokens/max_completion_tokens/n/
response_format/tools/tool_choice/reasoning_effort/thinking. v1's
``_check_valid_arg`` RAISES UnsupportedParamsError on anything else (probed:
frequency_penalty, presence_penalty, seed, logprobs, parallel_tool_calls);
``user`` is silently dropped upstream (model-list gated, never raised).

The central structural fact is the ``"claude" in model`` SUBSTRING fork
(DB-R1): a model whose name CONTAINS "claude" (custom serving endpoints are
user-named, so ``my-claude-proxy-llama`` counts) gets anthropic param
treatment for tools / response_format / reasoning_effort. The gate uses the
SAME substring, never a model-map lookup.

Probed serve/fallback split (the wave-3 dossier's narrower set):

- SERVE: plain chat all models; temperature/top_p/stop/n; mct -> max_tokens
  (ALWAYS renamed — v1 never re-emits max_completion_tokens); top_k verbatim
  top-level (researcher-5 said unsupported; the HEAD probe REFUTES it — v1
  serves it on the wire, both arms); tools on NON-claude verbatim; tools on
  claude (the openai -> anthropic -> databricks round-trip, pure); tool_choice;
  thinking passthrough + the max-bump arithmetic; reasoning_effort on
  NON-claude WITH max_tokens (verbatim one-liner); response_format on
  NON-claude verbatim.
- TYPED FALLBACK (v1 SERVES): response_format on claude (the json_tool_call +
  json_mode machinery spans request/response/stream state — DB-R2's
  json_object SILENT DROP lives in this arm too); reasoning_effort on claude
  (the thinking machinery + the adaptive output_config arm); cache_control on
  ANY message (v1 MOVES message-level markers into a text block, and a
  whitespace-only message LOSES the marker with the sanitized block — a trap;
  guard-side); ``user`` (v1 serves its own drop).
- FALLBACK (v1 RAISES): parallel_tool_calls (UPE); reasoning_effort on
  NON-claude WITHOUT max_tokens (the raw ``KeyError: 'thinking'`` crash,
  DB-R3 — v1 serves its own crash, the row asserts v1 raises KeyError).
"""

from __future__ import annotations

from ...ir import ChatRequest


def is_claude(model: str) -> bool:
    """v1's ``"claude" in model`` substring (DB-R1) — NOT a model-map
    capability lookup; user-named endpoints carrying the substring fork too."""
    return "claude" in model


def unsupported_params(request: ChatRequest) -> str | None:
    if request.parallel_tool_calls.is_some():
        return (
            "parallel_tool_calls is outside databricks' supported list; v1's "
            "get_optional_params raises UnsupportedParamsError"
        )
    if request.user.is_some():
        return (
            "user is model-list gated upstream and silently dropped for "
            "databricks; v1 serves its own drop"
        )
    claude = is_claude(request.model)
    if request.response_format.is_some() and claude:
        return (
            "response_format on a claude-substring model: v1 routes json_schema "
            "through the json_tool_call + json_mode machinery and SILENTLY DROPS "
            "json_object entirely (DB-R2) — the cross-plane state is unported, "
            "typed fallback so v1 serves it"
        )
    if request.reasoning_effort.is_some() and claude:
        return (
            "reasoning_effort on a claude-substring model: v1 maps it to "
            "thinking{enabled,budget} (and the adaptive output_config arm on "
            "opus-4-6 names) — unported, typed fallback so v1 serves it"
        )
    return _nonclaude_reasoning_effort(request, claude)


def _nonclaude_reasoning_effort(request: ChatRequest, claude: bool) -> str | None:
    if claude or request.reasoning_effort.is_none():
        return None
    if (
        request.params.max_tokens.is_none()
        and request.params.max_completion_tokens.is_none()
    ):
        return (
            "reasoning_effort on a NON-claude model WITHOUT max_tokens: v1 "
            "CRASHES with a raw KeyError('thinking') in "
            "update_optional_params_with_thinking_tokens (DB-R3, probed) — typed "
            "fallback so v1 serves its own crash"
        )
    return None
