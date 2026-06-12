"""Per-provider parameter gates for the SDK-path openai-compat family.

v1's gate for every provider here is ``_check_valid_arg`` over the provider
config's ``get_supported_openai_params``: an unsupported param RAISES
``UnsupportedParamsError`` unless ``drop_params``, in which case it is popped
BEFORE ``map_openai_params`` runs. v2 mirrors the SUPPORTED-LIST truth as
typed fallbacks (the v2-openai/xai precedent — never re-implement the
raise-vs-drop interplay): every IR-carried param a provider's list excludes
falls back so v1 serves its own raise or drop. Params outside the IR
(seed, penalties, logprobs, n, stream_options, ...) already fall back at the
inbound boundary and never reach these gates.

The capability reads mirror v1's model-map lookups through
``deps.supports_capability`` over the ``{provider}/{model}`` map key — the
provider prefix is LOAD-BEARING (bare wire models have no model-map rows;
see the xai drift-gate note). Verified in-process at HEAD:
``together_ai/...`` rows answer ``supports_function_calling`` and
``cerebras/...`` rows answer ``supports_reasoning``; the bare keys are False
even for capable models.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Literal

from ...deps import TranslationDeps
from ...ir import ChatRequest
from ..openai_compat.params import (
    RESPONSE_FORMAT_UNSUPPORTED_MODELS,
    unsupported_response_format,
)

CompatSdkProvider = Literal[
    "together_ai",
    "cerebras",
    "nvidia_nim",
    "lm_studio",
    "llamafile",
    "lambda_ai",
    "nebius",
    "novita",
    "wandb",
    "featherless_ai",
    "nscale",
    "hyperbolic",
    "volcengine",
    # wave-2a: profile rows + named gates (researcher-4 Part 1, entries 1-5).
    # cometapi is the family's first httpx-path member: same request profile,
    # but NO seam model preset (bare wire model, the xai R4 pin) and its own
    # reasoning-rename stream parser (cometapi_stream.py).
    "perplexity",
    "sambanova",
    "deepinfra",
    "moonshot",
    "cometapi",
]

_Present = Callable[[ChatRequest], bool]

# IR-carried params, checked in a stable order so fallback reasons are
# deterministic. ``max_tokens`` only counts as caller-sent when
# ``max_completion_tokens`` is absent: the inbound parse collapses mct into
# max_tokens and the raw guard rejects requests carrying both keys.
_CHECKS: tuple[tuple[str, _Present], ...] = (
    (
        "max_tokens",
        lambda r: r.params.max_tokens.is_some()
        and r.params.max_completion_tokens.is_none(),
    ),
    ("max_completion_tokens", lambda r: r.params.max_completion_tokens.is_some()),
    ("temperature", lambda r: r.params.temperature.is_some()),
    ("top_p", lambda r: r.params.top_p.is_some()),
    ("top_k", lambda r: r.params.top_k.is_some()),
    ("stream", lambda r: r.stream),
    ("stop", lambda r: len(r.params.stop) > 0),
    ("tools", lambda r: len(r.tools) > 0),
    ("tool_choice", lambda r: r.tool_choice.is_some()),
    ("parallel_tool_calls", lambda r: r.parallel_tool_calls.is_some()),
    ("response_format", lambda r: r.response_format.is_some()),
    ("user", lambda r: r.user.is_some()),
    ("reasoning_effort", lambda r: r.reasoning_effort.is_some()),
    ("thinking", lambda r: r.thinking.is_some()),
)

_NO_NOTES: Mapping[str, str] = MappingProxyType({})


def unsupported_against(
    request: ChatRequest,
    *,
    provider: str,
    allowed: frozenset[str],
    notes: Mapping[str, str] = _NO_NOTES,
) -> str | None:
    for key, present in _CHECKS:
        if not present(request) or key in allowed:
            continue
        note = notes.get(key)
        if note is not None:
            return note
        if key == "top_k":
            # top_k is not an OpenAI param: it never enters
            # non_default_params, so _check_valid_arg never sees it and v1
            # drops it WITHOUT drop_params (verified in-process at HEAD;
            # verifier-wave1a F6).
            return (
                f"top_k on {provider}: not an OpenAI param; v1's "
                "get_optional_params silently drops it (no raise, even "
                "without drop_params)"
            )
        return (
            f"{key} on {provider}: outside v1's supported list; "
            "get_optional_params raises UnsupportedParamsError "
            "(or drops it under drop_params)"
        )
    return None


def _user_note(provider: str) -> str:
    return (
        f"user on {provider}: gated on litellm.open_ai_chat_completion_models "
        "membership in v1's base supported list; v1 handles it"
    )


# OpenAIGPTConfig's base list restricted to IR-carried keys. ``user`` is
# deliberately absent (model-list gated in v1) and ``response_format`` rides
# the base list's gpt-4/gpt-3.5-turbo-16k name gate, applied per provider
# below for the configs that inherit the base list.
_BASE_LIST = frozenset(
    {
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stream",
        "stop",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
    }
)

_FUNCTION_CALLING_KEYS = frozenset({"tools", "tool_choice", "response_format"})


def base_list_unsupported(
    request: ChatRequest, deps: TranslationDeps, provider: str
) -> str | None:
    """llamafile / novita / lm_studio (plain base config) and lambda_ai /
    nebius / wandb (base list + mct rename, applied in serialize)."""
    return unsupported_against(
        request,
        provider=provider,
        allowed=_BASE_LIST,
        notes={"user": _user_note(provider)},
    ) or unsupported_response_format(request)


def supports_together_tools(model: str, deps: TranslationDeps) -> bool:
    return deps.supports_capability(f"together_ai/{model}", "supports_function_calling")


def together_ai_unsupported(request: ChatRequest, deps: TranslationDeps) -> str | None:
    """TogetherAIConfig removes tools/tool_choice/response_format from the
    base list unless ``supports_function_calling(model, "together_ai")`` is
    True (together_ai/chat.py); parallel_tool_calls stays supported either
    way (v1 truth, not an oversight here)."""
    supports_tools = supports_together_tools(request.model, deps)
    if request.model in RESPONSE_FORMAT_UNSUPPORTED_MODELS and not supports_tools:
        # The base list already dropped response_format for this model name,
        # so the non-fc fork's list.remove("response_format") crashes inside
        # get_supported_openai_params -- which _check_valid_arg runs on EVERY
        # together_ai request, plain text included (verifier-wave1a F2).
        return (
            f"model {request.model!r} on together_ai without the model-map "
            "supports_function_calling flag: v1's TogetherAIConfig."
            "get_supported_openai_params raises ValueError (list.remove on "
            "a base list that already dropped response_format for this "
            "model name) on every request; v1 raises its own error"
        )
    allowed = _BASE_LIST if supports_tools else _BASE_LIST - _FUNCTION_CALLING_KEYS
    return unsupported_against(
        request,
        provider="together_ai",
        allowed=allowed,
        notes={"user": _user_note("together_ai")},
    ) or unsupported_response_format(request)


# The MAXIMAL cerebras surface: ``reasoning_effort`` is capability-narrowed
# per model in the gate below, so its membership here means "emittable", not
# "always allowed" (the ALLOWED table at the bottom feeds emission).
_CEREBRAS_LIST = frozenset(
    {
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stream",
        "stop",
        "tools",
        "tool_choice",
        "response_format",
        "user",
        "reasoning_effort",
    }
)


def supports_cerebras_reasoning(model: str, deps: TranslationDeps) -> bool:
    return deps.supports_capability(f"cerebras/{model}", "supports_reasoning")


def cerebras_unsupported(request: ChatRequest, deps: TranslationDeps) -> str | None:
    allowed = (
        _CEREBRAS_LIST
        if supports_cerebras_reasoning(request.model, deps)
        else _CEREBRAS_LIST - {"reasoning_effort"}
    )
    return unsupported_against(
        request,
        provider="cerebras",
        allowed=allowed,
        notes={
            "reasoning_effort": (
                f"reasoning_effort on non-reasoning cerebras model {request.model} "
                "(model-map supports_reasoning gate); v1 raises or drops it"
            )
        },
    )


# NvidiaNimConfig's static per-model allowlists (nvidia_nim/chat/
# transformation.py), restricted to IR-carried keys; the drift gate re-derives
# them from the v1 config at HEAD.
_NVIDIA_GEMMA_MODELS = frozenset(
    {
        "google/recurrentgemma-2b",
        "google/gemma-2-27b-it",
        "google/gemma-2-9b-it",
        "gemma-2-9b-it",
    }
)
_NVIDIA_GEMMA_LIST = frozenset({"stream", "temperature", "top_p", "max_tokens", "stop"})
_NVIDIA_NEMOTRON_INSTRUCT_LIST = frozenset(
    {"stream", "temperature", "top_p", "max_tokens", "max_completion_tokens"}
)
_NVIDIA_REWARD_LIST = frozenset({"stream"})
_NVIDIA_CODEGEMMA_LIST = frozenset(
    {"stream", "temperature", "top_p", "max_tokens", "max_completion_tokens", "stop"}
)
_NVIDIA_DEFAULT_LIST = frozenset(
    {
        "stream",
        "temperature",
        "top_p",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "response_format",
    }
)


def nvidia_nim_allowed(model: str) -> frozenset[str]:
    if model in _NVIDIA_GEMMA_MODELS:
        return _NVIDIA_GEMMA_LIST
    if model == "nvidia/nemotron-4-340b-instruct":
        return _NVIDIA_NEMOTRON_INSTRUCT_LIST
    if model == "nvidia/nemotron-4-340b-reward":
        return _NVIDIA_REWARD_LIST
    if model == "google/codegemma-1.1-7b":
        return _NVIDIA_CODEGEMMA_LIST
    return _NVIDIA_DEFAULT_LIST


def nvidia_nim_unsupported(request: ChatRequest, deps: TranslationDeps) -> str | None:
    return unsupported_against(
        request, provider="nvidia_nim", allowed=nvidia_nim_allowed(request.model)
    )


_FEATHERLESS_LIST = frozenset(
    {
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stream",
        "stop",
    }
)


def featherless_ai_unsupported(
    request: ChatRequest, deps: TranslationDeps
) -> str | None:
    """tools and tool_choice are outside FeatherlessAIConfig's supported
    list, so ``_check_valid_arg`` raises before the map's tool_choice
    auto/none arm can run — that arm is dead code (the xai R2 pattern,
    verified in-process at HEAD)."""
    return unsupported_against(
        request, provider="featherless_ai", allowed=_FEATHERLESS_LIST
    )


_NSCALE_LIST = frozenset(
    {"max_tokens", "temperature", "top_p", "stream", "stop", "response_format"}
)


def nscale_unsupported(request: ChatRequest, deps: TranslationDeps) -> str | None:
    """NscaleConfig inherits the BASE map (no mct rename), and mct is outside
    its supported list, so max_completion_tokens RAISES (verified at HEAD)."""
    return unsupported_against(request, provider="nscale", allowed=_NSCALE_LIST)


_HYPERBOLIC_LIST = frozenset(
    {
        "max_tokens",
        "temperature",
        "top_p",
        "stream",
        "stop",
        "tools",
        "tool_choice",
        "response_format",
        "user",
    }
)


def hyperbolic_unsupported(request: ChatRequest, deps: TranslationDeps) -> str | None:
    """max_completion_tokens is outside HyperbolicChatConfig's own list, so
    OpenAILikeChatConfig's rename arm is dead code — v1 raises (verified at
    HEAD; the xai R2 trap again)."""
    return unsupported_against(request, provider="hyperbolic", allowed=_HYPERBOLIC_LIST)


_VOLCENGINE_LIST = frozenset(
    {
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stream",
        "stop",
        "tools",
        "tool_choice",
    }
)


def volcengine_unsupported(request: ChatRequest, deps: TranslationDeps) -> str | None:
    """response_format is OUTSIDE VolcEngineChatConfig's supported list
    (v1 raises, verified at HEAD). ``thinking`` IS supported in v1 but its
    map packs the verbatim dict into ``extra_body`` for the SDK to merge
    top-level — an unported crossing, so it falls back typed."""
    return unsupported_against(
        request,
        provider="volcengine",
        allowed=_VOLCENGINE_LIST,
        notes={
            "thinking": (
                "thinking on volcengine: v1 packs the verbatim dict into "
                "extra_body (VolcEngineChatConfig.map_openai_params) and the "
                "SDK merges it top-level; that crossing is unported, v1 "
                "serves it"
            )
        },
    )


# ---------------------------------------------------------------------------
# wave-2a gates. Shared fact (verified in-process at HEAD): for every provider
# below, get_optional_params packs top_k into ``extra_body`` (the
# provider-specific passthrough for openai-compatible providers) and the
# SDK/handler merges it top-level — v1 SERVES it; the crossing is unported.
# ---------------------------------------------------------------------------


def _top_k_extra_body_note(provider: str) -> str:
    return (
        f"top_k on {provider}: v1 packs it into extra_body (the "
        "provider-specific passthrough) and the SDK/handler merges it "
        "top-level; that crossing is unported, v1 serves it"
    )


# Perplexity's own reduced list (perplexity/chat/transformation.py:44-55),
# IR-restricted: stop/tools/tool_choice RAISE; mct passes VERBATIM (no map
# override); reasoning_effort is capability-narrowed per model below.
_PERPLEXITY_LIST = frozenset(
    {
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stream",
        "response_format",
        "reasoning_effort",
    }
)


def supports_perplexity_reasoning(model: str, deps: TranslationDeps) -> bool:
    return deps.supports_capability(f"perplexity/{model}", "supports_reasoning")


def perplexity_unsupported(request: ChatRequest, deps: TranslationDeps) -> str | None:
    allowed = (
        _PERPLEXITY_LIST
        if supports_perplexity_reasoning(request.model, deps)
        else _PERPLEXITY_LIST - {"reasoning_effort"}
    )
    return unsupported_against(
        request,
        provider="perplexity",
        allowed=allowed,
        notes={
            "user": _user_note("perplexity"),
            "top_k": _top_k_extra_body_note("perplexity"),
            "reasoning_effort": (
                f"reasoning_effort on non-reasoning perplexity model "
                f"{request.model} (model-map supports_reasoning gate); "
                "v1 raises or drops it"
            ),
        },
    )


# SambanovaConfig's list (sambanova/chat.py:56-80), IR-restricted. top_k IS in
# v1's list but never enters non_default_params, so it rides the extra_body
# passthrough instead (note above); stream_options is outside the IR (inbound
# fallback). tools/tool_choice/parallel_tool_calls are fc-capability-gated.
_SAMBANOVA_LIST = frozenset(
    {
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stream",
        "stop",
        "response_format",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
    }
)

_SAMBANOVA_FC_KEYS = frozenset({"tools", "tool_choice", "parallel_tool_calls"})


def supports_sambanova_tools(model: str, deps: TranslationDeps) -> bool:
    return deps.supports_capability(f"sambanova/{model}", "supports_function_calling")


def _has_non_text_content_block(request: ChatRequest) -> bool:
    return any(
        block.tag not in ("text", "tool_use", "tool_result")
        for message in request.messages
        for block in message.content
    )


def sambanova_unsupported(request: ChatRequest, deps: TranslationDeps) -> str | None:
    """SambanovaConfig._transform_messages flattens EVERY content list via
    handle_messages_with_content_list_to_str_conversion and never calls
    super(): non-text parts are silently dropped (text+image lists) or left
    un-normalized (image-only lists) — fall back so v1 serves its own lossy
    flatten; text-only lists are the serializer's flatten delta."""
    if _has_non_text_content_block(request):
        return (
            "non-text content block on sambanova: v1's content-list flatten "
            "drops non-text parts and skips the base image transforms "
            "(sambanova/chat.py _transform_messages); v1 serves it"
        )
    allowed = (
        _SAMBANOVA_LIST
        if supports_sambanova_tools(request.model, deps)
        else _SAMBANOVA_LIST - _SAMBANOVA_FC_KEYS
    )
    return unsupported_against(
        request,
        provider="sambanova",
        allowed=allowed,
        notes={
            "user": _user_note("sambanova"),
            "top_k": _top_k_extra_body_note("sambanova"),
        },
    )


# DeepInfraConfig's list (deepinfra/chat/transformation.py:61-85),
# IR-restricted; reasoning_effort capability-narrowed. tool_choice has a
# custom three-way arm in v1's map: "auto"/"none" are silently DROPPED (the
# arm never copies them — the serializer delta mirrors the drop), every other
# value RAISES UnsupportedParamsError unless drop_params.
_DEEPINFRA_LIST = frozenset(
    {
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stream",
        "stop",
        "response_format",
        "tools",
        "tool_choice",
        "reasoning_effort",
    }
)


def supports_deepinfra_reasoning(model: str, deps: TranslationDeps) -> bool:
    return deps.supports_capability(f"deepinfra/{model}", "supports_reasoning")


def deepinfra_unsupported(request: ChatRequest, deps: TranslationDeps) -> str | None:
    choice = request.tool_choice.default_value(None)
    if choice is not None and choice.tag not in ("auto", "none"):
        return (
            f"tool_choice {choice.tag!r} on deepinfra: v1's map raises "
            "UnsupportedParamsError on every value outside {auto, none} "
            "(deepinfra/chat/transformation.py map_openai_params); v1 raises"
        )
    allowed = (
        _DEEPINFRA_LIST
        if supports_deepinfra_reasoning(request.model, deps)
        else _DEEPINFRA_LIST - {"reasoning_effort"}
    )
    return unsupported_against(
        request,
        provider="deepinfra",
        allowed=allowed,
        notes={
            "user": _user_note("deepinfra"),
            "top_k": _top_k_extra_body_note("deepinfra"),
            "reasoning_effort": (
                f"reasoning_effort on non-reasoning deepinfra model "
                f"{request.model} (model-map supports_reasoning gate); "
                "v1 raises or drops it"
            ),
        },
    )


# MoonshotChatConfig inherits the BASE list minus ``functions`` (not
# IR-carried), minus tools/tool_choice for kimi-thinking-preview models
# (moonshot/chat/transformation.py:91-112). The temperature pop/clamps are
# VALUE rewrites in serialize.py; the two body-touching transforms below
# fall back so v1 serves its own rewrites.
_MOONSHOT_THINKING_PREVIEW = "kimi-thinking-preview"


def supports_moonshot_reasoning(model: str, deps: TranslationDeps) -> bool:
    return deps.supports_capability(f"moonshot/{model}", "supports_reasoning")


def _has_assistant_tool_use(request: ChatRequest) -> bool:
    return any(
        block.tag == "tool_use"
        for message in request.messages
        if message.role == "assistant"
        for block in message.content
    )


def moonshot_unsupported(request: ChatRequest, deps: TranslationDeps) -> str | None:
    choice = request.tool_choice.default_value(None)
    if choice is not None and choice.tag == "required":
        return (
            "tool_choice 'required' on moonshot: v1 pops it and APPENDS a "
            "synthetic user message ('Please select a tool to handle the "
            "current issue.', _add_tool_choice_required_message); v1 serves "
            "its rewrite"
        )
    if supports_moonshot_reasoning(request.model, deps) and _has_assistant_tool_use(
        request
    ):
        return (
            "assistant tool-call history on a moonshot reasoning model: v1's "
            "fill_reasoning_content injects reasoning_content (a single-space "
            "placeholder) into every such message; v1 serves its rewrite"
        )
    allowed = _BASE_LIST
    notes: dict[str, str] = {
        "user": _user_note("moonshot"),
        "top_k": _top_k_extra_body_note("moonshot"),
    }
    if _MOONSHOT_THINKING_PREVIEW in request.model:
        allowed = allowed - frozenset({"tools", "tool_choice"})
        notes = {
            **notes,
            "tools": (
                f"tools on {request.model}: kimi-thinking-preview models are "
                "excluded from MoonshotChatConfig's supported list; v1 raises "
                "UnsupportedParamsError (or drops under drop_params)"
            ),
            "tool_choice": (
                f"tool_choice on {request.model}: kimi-thinking-preview models "
                "are excluded from MoonshotChatConfig's supported list; v1 "
                "raises UnsupportedParamsError (or drops under drop_params)"
            ),
        }
    return unsupported_against(
        request, provider="moonshot", allowed=allowed, notes=notes
    ) or unsupported_response_format(request)


def cometapi_unsupported(request: ChatRequest, deps: TranslationDeps) -> str | None:
    """CometAPIConfig's map is super() over the plain base list (its
    extra_body stub is always empty and transform_request pops it — a no-op,
    verified at HEAD), so the gate is the base-list gate with the shared
    wave-2a top_k note."""
    return unsupported_against(
        request,
        provider="cometapi",
        allowed=_BASE_LIST,
        notes={
            "user": _user_note("cometapi"),
            "top_k": _top_k_extra_body_note("cometapi"),
        },
    ) or unsupported_response_format(request)


# The single source of per-provider supported-surface truth: each provider's
# MAXIMAL allowed set (capability and per-model gates above only ever NARROW
# it). Serialization derives user/reasoning_effort emission from membership
# here, so the gate facts and the emission facts can never desync
# (critic-wave1a M3).
ALLOWED: Mapping[CompatSdkProvider, frozenset[str]] = MappingProxyType(
    {
        "together_ai": _BASE_LIST,
        "cerebras": _CEREBRAS_LIST,
        "nvidia_nim": _NVIDIA_DEFAULT_LIST,
        "lm_studio": _BASE_LIST,
        "llamafile": _BASE_LIST,
        "lambda_ai": _BASE_LIST,
        "nebius": _BASE_LIST,
        "novita": _BASE_LIST,
        "wandb": _BASE_LIST,
        "featherless_ai": _FEATHERLESS_LIST,
        "nscale": _NSCALE_LIST,
        "hyperbolic": _HYPERBOLIC_LIST,
        "volcengine": _VOLCENGINE_LIST,
        "perplexity": _PERPLEXITY_LIST,
        "sambanova": _SAMBANOVA_LIST,
        "deepinfra": _DEEPINFRA_LIST,
        "moonshot": _BASE_LIST,
        "cometapi": _BASE_LIST,
    }
)
