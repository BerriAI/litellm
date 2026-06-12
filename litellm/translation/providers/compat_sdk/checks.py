"""The generic supported-list checker shared by BOTH compat families.

v1's gate for every openai-compat family provider is ``_check_valid_arg``
over the provider config's ``get_supported_openai_params``: an unsupported
param RAISES ``UnsupportedParamsError`` unless ``drop_params``, in which case
it is popped BEFORE ``map_openai_params`` runs. ``unsupported_against``
mirrors that SUPPORTED-LIST truth as typed fallbacks (the v2-openai/xai
precedent — never re-implement the raise-vs-drop interplay): every IR-carried
param a provider's allowed set excludes falls back so v1 serves its own raise
or drop. Params outside the IR (seed, penalties, logprobs, n, stream_options,
...) already fall back at the inbound boundary and never reach this checker.

This module is pure machinery plus the one cross-family data constant
(``BASE_LIST``); the per-provider allowed sets and gates live in each
family's params.py (compat_sdk and compat_httpx both import from here, so
the cross-family dependency carries an honest name — critic-wave1b M3).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

from ...deps import TranslationDeps
from ...ir import ChatRequest
from ..openai_compat.params import unsupported_response_format

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
            # non_default_params, so _check_valid_arg never sees it (no
            # raise, even without drop_params — verifier-wave1a F6's half of
            # the story). It is NOT dropped, though: get_optional_params
            # packs it into extra_body (the provider-specific passthrough)
            # and the SDK/handler merges extra_body top-level, so v1 SERVES
            # it. Re-verified in-process at HEAD for ALL 18 family
            # providers; critic-wave2a M1 made this verified truth the
            # default arm instead of the transform-output-only "silently
            # drops" reading.
            return (
                f"top_k on {provider}: not an OpenAI param; v1's "
                "get_optional_params packs it into extra_body and the "
                "SDK/handler merges it top-level — that crossing is "
                "unported, v1 serves it"
            )
        return (
            f"{key} on {provider}: outside v1's supported list; "
            "get_optional_params raises UnsupportedParamsError "
            "(or drops it under drop_params)"
        )
    return None


def user_note(provider: str) -> str:
    return (
        f"user on {provider}: gated on litellm.open_ai_chat_completion_models "
        "membership in v1's base supported list; v1 handles it"
    )


def user_never_note(provider: str) -> str:
    return (
        f"user on {provider}: v1's _check_valid_arg always skips user and this "
        "config's own supported list never carries it; v1 silently drops it"
    )


# OpenAIGPTConfig's base list restricted to IR-carried keys. ``user`` is
# deliberately absent (model-list gated in v1) and ``response_format`` rides
# the base list's gpt-4/gpt-3.5-turbo-16k name gate, applied per provider
# by the configs that inherit the base list.
BASE_LIST = frozenset(
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


def base_list_unsupported(
    request: ChatRequest, deps: TranslationDeps, provider: str
) -> str | None:
    """The plain base-list gate (compat_sdk llamafile / novita / lm_studio
    and the rename-only rows; compat_httpx's default gate)."""
    return unsupported_against(
        request,
        provider=provider,
        allowed=BASE_LIST,
        notes={"user": user_note(provider)},
    ) or unsupported_response_format(request)
