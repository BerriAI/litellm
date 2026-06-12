"""Parameter gates for the snowflake (Cortex REST) serializer.

v1's gate is ``_check_valid_arg`` over ``SnowflakeBaseConfig.
get_supported_openai_params`` — a SMALL static list (temperature,
max_tokens, top_p, stream, response_format, tools, tool_choice; no
capability forks, no name gates). ``max_completion_tokens`` RAISES (no
rename arm — researcher-4 confirmed), as do stop/n/seed/penalties/
parallel_tool_calls. ``user`` is silently dropped (``_check_valid_arg``
skips it and the list never carries it). snowflake is NOT in
``openai_compatible_providers``, so ``top_k`` rides the TOP-LEVEL
passthrough into the body — v1 SERVES it (wire-proven in the request
gate), so v2 emits it verbatim.
"""

from __future__ import annotations

from ...deps import TranslationDeps
from ...ir import ChatRequest
from ..compat_sdk.checks import unsupported_against, user_never_note

_SNOWFLAKE_LIST = frozenset(
    {
        "temperature",
        "max_tokens",
        "top_p",
        "stream",
        "response_format",
        "tools",
        "tool_choice",
        "top_k",
    }
)


def unsupported_params(request: ChatRequest, deps: TranslationDeps) -> str | None:
    return unsupported_against(
        request,
        provider="snowflake",
        allowed=_SNOWFLAKE_LIST,
        notes={"user": user_never_note("snowflake")},
    )
