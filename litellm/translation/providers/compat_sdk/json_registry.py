"""The providers.json registry mechanism (wave-1b's 14 enum members).

Co-locates the registry truths the mirror test reconciles (critic-wave1b
M3): the function-calling capability fork and the ``param_mappings``
mct -> max_tokens rename set. Membership itself
(``params.JSON_REGISTRY_PROVIDERS``) stays beside ``ALLOWED`` in the
family's params.py — it feeds the ALLOWED table and the provider Literal,
and duplicating the 14 names into a second Literal here would mint a drift
surface. ``test_json_rename_set_mirrors_providers_json_at_head`` re-derives
ALL of it (membership == registry ∩ enum, JSON_RENAME == the param_mappings
rows, constraints/special_handling pinned absent) from providers.json at
HEAD.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...deps import TranslationDeps
from ...ir import ChatRequest
from ..openai_compat.params import unsupported_response_format
from .checks import BASE_LIST, unsupported_against, user_note

if TYPE_CHECKING:
    from .params import CompatSdkProvider

_JSON_TOOL_KEYS = frozenset({"tools", "tool_choice", "parallel_tool_calls"})


def supports_json_provider_tools(
    model: str, deps: TranslationDeps, provider: CompatSdkProvider
) -> bool:
    return deps.supports_capability(f"{provider}/{model}", "supports_function_calling")


def json_registry_unsupported(
    request: ChatRequest, deps: TranslationDeps, provider: CompatSdkProvider
) -> str | None:
    """The dynamic ``JSONProviderConfig`` (openai_like/dynamic_config.py):
    base GPT list minus tools/tool_choice/parallel_tool_calls unless
    ``supports_function_calling(model, slug)`` — membership-guarded
    ``.remove``, so together_ai's gpt-4-name ValueError corner does NOT
    apply here; ``response_format`` stays (only the base list's own
    gpt-4/gpt-3.5-turbo-16k name gate removes it)."""
    allowed = (
        BASE_LIST
        if supports_json_provider_tools(request.model, deps, provider)
        else BASE_LIST - _JSON_TOOL_KEYS
    )
    return unsupported_against(
        request,
        provider=provider,
        allowed=allowed,
        notes={"user": user_note(provider)},
    ) or unsupported_response_format(request)


# providers.json ``param_mappings`` carrying max_completion_tokens ->
# max_tokens at HEAD (the mapping arm runs FIRST in JSONProviderConfig.
# map_openai_params, before the supported-list copy loop); the drift gate
# re-derives this set from providers.json in the request differential.
JSON_RENAME = frozenset(
    {
        "publicai",
        "xiaomi_mimo",
        "synthetic",
        "apertis",
        "nano-gpt",
        "poe",
        "chutes",
        "charity_engine",
        "neosantara",
        "tensormesh",
    }
)
