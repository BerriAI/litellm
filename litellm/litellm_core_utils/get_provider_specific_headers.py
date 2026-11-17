from collections.abc import Mapping, Sequence
from itertools import chain
from typing import Final, TypeVar

from litellm.types.llms.anthropic import ANTHROPIC_USER_PROFILE_ID_HEADER
from litellm.types.utils import LlmProviders, ProviderSpecificHeader

_HeaderValue: Final = TypeVar("_HeaderValue")
_CASE_INSENSITIVE_OVERRIDES: Final = frozenset({ANTHROPIC_USER_PROFILE_ID_HEADER})


def override_headers(
    headers: Mapping[str, _HeaderValue], overrides: Mapping[str, _HeaderValue]
) -> dict[str, _HeaderValue]:  # mutable-ok: legacy callers pass the merged headers on as a dict
    overridden: Final = frozenset(name.lower() for name in overrides) & _CASE_INSENSITIVE_OVERRIDES
    return {**{name: value for name, value in headers.items() if name.lower() not in overridden}, **overrides}


ANTHROPIC_USER_PROFILE_ID_PROVIDERS: Final = frozenset(
    (
        LlmProviders.ANTHROPIC.value,
        LlmProviders.AZURE_AI.value,
        LlmProviders.BEDROCK.value,
        LlmProviders.BEDROCK_MANTLE.value,
        LlmProviders.VERTEX_AI.value,
    )
)


def names_a_claude_model(model: str | None) -> bool:
    return model is not None and "claude" in model.lower()


def _applies_to_call(
    scoped_header: ProviderSpecificHeader,
    custom_llm_provider: str,
    model: str | None,
) -> bool:
    if scoped_header.get("claude_models_only") and not names_a_claude_model(model):
        return False
    scoped_providers: Final = scoped_header.get("custom_llm_provider", "").split(",")
    return custom_llm_provider in (provider.strip() for provider in scoped_providers)


def _contributed_headers(scoped_header: ProviderSpecificHeader, configured_names: frozenset[str]) -> Mapping[str, str]:
    headers: Final[Mapping[str, str]] = scoped_header.get("extra_headers", {})
    if not scoped_header.get("yields_to_configured"):
        return headers
    return {name: value for name, value in headers.items() if name.lower() not in configured_names}


class ProviderSpecificHeaderUtils:
    @staticmethod
    def get_provider_specific_headers(
        provider_specific_header: ProviderSpecificHeader | Sequence[ProviderSpecificHeader] | None,
        custom_llm_provider: str | None,
        model: str | None = None,
        configured_headers: Mapping[str, object] | None = None,
    ) -> Mapping[str, str]:
        """
        Get the provider specific headers for the given custom llm provider.

        Accepts either a single ProviderSpecificHeader or a sequence of them. Each entry
        carries its own comma-separated provider list, so headers that are safe for several
        providers and headers that are safe for exactly one can travel on the same request
        without sharing a scope. Entries whose provider list does not contain
        `custom_llm_provider` contribute nothing. Entries marked `claude_models_only` contribute only
        when `model` names a Claude model, and entries marked `yields_to_configured` never replace a
        header already in `configured_headers`.

        Returns:
            Dict: The provider specific headers for the given custom llm provider
        """
        if provider_specific_header is None or custom_llm_provider is None:
            return {}

        scoped_headers: Final = (
            (provider_specific_header,) if isinstance(provider_specific_header, dict) else provider_specific_header
        )

        configured_names: Final = frozenset(name.lower() for name in configured_headers or {})
        applicable: Final = (
            scoped_header
            for scoped_header in scoped_headers
            if _applies_to_call(scoped_header, custom_llm_provider, model)
        )
        return dict(
            chain.from_iterable(
                _contributed_headers(scoped_header, configured_names).items() for scoped_header in applicable
            )
        )
