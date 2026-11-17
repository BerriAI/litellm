from collections.abc import Mapping, Sequence
from itertools import chain
from typing import Final, TypeVar

from pydantic import TypeAdapter

from litellm.types.utils import ProviderSpecificHeader

_PROVIDER_HEADERS: Final[TypeAdapter[ProviderSpecificHeader | Sequence[ProviderSpecificHeader] | None]] = TypeAdapter(
    ProviderSpecificHeader | Sequence[ProviderSpecificHeader] | None
)

_HeaderValue: Final = TypeVar("_HeaderValue")
_REPEATABLE_HEADERS: Final = frozenset({"anthropic-beta"})


def override_headers(
    headers: Mapping[str, _HeaderValue], overrides: Mapping[str, _HeaderValue]
) -> dict[str, _HeaderValue]:  # mutable-ok: legacy callers pass the merged headers on as a dict
    overridden: Final = frozenset(name.lower() for name in overrides) - _REPEATABLE_HEADERS
    return {**{name: value for name, value in headers.items() if name.lower() not in overridden}, **overrides}


def _names_a_claude_model(model: str | None) -> bool:
    return model is not None and "claude" in model.lower()


def _applies_to_call(
    scoped_header: ProviderSpecificHeader,
    custom_llm_provider: str,
    native_anthropic_messages: bool,
    model: str | None,
) -> bool:
    if scoped_header.get("native_anthropic_messages_only") and not native_anthropic_messages:
        return False
    if scoped_header.get("claude_models_only") and not _names_a_claude_model(model):
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
    def validated_provider_specific_headers(
        provider_specific_header: object, custom_llm_provider: str | None
    ) -> Mapping[str, str]:
        return ProviderSpecificHeaderUtils.get_provider_specific_headers(
            _PROVIDER_HEADERS.validate_python(provider_specific_header), custom_llm_provider
        )

    @staticmethod
    def forwarded_authorization(provider_specific_header: object, custom_llm_provider: str | None) -> str | None:
        headers: Final = ProviderSpecificHeaderUtils.validated_provider_specific_headers(
            provider_specific_header, custom_llm_provider
        )
        return next((value for key, value in headers.items() if key.lower() == "authorization"), None)

    @staticmethod
    def get_provider_specific_headers(
        provider_specific_header: ProviderSpecificHeader | Sequence[ProviderSpecificHeader] | None,
        custom_llm_provider: str | None,
        native_anthropic_messages: bool = False,
        model: str | None = None,
        configured_headers: Mapping[str, object] | None = None,
    ) -> Mapping[str, str]:
        """
        Get the provider specific headers for the given custom llm provider.

        Accepts either a single ProviderSpecificHeader or a sequence of them. Each entry
        carries its own comma-separated provider list, so headers that are safe for several
        providers and headers that are safe for exactly one can travel on the same request
        without sharing a scope. Entries whose provider list does not contain
        `custom_llm_provider` contribute nothing, and entries marked
        `native_anthropic_messages_only` contribute only to a provider's native Messages route.
        Entries marked `claude_models_only` contribute only when `model` names a Claude model, and
        entries marked `yields_to_configured` never replace a header already in `configured_headers`.

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
            if _applies_to_call(scoped_header, custom_llm_provider, native_anthropic_messages, model)
        )
        return dict(
            chain.from_iterable(
                _contributed_headers(scoped_header, configured_names).items() for scoped_header in applicable
            )
        )
