from bisect import bisect_left
from collections.abc import Mapping, Sequence
from itertools import chain, groupby
from types import MappingProxyType
from typing import Final, TypeVar

from pydantic import TypeAdapter

from litellm.types.llms.anthropic import ANTHROPIC_USER_PROFILE_ID_HEADER
from litellm.types.utils import LlmProviders, ProviderSpecificHeader

_PROVIDER_HEADERS: Final[TypeAdapter[ProviderSpecificHeader | Sequence[ProviderSpecificHeader] | None]] = TypeAdapter(
    ProviderSpecificHeader | Sequence[ProviderSpecificHeader] | None
)

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
    native_anthropic_messages: bool,
    model: str | None,
) -> bool:
    if scoped_header.get("native_anthropic_messages_only") and not native_anthropic_messages:
        return False
    if scoped_header.get("claude_models_only") and not names_a_claude_model(model):
        return False
    if scoped_header.get("non_claude_models_only") and names_a_claude_model(model):
        return False
    scoped_providers: Final = scoped_header.get("custom_llm_provider", "").split(",")
    return custom_llm_provider in (provider.strip() for provider in scoped_providers)


def _sorted_text_values_by_name(headers: Mapping[str, object]) -> Mapping[str, tuple[str, ...]]:
    pairs: Final = sorted((name.lower(), value if isinstance(value, str) else "") for name, value in headers.items())
    return MappingProxyType(
        {name: tuple(value for _, value in group) for name, group in groupby(pairs, key=lambda pair: pair[0])}
    )


def _carries(values_by_name: Mapping[str, tuple[str, ...]], name: str, prefix: str) -> bool:
    values: Final = values_by_name.get(name, ())
    position: Final = bisect_left(values, prefix)
    return position < len(values) and values[position].startswith(prefix)


def _configures_a_listed_credential(
    scoped_header: ProviderSpecificHeader, configured_values: Mapping[str, tuple[str, ...]]
) -> bool:
    credential_headers: Final = frozenset(
        name.lower() for name in scoped_header.get("configured_credential_headers", ())
    )
    prefix: Final = scoped_header.get("configured_credential_prefix", "")
    return not credential_headers or any(_carries(configured_values, name, prefix) for name in credential_headers)


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
    def forwarded_authorization(
        provider_specific_header: object, custom_llm_provider: str | None, model: str | None
    ) -> str | None:
        headers: Final = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            _PROVIDER_HEADERS.validate_python(provider_specific_header), custom_llm_provider, model=model
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
        Entries marked `claude_models_only` contribute only when `model` names a Claude model, entries marked
        `non_claude_models_only` only when it does not, and
        entries marked `yields_to_configured` never replace a header already in `configured_headers`, and
        entries listing `configured_credential_headers` contribute only when `configured_headers` carries one of them,
        whose value starts with `configured_credential_prefix` when the entry sets one, under a name no other
        applicable entry replaces.

        Returns:
            Dict: The provider specific headers for the given custom llm provider
        """
        if provider_specific_header is None or custom_llm_provider is None:
            return {}

        scoped_headers: Final = (
            (provider_specific_header,) if isinstance(provider_specific_header, dict) else provider_specific_header
        )

        configured: Final[Mapping[str, object]] = configured_headers or {}
        configured_names: Final = frozenset(name.lower() for name in configured)
        applicable: Final = tuple(
            scoped_header
            for scoped_header in scoped_headers
            if _applies_to_call(scoped_header, custom_llm_provider, native_anthropic_messages, model)
        )
        replaced_names: Final = frozenset(
            chain.from_iterable(
                _contributed_headers(scoped_header, configured_names)
                for scoped_header in applicable
                if not scoped_header.get("configured_credential_headers")
            )
        )
        kept_configured: Final = _sorted_text_values_by_name(
            {name: value for name, value in configured.items() if name not in replaced_names}
        )
        return dict(
            chain.from_iterable(
                _contributed_headers(scoped_header, configured_names).items()
                for scoped_header in applicable
                if _configures_a_listed_credential(scoped_header, kept_configured)
            )
        )
