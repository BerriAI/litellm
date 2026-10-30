from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import TypeAdapter

from litellm.types.utils import ProviderSpecificHeader

_PROVIDER_HEADERS: Final[TypeAdapter[ProviderSpecificHeader | Sequence[ProviderSpecificHeader] | None]] = TypeAdapter(
    ProviderSpecificHeader | Sequence[ProviderSpecificHeader] | None
)


class ProviderSpecificHeaderUtils:
    @staticmethod
    def forwarded_authorization(provider_specific_header: object, custom_llm_provider: str | None) -> str | None:
        headers: Final = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            _PROVIDER_HEADERS.validate_python(provider_specific_header), custom_llm_provider
        )
        return next((value for key, value in headers.items() if key.lower() == "authorization"), None)

    @staticmethod
    def get_provider_specific_headers(
        provider_specific_header: ProviderSpecificHeader | Sequence[ProviderSpecificHeader] | None,
        custom_llm_provider: str | None,
        native_anthropic_messages: bool = False,
    ) -> Mapping[str, str]:
        """
        Get the provider specific headers for the given custom llm provider.

        Accepts either a single ProviderSpecificHeader or a sequence of them. Each entry
        carries its own comma-separated provider list, so headers that are safe for several
        providers and headers that are safe for exactly one can travel on the same request
        without sharing a scope. Entries whose provider list does not contain
        `custom_llm_provider` contribute nothing, and entries marked
        `native_anthropic_messages_only` contribute only to a provider's native Messages route.

        Returns:
            Dict: The provider specific headers for the given custom llm provider
        """
        if provider_specific_header is None or custom_llm_provider is None:
            return {}

        scoped_headers: Final = (
            (provider_specific_header,) if isinstance(provider_specific_header, dict) else provider_specific_header
        )

        matched_headers: Final = {}
        for scoped_header in scoped_headers:
            if (
                "native_anthropic_messages_only" in scoped_header
                and scoped_header["native_anthropic_messages_only"]
                and not native_anthropic_messages
            ):
                continue
            stored_providers = scoped_header.get("custom_llm_provider", "")
            provider_list = [p.strip() for p in stored_providers.split(",")]
            if custom_llm_provider in provider_list:
                matched_headers.update(scoped_header.get("extra_headers", {}))

        return matched_headers
