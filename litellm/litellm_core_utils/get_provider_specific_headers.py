from collections.abc import Sequence
from typing import Final

from litellm.types.utils import ProviderSpecificHeader


class _CaseInsensitiveDict(dict):
    """A dict subclass whose __getitem__, __contains__, and .get() are case-insensitive.

    Keys are stored with their original casing so that iteration, .items(), and
    equality checks preserve the caller's format, but lookups always fold to lower-case.
    """

    def __getitem__(self, key: str) -> str:  # pyright: ignore[reportIncompatibleMethodOverride]  # narrower key type is intentional; HTTP header keys are always str
        try:
            return super().__getitem__(key)
        except KeyError:
            key_lower = key.lower()
            for k, v in self.items():
                if k.lower() == key_lower:
                    return v
            raise

    def __contains__(self, key: object) -> bool:
        if super().__contains__(key):
            return True
        if isinstance(key, str):
            key_lower = key.lower()
            return any(k.lower() == key_lower for k in self.keys())
        return False

    def get(self, key: str, default=None):  # pyright: ignore[reportIncompatibleMethodOverride]  # narrower key type is intentional; HTTP header keys are always str
        try:
            return self[key]
        except KeyError:
            return default


class ProviderSpecificHeaderUtils:
    @staticmethod
    def get_provider_specific_headers(
        provider_specific_header: ProviderSpecificHeader | Sequence[ProviderSpecificHeader] | None,
        custom_llm_provider: str | None,
    ) -> dict:
        """
        Get the provider specific headers for the given custom llm provider.

        Accepts either a single ProviderSpecificHeader or a sequence of them. Each entry
        carries its own comma-separated provider list, so headers that are safe for several
        providers and headers that are safe for exactly one can travel on the same request
        without sharing a scope. Entries whose provider list does not contain
        `custom_llm_provider` contribute nothing.

        Returns:
            CaseInsensitiveDict: The provider specific headers for the given custom llm
            provider. Key lookups are case-insensitive so that ``result["Authorization"]``,
            ``result["authorization"]``, and ``result["AUTHORIZATION"]`` all resolve
            regardless of the casing used when the header was stored.
        """
        if provider_specific_header is None or custom_llm_provider is None:
            return _CaseInsensitiveDict()

        scoped_headers: Final = (
            (provider_specific_header,) if isinstance(provider_specific_header, dict) else provider_specific_header
        )

        matched_headers: Final = _CaseInsensitiveDict()
        for scoped_header in scoped_headers:
            stored_providers = scoped_header.get("custom_llm_provider", "")
            provider_list = [p.strip() for p in stored_providers.split(",")]
            if custom_llm_provider in provider_list:
                matched_headers.update(scoped_header.get("extra_headers", {}))

        return matched_headers
