"""
Directory-search provider abstraction.

Lets `internal_user_endpoints.py` (GET /user/directory_search) and
`ui_sso.py` (the directory-search-enabled flag in /sso/get_ui_settings) look
up a directory user without knowing which identity provider backs it. Add a
new provider (Okta, Google Workspace, generic SAML/SCIM, ...) by implementing
`DirectorySearchProvider` in a sibling module and adding an instance to
`_providers()` below - no caller changes needed.
"""

from typing import Protocol

from litellm.types.proxy.management_endpoints.internal_user_endpoints import DirectoryUser

# Shared across providers so a stray 1-2 character query never reaches any
# directory backend, regardless of which provider is configured.
DIRECTORY_SEARCH_MIN_QUERY_LENGTH = 2


class DirectorySearchProvider(Protocol):
    def is_configured(self) -> bool:
        """Whether this provider has the config/credentials it needs to search."""
        ...

    async def search(self, query: str) -> tuple[DirectoryUser, ...]:
        """Search this provider's directory. Only called when `is_configured()` is True."""
        ...


def _providers() -> tuple[DirectorySearchProvider, ...]:
    from litellm.proxy.management_endpoints.directory_search.microsoft import (
        MicrosoftDirectorySearchProvider,
    )

    return (MicrosoftDirectorySearchProvider(),)


def get_configured_directory_search_provider() -> DirectorySearchProvider | None:
    """Returns the first configured provider, or None if none are configured."""
    return next((provider for provider in _providers() if provider.is_configured()), None)
