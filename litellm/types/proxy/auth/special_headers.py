import enum


class SpecialHeaders(enum.Enum):
    """Used by user_api_key_auth.py to get litellm key"""

    openai_authorization = "Authorization"
    azure_authorization = "API-Key"
    anthropic_authorization = "x-api-key"
    google_ai_studio_authorization = "x-goog-api-key"
    azure_apim_authorization = "Ocp-Apim-Subscription-Key"
    custom_litellm_api_key = "x-litellm-api-key"
    mcp_auth = "x-mcp-auth"
    mcp_servers = "x-mcp-servers"
    mcp_access_groups = "x-mcp-access-groups"

    @classmethod
    def litellm_credential_header_names(cls) -> "frozenset[str]":
        """Lowercased header names user_api_key_auth accepts as a litellm key.

        Every header here authenticates the caller, so any code that forwards a
        request onward (e.g. the plugin reverse proxy) must strip all of them to
        avoid leaking the caller's litellm credential downstream. The static
        custom-key header (general_settings.litellm_key_header_name) is runtime
        config and must be added on top of this set by the caller.
        """
        return frozenset(
            header.value.lower()
            for header in (
                cls.openai_authorization,
                cls.azure_authorization,
                cls.anthropic_authorization,
                cls.google_ai_studio_authorization,
                cls.azure_apim_authorization,
                cls.custom_litellm_api_key,
            )
        )
