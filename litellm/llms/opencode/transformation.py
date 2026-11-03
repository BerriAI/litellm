from litellm.llms.opencode.common_utils import is_opencode_api_base


class OpenCodeProviderAdapter:
    def get_provider_affinity_header(self, api_base: str | None) -> str | None:
        return "x-opencode-session" if is_opencode_api_base(api_base) else None
