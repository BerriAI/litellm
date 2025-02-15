from typing import Dict, Optional

from litellm._logging import verbose_logger
from litellm.secret_managers.main import get_secret_str


class PassthroughEndpointRouter:
    """
    Use this class to Set/Get credentials for pass-through endpoints
    """

    def __init__(self):
        self.credentials: Dict[str, str] = {}

    def set_pass_through_credentials(
        self,
        custom_llm_provider: str,
        api_base: Optional[str],
        api_key: Optional[str],
    ):
        """
        Set credentials for a pass-through endpoint. Used when a user adds a pass-through LLM endpoint on the UI.

        Args:
            custom_llm_provider: The provider of the pass-through endpoint
            api_base: The base URL of the pass-through endpoint
            api_key: The API key for the pass-through endpoint
        """
        credential_name = self._get_credential_name_for_provider(
            custom_llm_provider=custom_llm_provider,
            region_name=self._get_region_name_from_api_base(
                api_base=api_base, custom_llm_provider=custom_llm_provider
            ),
        )
        if api_key is None:
            raise ValueError("api_key is required for setting pass-through credentials")
        self.credentials[credential_name] = api_key

    def get_credentials(
        self,
        custom_llm_provider: str,
        region_name: Optional[str],
    ) -> Optional[str]:
        credential_name = self._get_credential_name_for_provider(
            custom_llm_provider=custom_llm_provider,
            region_name=region_name,
        )
        verbose_logger.debug(
            f"Pass-through llm endpoints router, looking for credentials for {credential_name}"
        )
        if credential_name in self.credentials:
            verbose_logger.debug(f"Found credentials for {credential_name}")
            return self.credentials[credential_name]
        else:
            verbose_logger.debug(
                f"No credentials found for {credential_name}, looking for env variable"
            )
            _env_variable_name = (
                self._get_default_env_variable_name_passthrough_endpoint(
                    custom_llm_provider=custom_llm_provider,
                )
            )
            return get_secret_str(_env_variable_name)

    def _get_credential_name_for_provider(
        self,
        custom_llm_provider: str,
        region_name: Optional[str],
    ) -> str:
        if region_name is None:
            return f"{custom_llm_provider.upper()}_API_KEY"
        return f"{custom_llm_provider.upper()}_{region_name.upper()}_API_KEY"

    def _get_region_name_from_api_base(
        self,
        custom_llm_provider: str,
        api_base: Optional[str],
    ) -> Optional[str]:
        """
        Get the region name from the API base.

        Each provider might have a different way of specifying the region in the API base - this is where you can use conditional logic to handle that.
        """
        if custom_llm_provider == "assemblyai":
            if api_base and "eu" in api_base:
                return "eu"
        return None

    @staticmethod
    def _get_default_env_variable_name_passthrough_endpoint(
        custom_llm_provider: str,
    ) -> str:
        return f"{custom_llm_provider.upper()}_API_KEY"
