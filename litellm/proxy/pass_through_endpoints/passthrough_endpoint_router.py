import os
from typing import Dict, Optional


class PassthroughEndpointRouter:
    """
    Use this class to Set/Get credentials for pass-through endpoints
    """

    def __init__(self):
        self.credentials: Dict[str, str] = {}

    def set_pass_through_credentials(
        self,
        custom_llm_provider: str,
        api_base: str,
        api_key: str,
        region_name: str,
    ):
        credential_name = self._get_credential_name_for_provider(
            custom_llm_provider=custom_llm_provider,
            region_name=region_name,
        )
        self.credentials[credential_name] = api_key

    def get_credentials(
        self,
        custom_llm_provider: Optional[str],
        region_name: Optional[str],
    ):
        credential_name = self._get_credential_name_for_provider(
            custom_llm_provider=custom_llm_provider,
            region_name=region_name,
        )
        if credential_name in self.credentials:
            return self.credentials[credential_name]
        else:
            _env_variable_name = (
                self._get_default_env_variable_name_passthrough_endpoint(
                    custom_llm_provider=custom_llm_provider,
                )
            )
            return os.environ[_env_variable_name]

    @staticmethod
    def _get_credential_name_for_provider(
        custom_llm_provider: Optional[str],
        region_name: Optional[str],
    ) -> str:
        if custom_llm_provider is None:
            raise ValueError(
                "custom_llm_provider is required for setting pass-through credentials"
            )
        if region_name is None:
            return f"{custom_llm_provider.upper()}_API_KEY"
        return f"{custom_llm_provider.upper()}_{region_name.upper()}_API_KEY"

    @staticmethod
    def _get_default_env_variable_name_passthrough_endpoint(
        custom_llm_provider: Optional[str],
    ) -> str:
        if custom_llm_provider is None:
            raise ValueError(
                "custom_llm_provider is required for setting pass-through credentials"
            )
        return f"{custom_llm_provider.upper()}_API_KEY"
