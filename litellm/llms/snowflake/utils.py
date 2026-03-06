from typing import TYPE_CHECKING, Any, List, Optional, Tuple

from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.llms.base_llm.chat.transformation import BaseLLMException

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class SnowflakeException(BaseLLMException):
    """Snowflake AI Endpoints exception handling class"""

    pass


class SnowflakeBaseConfig:
    def get_supported_openai_params(self, model: str) -> List[str]:
        return [
            "temperature",
            "max_tokens",
            "top_p",
            "response_format",
            "tools",
            "tool_choice",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        If any supported_openai_params are in non_default_params, add them to optional_params, so they are used in API call

        Args:
            non_default_params (dict): Non-default parameters to filter.
            optional_params (dict): Optional parameters to update.
            model (str): Model name for parameter support check.

        Returns:
            dict: Updated optional_params with supported non-default parameters.
        """
        supported_openai_params = self.get_supported_openai_params(model)
        for param, value in non_default_params.items():
            if param in supported_openai_params:
                optional_params[param] = value
        return optional_params

    def _get_api_base(self, api_base, optional_params):
        if not api_base:
            if "account_id" in optional_params:
                account_id = optional_params.pop("account_id")
            else:
                account_id = get_secret_str("SNOWFLAKE_ACCOUNT_ID")
            if account_id is None:
                raise ValueError("Missing snowflake account_id")
            api_base = f"https://{account_id}.snowflakecomputing.com/api/v2"

        api_base = api_base.rstrip("/")
        if not api_base.endswith("/api/v2"):
            api_base += "/api/v2"
        return api_base

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Return headers to use for Snowflake completion request

        Snowflake REST API Ref: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-llm-rest-api#api-reference
        Expected headers:
        {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer " + <JWT>,
            "X-Snowflake-Authorization-Token-Type": "KEYPAIR_JWT"
        }
        """

        auth_type = "KEYPAIR_JWT"

        if api_key is None:
            raise ValueError("Missing Snowflake JWT key")
        else:
            pat_key_prefix = "pat/"
            if api_key.startswith(pat_key_prefix):
                api_key = api_key[len(pat_key_prefix) :]
                auth_type = "PROGRAMMATIC_ACCESS_TOKEN"

        headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": "Bearer " + api_key,
                "X-Snowflake-Authorization-Token-Type": auth_type,
            }
        )
        return headers

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        dynamic_api_key = api_key or get_secret_str("SNOWFLAKE_JWT")
        return api_base, dynamic_api_key
