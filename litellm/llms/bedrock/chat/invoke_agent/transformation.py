"""
Transformation for Bedrock Invoke Agent

https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_InvokeAgent.html
"""

from typing import List, Optional

from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM


class AmazonInvokeAgentConfig(BaseConfig, BaseAWSLLM):
    def __init__(self, **kwargs):
        BaseConfig.__init__(self, **kwargs)
        BaseAWSLLM.__init__(self, **kwargs)

    def get_supported_openai_params(self, model: str) -> List[str]:
        """
        This is a base invoke model mapping. For Invoke - define a bedrock provider specific config that extends this class.
        """
        return [
            "max_tokens",
            "max_completion_tokens",
            "stream",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        This is a base invoke model mapping. For Invoke - define a bedrock provider specific config that extends this class.
        """
        for param, value in non_default_params.items():
            if param == "max_tokens" or param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            if param == "stream":
                optional_params["stream"] = value
        return optional_params

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete url for the request
        """
        ### SET RUNTIME ENDPOINT ###
        aws_bedrock_runtime_endpoint = optional_params.get(
            "aws_bedrock_runtime_endpoint", None
        )  # https://bedrock-runtime.{region_name}.amazonaws.com
        endpoint_url, _ = self.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=aws_bedrock_runtime_endpoint,
            aws_region_name=self._get_aws_region_name(
                optional_params=optional_params, model=model
            ),
            endpoint_type="agent",
        )

        agentAliasID = optional_params.get("agentAliasID", None)
        sessionID = optional_params.get("sessionID", None)

        endpoint_url = f"{endpoint_url}/agents/{model}/agentAliases/{agentAliasID}/sessions/{sessionID}/text"

        return endpoint_url
