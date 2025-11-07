"""
Common utilities used across bedrock chat/embedding/image generation
"""

import json
import os
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Union

if TYPE_CHECKING:
    from litellm.types.llms.bedrock import BedrockCreateBatchRequest

import httpx

import litellm
from litellm.llms.base_llm.anthropic_messages.transformation import (
    BaseAnthropicMessagesConfig,
)
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret

if TYPE_CHECKING:
    from litellm.types.llms.openai import AllMessageValues


class BedrockError(BaseLLMException):
    pass


class AmazonBedrockGlobalConfig:
    def __init__(self):
        pass

    def get_mapped_special_auth_params(self) -> dict:
        """
        Mapping of common auth params across bedrock/vertex/azure/watsonx
        """
        return {"region_name": "aws_region_name"}

    def map_special_auth_params(self, non_default_params: dict, optional_params: dict):
        mapped_params = self.get_mapped_special_auth_params()
        for param, value in non_default_params.items():
            if param in mapped_params:
                optional_params[mapped_params[param]] = value
        return optional_params

    def get_all_regions(self) -> List[str]:
        return (
            self.get_us_regions()
            + self.get_eu_regions()
            + self.get_ap_regions()
            + self.get_ca_regions()
            + self.get_sa_regions()
        )

    def get_ap_regions(self) -> List[str]:
        """
        Source: https://www.aws-services.info/bedrock.html
        """
        return [
            "ap-northeast-1",  # Asia Pacific (Tokyo)
            "ap-northeast-2",  # Asia Pacific (Seoul)
            "ap-northeast-3",  # Asia Pacific (Osaka)
            "ap-south-1",  # Asia Pacific (Mumbai)
            "ap-south-2",  # Asia Pacific (Hyderabad)
            "ap-southeast-1",  # Asia Pacific (Singapore)
            "ap-southeast-2",  # Asia Pacific (Sydney)
        ]

    def get_sa_regions(self) -> List[str]:
        return ["sa-east-1"]

    def get_eu_regions(self) -> List[str]:
        """
        Source: https://www.aws-services.info/bedrock.html
        """
        return [
            "eu-west-1",  # Europe (Ireland)
            "eu-west-2",  # Europe (London)
            "eu-west-3",  # Europe (Paris)
            "eu-central-1",  # Europe (Frankfurt)
            "eu-central-2",  # Europe (Zurich)
            "eu-south-1",  # Europe (Milan)
            "eu-south-2",  # Europe (Spain)
            "eu-north-1",  # Europe (Stockholm)
        ]

    def get_ca_regions(self) -> List[str]:
        return ["ca-central-1"]

    def get_us_regions(self) -> List[str]:
        """
        Source: https://www.aws-services.info/bedrock.html
        """
        return [
            "us-east-1",  # US East (N. Virginia)
            "us-east-2",  # US East (Ohio)
            "us-west-1",  # US West (N. California)
            "us-west-2",  # US West (Oregon)
            "us-gov-east-1",  # AWS GovCloud (US-East)
            "us-gov-west-1",  # AWS GovCloud (US-West)
        ]


def add_custom_header(headers):
    """Closure to capture the headers and add them."""

    def callback(request, **kwargs):
        """Actual callback function that Boto3 will call."""
        for header_name, header_value in headers.items():
            request.headers.add_header(header_name, header_value)

    return callback


def init_bedrock_client(
    region_name=None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_region_name: Optional[str] = None,
    aws_bedrock_runtime_endpoint: Optional[str] = None,
    aws_session_name: Optional[str] = None,
    aws_profile_name: Optional[str] = None,
    aws_role_name: Optional[str] = None,
    aws_web_identity_token: Optional[str] = None,
    extra_headers: Optional[dict] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
):
    # check for custom AWS_REGION_NAME and use it if not passed to init_bedrock_client
    litellm_aws_region_name = get_secret("AWS_REGION_NAME", None)
    standard_aws_region_name = get_secret("AWS_REGION", None)
    ## CHECK IS  'os.environ/' passed in
    # Define the list of parameters to check
    params_to_check = [
        aws_access_key_id,
        aws_secret_access_key,
        aws_region_name,
        aws_bedrock_runtime_endpoint,
        aws_session_name,
        aws_profile_name,
        aws_role_name,
        aws_web_identity_token,
    ]

    # Iterate over parameters and update if needed
    for i, param in enumerate(params_to_check):
        if param and param.startswith("os.environ/"):
            params_to_check[i] = get_secret(param)  # type: ignore
    # Assign updated values back to parameters
    (
        aws_access_key_id,
        aws_secret_access_key,
        aws_region_name,
        aws_bedrock_runtime_endpoint,
        aws_session_name,
        aws_profile_name,
        aws_role_name,
        aws_web_identity_token,
    ) = params_to_check

    # SSL certificates (a.k.a CA bundle) used to verify the identity of requested hosts.
    ssl_verify = os.getenv("SSL_VERIFY", litellm.ssl_verify)

    ### SET REGION NAME
    if region_name:
        pass
    elif aws_region_name:
        region_name = aws_region_name
    elif litellm_aws_region_name:
        region_name = litellm_aws_region_name
    elif standard_aws_region_name:
        region_name = standard_aws_region_name
    else:
        raise BedrockError(
            message="AWS region not set: set AWS_REGION_NAME or AWS_REGION env variable or in .env file",
            status_code=401,
        )

    # check for custom AWS_BEDROCK_RUNTIME_ENDPOINT and use it if not passed to init_bedrock_client
    env_aws_bedrock_runtime_endpoint = get_secret("AWS_BEDROCK_RUNTIME_ENDPOINT")
    if aws_bedrock_runtime_endpoint:
        endpoint_url = aws_bedrock_runtime_endpoint
    elif env_aws_bedrock_runtime_endpoint:
        endpoint_url = env_aws_bedrock_runtime_endpoint
    else:
        endpoint_url = f"https://bedrock-runtime.{region_name}.amazonaws.com"

    import boto3

    if isinstance(timeout, float):
        config = boto3.session.Config(connect_timeout=timeout, read_timeout=timeout)  # type: ignore
    elif isinstance(timeout, httpx.Timeout):
        config = boto3.session.Config(  # type: ignore
            connect_timeout=timeout.connect, read_timeout=timeout.read
        )
    else:
        config = boto3.session.Config()  # type: ignore

    ### CHECK STS ###
    if (
        aws_web_identity_token is not None
        and aws_role_name is not None
        and aws_session_name is not None
    ):
        oidc_token = get_secret(aws_web_identity_token)

        if oidc_token is None:
            raise BedrockError(
                message="OIDC token could not be retrieved from secret manager.",
                status_code=401,
            )

        sts_client = boto3.client("sts")

        # https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRoleWithWebIdentity.html
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sts/client/assume_role_with_web_identity.html
        sts_response = sts_client.assume_role_with_web_identity(
            RoleArn=aws_role_name,
            RoleSessionName=aws_session_name,
            WebIdentityToken=oidc_token,
            DurationSeconds=3600,
        )

        client = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=sts_response["Credentials"]["AccessKeyId"],
            aws_secret_access_key=sts_response["Credentials"]["SecretAccessKey"],
            aws_session_token=sts_response["Credentials"]["SessionToken"],
            region_name=region_name,
            endpoint_url=endpoint_url,
            config=config,
            verify=ssl_verify,
        )
    elif aws_role_name is not None and aws_session_name is not None:
        # use sts if role name passed in
        sts_client = boto3.client(
            "sts",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            verify=ssl_verify
        )

        sts_response = sts_client.assume_role(
            RoleArn=aws_role_name, RoleSessionName=aws_session_name
        )

        client = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=sts_response["Credentials"]["AccessKeyId"],
            aws_secret_access_key=sts_response["Credentials"]["SecretAccessKey"],
            aws_session_token=sts_response["Credentials"]["SessionToken"],
            region_name=region_name,
            endpoint_url=endpoint_url,
            config=config,
            verify=ssl_verify,
        )
    elif aws_access_key_id is not None:
        # uses auth params passed to completion
        # aws_access_key_id is not None, assume user is trying to auth using litellm.completion

        client = boto3.client(
            service_name="bedrock-runtime",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name,
            endpoint_url=endpoint_url,
            config=config,
            verify=ssl_verify,
        )
    elif aws_profile_name is not None:
        # uses auth values from AWS profile usually stored in ~/.aws/credentials

        client = boto3.Session(profile_name=aws_profile_name).client(
            service_name="bedrock-runtime",
            region_name=region_name,
            endpoint_url=endpoint_url,
            config=config,
            verify=ssl_verify,
        )
    else:
        # aws_access_key_id is None, assume user is trying to auth using env variables
        # boto3 automatically reads env variables

        client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region_name,
            endpoint_url=endpoint_url,
            config=config,
            verify=ssl_verify,
        )
    if extra_headers:
        client.meta.events.register(
            "before-sign.bedrock-runtime.*", add_custom_header(extra_headers)
        )

    return client


class ModelResponseIterator:
    def __init__(self, model_response):
        self.model_response = model_response
        self.is_done = False

    # Sync iterator
    def __iter__(self):
        return self

    def __next__(self):
        if self.is_done:
            raise StopIteration
        self.is_done = True
        return self.model_response

    # Async iterator
    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.is_done:
            raise StopAsyncIteration
        self.is_done = True
        return self.model_response


def get_bedrock_tool_name(response_tool_name: str) -> str:
    """
    If litellm formatted the input tool name, we need to convert it back to the original name.

    Args:
        response_tool_name (str): The name of the tool as received from the response.

    Returns:
        str: The original name of the tool.
    """

    if response_tool_name in litellm.bedrock_tool_name_mappings.cache_dict:
        response_tool_name = litellm.bedrock_tool_name_mappings.cache_dict[
            response_tool_name
        ]
    return response_tool_name


class BedrockModelInfo(BaseLLMModelInfo):
    global_config = AmazonBedrockGlobalConfig()
    all_global_regions = global_config.get_all_regions()

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
        """
        Get the API base for the given model.
        """
        return api_base

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        """
        Get the API key for the given model.
        """
        return api_key

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List["AllMessageValues"],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        return headers

    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> List[str]:
        return []

    @staticmethod
    def extract_model_name_from_arn(model: str) -> str:
        """
        Extract the model name from an AWS Bedrock ARN.
        Returns the string after the last '/' if 'arn' is in the input string.

        Args:
            arn (str): The ARN string to parse

        Returns:
            str: The extracted model name if 'arn' is in the string,
                otherwise returns the original string
        """
        if "arn" in model.lower():
            return model.split("/")[-1]
        return model

    @staticmethod
    def get_non_litellm_routing_model_name(model: str) -> str:
        if model.startswith("bedrock/"):
            model = model.split("/", 1)[1]

        if model.startswith("converse/"):
            model = model.split("/", 1)[1]

        if model.startswith("invoke/"):
            model = model.split("/", 1)[1]

        return model

    @staticmethod
    def get_base_model(model: str) -> str:
        """
        Get the base model from the given model name.

        Handle model names like - "us.meta.llama3-2-11b-instruct-v1:0" -> "meta.llama3-2-11b-instruct-v1"
        AND "meta.llama3-2-11b-instruct-v1:0" -> "meta.llama3-2-11b-instruct-v1"
        """

        model = BedrockModelInfo.get_non_litellm_routing_model_name(model=model)
        model = BedrockModelInfo.extract_model_name_from_arn(model)

        potential_region = model.split(".", 1)[0]

        alt_potential_region = model.split("/", 1)[
            0
        ]  # in model cost map we store regional information like `/us-west-2/bedrock-model`

        if (
            potential_region
            in BedrockModelInfo._supported_cross_region_inference_region()
        ):
            return model.split(".", 1)[1]
        elif (
            alt_potential_region in BedrockModelInfo.all_global_regions
            and len(model.split("/", 1)) > 1
        ):
            return model.split("/", 1)[1]

        return model

    @staticmethod
    def _supported_cross_region_inference_region() -> List[str]:
        """
        Abbreviations of regions AWS Bedrock supports for cross region inference
        """
        return ["global", "us", "eu", "apac", "jp", "au", "us-gov"]

    @staticmethod
    def get_bedrock_route(
        model: str,
    ) -> Literal["converse", "invoke", "converse_like", "agent", "agentcore", "async_invoke"]:
        """
        Get the bedrock route for the given model.
        """
        route_mappings: Dict[
            str, Literal["invoke", "converse_like", "converse", "agent", "agentcore", "async_invoke"]
        ] = {
            "invoke/": "invoke",
            "converse_like/": "converse_like",
            "converse/": "converse",
            "agent/": "agent",
            "agentcore/": "agentcore",
            "async_invoke/": "async_invoke",
        }

        # Check explicit routes first
        for prefix, route_type in route_mappings.items():
            if prefix in model:
                return route_type

        base_model = BedrockModelInfo.get_base_model(model)
        alt_model = BedrockModelInfo.get_non_litellm_routing_model_name(model=model)
        if (
            base_model in litellm.bedrock_converse_models
            or alt_model in litellm.bedrock_converse_models
        ):
            return "converse"
        return "invoke"

    @staticmethod
    def _explicit_converse_route(model: str) -> bool:
        """
        Check if the model is an explicit converse route.
        """
        return "converse/" in model

    @staticmethod
    def _explicit_invoke_route(model: str) -> bool:
        """
        Check if the model is an explicit invoke route.
        """
        return "invoke/" in model

    @staticmethod
    def _explicit_agent_route(model: str) -> bool:
        """
        Check if the model is an explicit agent route.
        """
        return "agent/" in model

    @staticmethod
    def _explicit_agentcore_route(model: str) -> bool:
        """
        Check if the model is an explicit agentcore route.
        """
        return "agentcore/" in model

    @staticmethod
    def _explicit_converse_like_route(model: str) -> bool:
        """
        Check if the model is an explicit converse like route.
        """
        return "converse_like/" in model

    @staticmethod
    def _explicit_async_invoke_route(model: str) -> bool:
        """
        Check if the model is an explicit async invoke route.
        """
        return "async_invoke/" in model

    @staticmethod
    def get_bedrock_provider_config_for_messages_api(
        model: str,
    ) -> Optional[BaseAnthropicMessagesConfig]:
        """
        Get the bedrock provider config for the given model.

        Only route to AmazonAnthropicClaude3MessagesConfig() for BaseMessagesConfig

        All other routes should return None since they will go through litellm.completion
        """

        #########################################################
        # Converse routes should go through litellm.completion()
        if BedrockModelInfo._explicit_converse_route(model):
            return None

        #########################################################
        # This goes through litellm.AmazonAnthropicClaude3MessagesConfig()
        # Since bedrock Invoke supports Native Anthropic Messages API
        #########################################################
        if "claude" in model:
            return litellm.AmazonAnthropicClaudeMessagesConfig()

        #########################################################
        # These routes will go through litellm.completion()
        #########################################################
        return None


def get_bedrock_chat_config(model: str):
    """
    Helper function to get the appropriate Bedrock chat config based on model and route.
    
    Args:
        model: The model name/identifier
        
    Returns:
        The appropriate Bedrock config class instance
    """
    bedrock_route = BedrockModelInfo.get_bedrock_route(model)
    bedrock_invoke_provider = litellm.BedrockLLM.get_bedrock_invoke_provider(
        model=model
    )
    base_model = BedrockModelInfo.get_base_model(model)

    # Handle explicit routes first
    if bedrock_route == "converse" or bedrock_route == "converse_like":
        return litellm.AmazonConverseConfig()
    elif bedrock_route == "agent":
        from litellm.llms.bedrock.chat.invoke_agent.transformation import (
            AmazonInvokeAgentConfig,
        )
        return AmazonInvokeAgentConfig()
    elif bedrock_route == "agentcore":
        from litellm.llms.bedrock.chat.agentcore.transformation import (
            AmazonAgentCoreConfig,
        )
        return AmazonAgentCoreConfig()

    # Handle provider-specific configs
    if bedrock_invoke_provider == "amazon":
        return litellm.AmazonTitanConfig()
    elif bedrock_invoke_provider == "anthropic":
        if (
            base_model
            in litellm.AmazonAnthropicConfig.get_legacy_anthropic_model_names()
        ):
            return litellm.AmazonAnthropicConfig()
        else:
            return litellm.AmazonAnthropicClaudeConfig()
    elif bedrock_invoke_provider == "meta" or bedrock_invoke_provider == "llama":
        return litellm.AmazonLlamaConfig()
    elif bedrock_invoke_provider == "ai21":
        return litellm.AmazonAI21Config()
    elif bedrock_invoke_provider == "cohere":
        return litellm.AmazonCohereConfig()
    elif bedrock_invoke_provider == "mistral":
        return litellm.AmazonMistralConfig()
    elif bedrock_invoke_provider == "deepseek_r1":
        return litellm.AmazonDeepSeekR1Config()
    elif bedrock_invoke_provider == "nova":
        return litellm.AmazonInvokeNovaConfig()
    elif bedrock_invoke_provider == "qwen3":
        return litellm.AmazonQwen3Config()
    else:
        return litellm.AmazonInvokeConfig()


class BedrockEventStreamDecoderBase:
    """
    Base class for event stream decoding for Bedrock
    """

    _response_stream_shape_cache = None

    def __init__(self):
        from botocore.parsers import EventStreamJSONParser

        self.parser = EventStreamJSONParser()

    def get_response_stream_shape(self):
        if self._response_stream_shape_cache is None:
            from botocore.loaders import Loader
            from botocore.model import ServiceModel

            loader = Loader()
            bedrock_service_dict = loader.load_service_model(
                "bedrock-runtime", "service-2"
            )
            bedrock_service_model = ServiceModel(bedrock_service_dict)
            self._response_stream_shape_cache = bedrock_service_model.shape_for(
                "ResponseStream"
            )

        return self._response_stream_shape_cache

    def _parse_message_from_event(self, event) -> Optional[str]:
        response_dict = event.to_response_dict()
        parsed_response = self.parser.parse(
            response_dict, self.get_response_stream_shape()
        )

        if response_dict["status_code"] != 200:
            decoded_body = response_dict["body"].decode()
            if isinstance(decoded_body, dict):
                error_message = decoded_body.get("message")
            elif isinstance(decoded_body, str):
                error_message = decoded_body
            else:
                error_message = ""
            exception_status = response_dict["headers"].get(":exception-type")
            error_message = exception_status + " " + error_message
            raise BedrockError(
                status_code=response_dict["status_code"],
                message=(
                    json.dumps(error_message)
                    if isinstance(error_message, dict)
                    else error_message
                ),
            )
        if "chunk" in parsed_response:
            chunk = parsed_response.get("chunk")
            if not chunk:
                return None
            return chunk.get("bytes").decode()  # type: ignore[no-any-return]
        else:
            chunk = response_dict.get("body")
            if not chunk:
                return None

            return chunk.decode()  # type: ignore[no-any-return]


def get_anthropic_beta_from_headers(headers: dict) -> List[str]:
    """
    Extract anthropic-beta header values and convert them to a list.
    Supports comma-separated values from user headers.

    Used by both converse and invoke transformations for consistent handling
    of anthropic-beta headers that should be passed to AWS Bedrock.

    Args:
        headers (dict): Request headers dictionary

    Returns:
        List[str]: List of anthropic beta feature strings, empty list if no header
    """
    anthropic_beta_header = headers.get("anthropic-beta")
    if not anthropic_beta_header:
        return []

    # Split comma-separated values and strip whitespace
    return [beta.strip() for beta in anthropic_beta_header.split(",")]


class CommonBatchFilesUtils:
    """
    Common utilities for Bedrock batch and file operations.
    Provides shared functionality to reduce code duplication between batches and files.
    """

    def __init__(self):
        # Import here to avoid circular imports
        from .base_aws_llm import BaseAWSLLM

        self._base_aws = BaseAWSLLM()

    def get_bedrock_model_id_from_litellm_model(self, model: str) -> str:
        """
        Extract the actual Bedrock model ID from LiteLLM model name.

        Args:
            model: LiteLLM model name (e.g., "bedrock/anthropic.claude-3-sonnet-20240229-v1:0")

        Returns:
            Bedrock model ID (e.g., "anthropic.claude-3-sonnet-20240229-v1:0")
        """
        if model.startswith("bedrock/"):
            return model[8:]  # Remove "bedrock/" prefix
        return model

    def parse_s3_uri(self, s3_uri: str) -> tuple:
        """
        Parse S3 URI into bucket and key components.

        Args:
            s3_uri: S3 URI (e.g., "s3://bucket/key/path")

        Returns:
            Tuple of (bucket, key)

        Raises:
            ValueError: If URI format is invalid
        """
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Invalid S3 URI format: {s3_uri}")

        s3_parts = s3_uri[5:].split("/", 1)  # Remove "s3://" and split on first "/"
        if len(s3_parts) != 2:
            raise ValueError(f"Invalid S3 URI format: {s3_uri}")

        return s3_parts[0], s3_parts[1]  # bucket, key

    def extract_model_from_s3_file_path(
        self, s3_uri: str, optional_params: dict
    ) -> str:
        """
        Extract model ID from S3 file path.

        The Bedrock file transformation creates S3 objects with the model name embedded:
        Format: s3://bucket/litellm-bedrock-files-{model}-{uuid}.jsonl
        """
        # Check if model is provided in optional_params first
        if "model" in optional_params and optional_params["model"]:
            return self.get_bedrock_model_id_from_litellm_model(
                optional_params["model"]
            )

        # Extract model from S3 URI path
        # Expected format: s3://bucket/litellm-bedrock-files-{model}-{uuid}.jsonl
        try:
            bucket, object_key = self.parse_s3_uri(s3_uri)

            # Extract model from object key if it follows our naming pattern
            if object_key.startswith("litellm-bedrock-files-"):
                # Remove prefix and suffix to get model part
                model_part = object_key[22:]  # Remove "litellm-bedrock-files-"
                # Find the last dash before the UUID
                parts = model_part.split("-")
                if len(parts) > 1:
                    # Reconstruct model name (everything except the last UUID part and .jsonl)
                    model_name = "-".join(parts[:-1])
                    if model_name.endswith(".jsonl"):
                        model_name = model_name[:-6]  # Remove .jsonl
                    return model_name
        except Exception:
            pass

        # Fallback to default model
        return "anthropic.claude-3-5-sonnet-20240620-v1:0"

    def sign_aws_request(
        self,
        service_name: str,
        data: Union[str, dict, "BedrockCreateBatchRequest"],
        endpoint_url: str,
        optional_params: dict,
        method: str = "POST",
    ) -> tuple:
        """
        Sign AWS request using Signature Version 4.

        Args:
            service_name: AWS service name ("bedrock" or "s3")
            data: Request data (string or dict)
            endpoint_url: Full endpoint URL
            optional_params: Optional parameters containing AWS credentials
            method: HTTP method (default: POST)

        Returns:
            Tuple of (signed_headers, signed_data)
        """
        try:
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

        # Get AWS credentials using existing methods
        aws_region_name = self._base_aws._get_aws_region_name(
            optional_params=optional_params, model=""
        )
        credentials = self._base_aws.get_credentials(
            aws_access_key_id=optional_params.get("aws_access_key_id"),
            aws_secret_access_key=optional_params.get("aws_secret_access_key"),
            aws_session_token=optional_params.get("aws_session_token"),
            aws_region_name=aws_region_name,
            aws_session_name=optional_params.get("aws_session_name"),
            aws_profile_name=optional_params.get("aws_profile_name"),
            aws_role_name=optional_params.get("aws_role_name"),
            aws_web_identity_token=optional_params.get("aws_web_identity_token"),
            aws_sts_endpoint=optional_params.get("aws_sts_endpoint"),
        )

        # Prepare the request data
        method_upper = method.upper()
        if method_upper == "GET":
            # GET requests should be signed with an empty payload
            request_data = ""
            headers = {}
        else:
            if isinstance(data, dict):
                import json

                request_data = json.dumps(data)
            else:
                request_data = data
            # Prepare headers for non-GET requests
            headers = {"Content-Type": "application/json"}

        # Create AWS request and sign it
        sigv4 = SigV4Auth(credentials, service_name, aws_region_name)
        request = AWSRequest(
            method=method_upper, url=endpoint_url, data=request_data, headers=headers
        )
        sigv4.add_auth(request)
        prepped = request.prepare()

        return (
            dict(prepped.headers),
            request_data.encode("utf-8")
            if isinstance(request_data, str)
            else request_data,
        )

    def generate_unique_job_name(self, model: str, prefix: str = "litellm") -> str:
        """
        Generate a unique job name for AWS services.
        AWS services often have length limits, so this creates a concise name.

        Args:
            model: Model name to include in the job name
            prefix: Prefix for the job name

        Returns:
            Unique job name (≤ 63 characters for Bedrock compatibility)
        """
        from litellm._uuid import uuid

        unique_id = str(uuid.uuid4())[:8]
        # Format: {prefix}-batch-{model}-{uuid}
        # Example: litellm-batch-claude-266c398e
        job_name = f"{prefix}-batch-{unique_id}"

        return job_name

    def get_s3_bucket_and_key_from_config(
        self,
        litellm_params: dict,
        optional_params: dict,
        bucket_env_var: str = "AWS_S3_BUCKET_NAME",
        key_prefix: str = "litellm",
    ) -> tuple:
        """
        Get S3 bucket and generate a unique key from configuration.

        Args:
            litellm_params: LiteLLM parameters
            optional_params: Optional parameters
            bucket_env_var: Environment variable name for bucket
            key_prefix: Prefix for the S3 key

        Returns:
            Tuple of (bucket_name, object_key)
        """
        import time

        from litellm._uuid import uuid

        # Get bucket name
        bucket_name = (
            litellm_params.get("s3_bucket_name")
            or optional_params.get("s3_bucket_name")
            or os.getenv(bucket_env_var)
        )
        if not bucket_name:
            raise ValueError(
                f"S3 bucket name is required. Set 's3_bucket_name' parameter or {bucket_env_var} env var"
            )

        # Generate unique object key
        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        object_key = f"{key_prefix}-{timestamp}-{unique_id}"

        return bucket_name, object_key

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[Dict, httpx.Headers]
    ) -> BaseLLMException:
        """
        Get Bedrock-specific error class.
        """
        return BedrockError(
            status_code=status_code, message=error_message, headers=headers
        )
