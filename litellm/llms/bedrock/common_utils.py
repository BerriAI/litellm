"""
Common utilities used across bedrock chat/embedding/image generation
"""

import os
from typing import List, Literal, Optional, Union

import httpx

import litellm
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret


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
        return ["ap-northeast-1", "ap-northeast-2", "ap-northeast-3", "ap-south-1"]

    def get_sa_regions(self) -> List[str]:
        return ["sa-east-1"]

    def get_eu_regions(self) -> List[str]:
        """
        Source: https://www.aws-services.info/bedrock.html
        """
        return [
            "eu-west-1",
            "eu-west-2",
            "eu-west-3",
            "eu-central-1",
        ]

    def get_ca_regions(self) -> List[str]:
        return ["ca-central-1"]

    def get_us_regions(self) -> List[str]:
        """
        Source: https://www.aws-services.info/bedrock.html
        """
        return [
            "us-east-2",
            "us-east-1",
            "us-west-1",
            "us-west-2",
            "us-gov-west-1",
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
    def get_base_model(model: str) -> str:
        """
        Get the base model from the given model name.

        Handle model names like - "us.meta.llama3-2-11b-instruct-v1:0" -> "meta.llama3-2-11b-instruct-v1"
        AND "meta.llama3-2-11b-instruct-v1:0" -> "meta.llama3-2-11b-instruct-v1"
        """
        if model.startswith("bedrock/"):
            model = model.split("/", 1)[1]

        if model.startswith("converse/"):
            model = model.split("/", 1)[1]

        if model.startswith("invoke/"):
            model = model.split("/", 1)[1]

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
        return ["us", "eu", "apac"]

    @staticmethod
    def get_bedrock_route(model: str) -> Literal["converse", "invoke", "converse_like"]:
        """
        Get the bedrock route for the given model.
        """
        base_model = BedrockModelInfo.get_base_model(model)
        if "invoke/" in model:
            return "invoke"
        elif "converse_like" in model:
            return "converse_like"
        elif "converse/" in model:
            return "converse"
        elif base_model in litellm.bedrock_converse_models:
            return "converse"
        return "invoke"
