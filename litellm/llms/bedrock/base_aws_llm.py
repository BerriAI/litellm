import hashlib
import json
import os
import urllib.parse
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    cast,
    get_args,
)

import httpx
from pydantic import BaseModel

from litellm._logging import verbose_logger
from litellm.caching.caching import DualCache
from litellm.constants import (
    BEDROCK_EMBEDDING_PROVIDERS_LITERAL,
    BEDROCK_INVOKE_PROVIDERS_LITERAL,
    BEDROCK_MAX_POLICY_SIZE,
)
from litellm.litellm_core_utils.dd_tracing import tracer
from litellm.secret_managers.main import get_secret, get_secret_str

if TYPE_CHECKING:
    from botocore.awsrequest import AWSPreparedRequest
    from botocore.credentials import Credentials
else:
    Credentials = Any
    AWSPreparedRequest = Any


class Boto3CredentialsInfo(BaseModel):
    credentials: Credentials
    aws_region_name: str
    aws_bedrock_runtime_endpoint: Optional[str]


class AwsAuthError(Exception):
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        self.request = httpx.Request(
            method="POST", url="https://us-west-2.console.aws.amazon.com/bedrock"
        )
        self.response = httpx.Response(status_code=status_code, request=self.request)
        super().__init__(
            self.message
        )  # Call the base class constructor with the parameters it needs


class BaseAWSLLM:
    def __init__(self) -> None:
        self.iam_cache = DualCache()
        super().__init__()
        self.aws_authentication_params = [
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
            "aws_region_name",
            "aws_session_name",
            "aws_profile_name",
            "aws_role_name",
            "aws_web_identity_token",
            "aws_sts_endpoint",
            "aws_bedrock_runtime_endpoint",
            "aws_external_id",
        ]

    def get_cache_key(self, credential_args: Dict[str, Optional[str]]) -> str:
        """
        Generate a unique cache key based on the credential arguments.
        """
        # Convert credential arguments to a JSON string and hash it to create a unique key
        credential_str = json.dumps(credential_args, sort_keys=True)
        return hashlib.sha256(credential_str.encode()).hexdigest()

    @tracer.wrap()
    def get_credentials(
        self,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_session_token: Optional[str] = None,
        aws_region_name: Optional[str] = None,
        aws_session_name: Optional[str] = None,
        aws_profile_name: Optional[str] = None,
        aws_role_name: Optional[str] = None,
        aws_web_identity_token: Optional[str] = None,
        aws_sts_endpoint: Optional[str] = None,
        aws_external_id: Optional[str] = None,
    ):
        """
        Return a boto3.Credentials object
        """
        ## CHECK IS  'os.environ/' passed in
        params_to_check: List[Optional[str]] = [
            aws_access_key_id,
            aws_secret_access_key,
            aws_session_token,
            aws_region_name,
            aws_session_name,
            aws_profile_name,
            aws_role_name,
            aws_web_identity_token,
            aws_sts_endpoint,
            aws_external_id,
        ]

        # Iterate over parameters and update if needed
        for i, param in enumerate(params_to_check):
            if param and param.startswith("os.environ/"):
                _v = get_secret(param)
                if _v is not None and isinstance(_v, str):
                    params_to_check[i] = _v
            elif param is None:  # check if uppercase value in env
                key = self.aws_authentication_params[i]
                if key.upper() in os.environ:
                    params_to_check[i] = os.getenv(key.upper())

        # Assign updated values back to parameters
        (
            aws_access_key_id,
            aws_secret_access_key,
            aws_session_token,
            aws_region_name,
            aws_session_name,
            aws_profile_name,
            aws_role_name,
            aws_web_identity_token,
            aws_sts_endpoint,
            aws_external_id,
        ) = params_to_check

        verbose_logger.debug(
            "in get credentials\n"
            "aws_access_key_id=%s\n"
            "aws_secret_access_key=%s\n"
            "aws_session_token=%s\n"
            "aws_region_name=%s\n"
            "aws_session_name=%s\n"
            "aws_profile_name=%s\n"
            "aws_role_name=%s\n"
            "aws_web_identity_token=%s\n"
            "aws_sts_endpoint=%s\n"
            "aws_external_id=%s",
            aws_access_key_id,
            aws_secret_access_key,
            aws_session_token,
            aws_region_name,
            aws_session_name,
            aws_profile_name,
            aws_role_name,
            aws_web_identity_token,
            aws_sts_endpoint,
            aws_external_id,
        )

        # create cache key for non-expiring auth flows
        args = {k: v for k, v in locals().items() if k.startswith("aws_")}

        cache_key = self.get_cache_key(args)
        _cached_credentials = self.iam_cache.get_cache(cache_key)
        if _cached_credentials:
            return _cached_credentials

        #########################################################
        # Handle diff boto3 auth flows
        # for each helper
        # Return:
        #   Credentials - boto3.Credentials
        #   cache ttl - Optional[int]. If None, the credentials are not cached. Some auth flows have no expiry time.
        #########################################################
        if (
            aws_web_identity_token is not None
            and aws_role_name is not None
            and aws_session_name is not None
        ):
            credentials, _cache_ttl = self._auth_with_web_identity_token(
                aws_web_identity_token=aws_web_identity_token,
                aws_role_name=aws_role_name,
                aws_session_name=aws_session_name,
                aws_region_name=aws_region_name,
                aws_sts_endpoint=aws_sts_endpoint,
                aws_external_id=aws_external_id,
            )
        elif aws_role_name is not None:
            # Check if we're in IRSA and trying to assume the same role we already have
            current_role_arn = os.getenv("AWS_ROLE_ARN")
            web_identity_token_file = os.getenv("AWS_WEB_IDENTITY_TOKEN_FILE")

            # In IRSA environments, we should skip role assumption if we're already running as the target role
            # This is true when:
            # 1. We have AWS_ROLE_ARN set (current role)
            # 2. We have AWS_WEB_IDENTITY_TOKEN_FILE set (IRSA environment)
            # 3. The current role matches the requested role
            if (
                current_role_arn
                and web_identity_token_file
                and current_role_arn == aws_role_name
            ):
                verbose_logger.debug(
                    "Using IRSA same-role optimization: calling _auth_with_env_vars"
                )
                # We're already running as this role via IRSA, no need to assume it again
                # Use the default boto3 credentials (which will use the IRSA credentials)
                credentials, _cache_ttl = self._auth_with_env_vars()
            else:
                verbose_logger.debug(
                    "Using role assumption: calling _auth_with_aws_role"
                )
                # If aws_session_name is not provided, generate a default one
                if aws_session_name is None:
                    aws_session_name = (
                        f"litellm-session-{int(datetime.now().timestamp())}"
                    )
                credentials, _cache_ttl = self._auth_with_aws_role(
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    aws_session_token=aws_session_token,
                    aws_role_name=aws_role_name,
                    aws_session_name=aws_session_name,
                    aws_external_id=aws_external_id,
                )

        elif aws_profile_name is not None:  ### CHECK SESSION ###
            credentials, _cache_ttl = self._auth_with_aws_profile(aws_profile_name)
        elif (
            aws_access_key_id is not None
            and aws_secret_access_key is not None
            and aws_session_token is not None
        ):
            credentials, _cache_ttl = self._auth_with_aws_session_token(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
            )
        elif (
            aws_access_key_id is not None
            and aws_secret_access_key is not None
            and aws_region_name is not None
        ):
            credentials, _cache_ttl = self._auth_with_access_key_and_secret_key(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_region_name=aws_region_name,
            )
        else:
            credentials, _cache_ttl = self._auth_with_env_vars()

        self.iam_cache.set_cache(cache_key, credentials, ttl=_cache_ttl)
        return credentials

    def _get_aws_region_from_model_arn(self, model: Optional[str]) -> Optional[str]:
        try:
            # First check if the string contains the expected prefix
            if not isinstance(model, str) or "arn:aws:bedrock" not in model:
                return None

            # Split the ARN and check if we have enough parts
            parts = model.split(":")
            if len(parts) < 4:
                return None

            # Get the region from the correct position
            region = parts[3]
            if not region:  # Check if region is empty
                return None

            return region
        except Exception:
            # Catch any unexpected errors and return None
            return None

    @staticmethod
    def _get_provider_from_model_path(
        model_path: str,
    ) -> Optional[BEDROCK_INVOKE_PROVIDERS_LITERAL]:
        """
        Helper function to get the provider from a model path with format: provider/model-name

        Args:
            model_path (str): The model path (e.g., 'llama/arn:aws:bedrock:us-east-1:086734376398:imported-model/r4c4kewx2s0n' or 'anthropic/model-name')

        Returns:
            Optional[str]: The provider name, or None if no valid provider found
        """
        parts = model_path.split("/")
        if len(parts) >= 1:
            provider = parts[0]
            if provider in get_args(BEDROCK_INVOKE_PROVIDERS_LITERAL):
                return cast(BEDROCK_INVOKE_PROVIDERS_LITERAL, provider)
        return None

    @staticmethod
    def get_bedrock_invoke_provider(
        model: str,
    ) -> Optional[BEDROCK_INVOKE_PROVIDERS_LITERAL]:
        """
        Helper function to get the bedrock provider from the model

        handles 3 scenarions:
        1. model=invoke/anthropic.claude-3-5-sonnet-20240620-v1:0 -> Returns `anthropic`
        2. model=anthropic.claude-3-5-sonnet-20240620-v1:0 -> Returns `anthropic`
        3. model=llama/arn:aws:bedrock:us-east-1:086734376398:imported-model/r4c4kewx2s0n -> Returns `llama`
        4. model=us.amazon.nova-pro-v1:0 -> Returns `nova`
        """
        if model.startswith("invoke/"):
            model = model.replace("invoke/", "", 1)

        _split_model = model.split(".")[0]
        if _split_model in get_args(BEDROCK_INVOKE_PROVIDERS_LITERAL):
            return cast(BEDROCK_INVOKE_PROVIDERS_LITERAL, _split_model)

        # If not a known provider, check for pattern with two slashes
        provider = BaseAWSLLM._get_provider_from_model_path(model)
        if provider is not None:
            return provider

        # check if provider == "nova"
        if "nova" in model:
            return "nova"
        else:
            for provider in get_args(BEDROCK_INVOKE_PROVIDERS_LITERAL):
                if provider in model:
                    return provider
        return None

    @staticmethod
    def get_bedrock_model_id(
        optional_params: dict,
        provider: Optional[BEDROCK_INVOKE_PROVIDERS_LITERAL],
        model: str,
    ) -> str:
        model_id = optional_params.pop("model_id", None)
        if model_id is not None:
            model_id = BaseAWSLLM.encode_model_id(model_id=model_id)
        else:
            model_id = model

        model_id = model_id.replace("invoke/", "", 1)
        if provider == "llama" and "llama/" in model_id:
            model_id = BaseAWSLLM._get_model_id_from_model_with_spec(
                model_id, spec="llama"
            )
        elif provider == "deepseek_r1" and "deepseek_r1/" in model_id:
            model_id = BaseAWSLLM._get_model_id_from_model_with_spec(
                model_id, spec="deepseek_r1"
            )
        elif provider == "openai" and "openai/" in model_id:
            model_id = BaseAWSLLM._get_model_id_from_model_with_spec(
                model_id, spec="openai"
            )
        return model_id

    @staticmethod
    def _get_model_id_from_model_with_spec(
        model: str,
        spec: str,
    ) -> str:
        """
        Remove `llama` from modelID since `llama` is simply a spec to follow for custom bedrock models
        """
        model_id = model.replace(spec + "/", "")
        return BaseAWSLLM.encode_model_id(model_id=model_id)

    @staticmethod
    def encode_model_id(model_id: str) -> str:
        """
        Double encode the model ID to ensure it matches the expected double-encoded format.
        Args:
            model_id (str): The model ID to encode.
        Returns:
            str: The double-encoded model ID.
        """
        return urllib.parse.quote(model_id, safe="")

    @staticmethod
    def get_bedrock_embedding_provider(
        model: str,
    ) -> Optional[BEDROCK_EMBEDDING_PROVIDERS_LITERAL]:
        """
        Helper function to get the bedrock embedding provider from the model

        Handles scenarios like:
        1. model=cohere.embed-english-v3:0 -> Returns `cohere`
        2. model=amazon.titan-embed-text-v1 -> Returns `amazon`
        3. model=amazon.nova-2-multimodal-embeddings-v1:0 -> Returns `nova`
        4. model=us.twelvelabs.marengo-embed-2-7-v1:0 -> Returns `twelvelabs`
        5. model=twelvelabs.marengo-embed-2-7-v1:0 -> Returns `twelvelabs`
        """
        # Special case: Check for "nova" in model name first (before "amazon")
        # This handles amazon.nova-* models
        if "nova" in model.lower():
            if "nova" in get_args(BEDROCK_EMBEDDING_PROVIDERS_LITERAL):
                return cast(BEDROCK_EMBEDDING_PROVIDERS_LITERAL, "nova")
        
        # Handle regional models like us.twelvelabs.marengo-embed-2-7-v1:0
        if "." in model:
            parts = model.split(".")
            # Check if the second part (after potential region) is a known provider
            if len(parts) >= 2:
                potential_provider = parts[
                    1
                ]  # e.g., "twelvelabs" from "us.twelvelabs.marengo-embed-2-7-v1:0"
                if potential_provider in get_args(BEDROCK_EMBEDDING_PROVIDERS_LITERAL):
                    return cast(BEDROCK_EMBEDDING_PROVIDERS_LITERAL, potential_provider)

            # Check if the first part is a known provider (standard format)
            potential_provider = parts[
                0
            ]  # e.g., "cohere" from "cohere.embed-english-v3:0"
            if potential_provider in get_args(BEDROCK_EMBEDDING_PROVIDERS_LITERAL):
                return cast(BEDROCK_EMBEDDING_PROVIDERS_LITERAL, potential_provider)

        # Fallback: check if any provider name appears in the model string
        for provider in get_args(BEDROCK_EMBEDDING_PROVIDERS_LITERAL):
            if provider in model:
                return cast(BEDROCK_EMBEDDING_PROVIDERS_LITERAL, provider)

        return None

    def _get_aws_region_name(
        self,
        optional_params: dict,
        model: Optional[str] = None,
        model_id: Optional[str] = None,
    ) -> str:
        """
        Get the AWS region name from the environment variables.

        Parameters:
            optional_params (dict): Optional parameters for the model call
            model (str): The model name
            model_id (str): The model ID. This is the ARN of the model, if passed in as a separate param.

        Returns:
            str: The AWS region name
        """
        aws_region_name = optional_params.get("aws_region_name", None)
        ### SET REGION NAME ###
        if aws_region_name is None:
            # check model arn #
            if model_id is not None:
                aws_region_name = self._get_aws_region_from_model_arn(model_id)
            else:
                aws_region_name = self._get_aws_region_from_model_arn(model)
            # check env #
            litellm_aws_region_name = get_secret("AWS_REGION_NAME", None)

            if (
                aws_region_name is None
                and litellm_aws_region_name is not None
                and isinstance(litellm_aws_region_name, str)
            ):
                aws_region_name = litellm_aws_region_name

            standard_aws_region_name = get_secret("AWS_REGION", None)
            if (
                aws_region_name is None
                and standard_aws_region_name is not None
                and isinstance(standard_aws_region_name, str)
            ):
                aws_region_name = standard_aws_region_name
        if aws_region_name is None:
            try:
                import boto3

                with tracer.trace("boto3.Session()"):
                    session = boto3.Session()
                configured_region = session.region_name
                if configured_region:
                    aws_region_name = configured_region
                else:
                    aws_region_name = "us-west-2"
            except Exception:
                aws_region_name = "us-west-2"

        return aws_region_name

    def get_aws_region_name_for_non_llm_api_calls(
        self,
        aws_region_name: Optional[str] = None,
    ):
        """
        Get the AWS region name for non-llm api calls.

        LLM API calls check the model arn and end up using that as the region name.

        For non-llm api calls eg. Guardrails, Vector Stores we just need to check the dynamic param or env vars.
        """
        if aws_region_name is None:
            # check env #
            litellm_aws_region_name = get_secret("AWS_REGION_NAME", None)

            if litellm_aws_region_name is not None and isinstance(
                litellm_aws_region_name, str
            ):
                aws_region_name = litellm_aws_region_name

            standard_aws_region_name = get_secret("AWS_REGION", None)
            if standard_aws_region_name is not None and isinstance(
                standard_aws_region_name, str
            ):
                aws_region_name = standard_aws_region_name

            if aws_region_name is None:
                aws_region_name = "us-west-2"
        return aws_region_name

    @tracer.wrap()
    def _auth_with_web_identity_token(
        self,
        aws_web_identity_token: str,
        aws_role_name: str,
        aws_session_name: str,
        aws_region_name: Optional[str],
        aws_sts_endpoint: Optional[str],
        aws_external_id: Optional[str] = None,
    ) -> Tuple[Credentials, Optional[int]]:
        """
        Authenticate with AWS Web Identity Token
        """
        import boto3

        verbose_logger.debug(
            f"IN Web Identity Token: {aws_web_identity_token} | Role Name: {aws_role_name} | Session Name: {aws_session_name}"
        )

        if aws_sts_endpoint is None:
            sts_endpoint = f"https://sts.{aws_region_name}.amazonaws.com"
        else:
            sts_endpoint = aws_sts_endpoint

        oidc_token = get_secret(aws_web_identity_token)

        if oidc_token is None:
            raise AwsAuthError(
                message="OIDC token could not be retrieved from secret manager.",
                status_code=401,
            )

        with tracer.trace("boto3.client(sts)"):
            sts_client = boto3.client(
                "sts",
                region_name=aws_region_name,
                endpoint_url=sts_endpoint,
            )

        # https://docs.aws.amazon.com/STS/latest/APIReference/API_AssumeRoleWithWebIdentity.html
        # https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sts/client/assume_role_with_web_identity.html
        assume_role_params = {
            "RoleArn": aws_role_name,
            "RoleSessionName": aws_session_name,
            "WebIdentityToken": oidc_token,
            "DurationSeconds": 3600,
            "Policy": '{"Version":"2012-10-17","Statement":[{"Sid":"BedrockLiteLLM","Effect":"Allow","Action":["bedrock:InvokeModel","bedrock:InvokeModelWithResponseStream"],"Resource":"*","Condition":{"Bool":{"aws:SecureTransport":"true"},"StringLike":{"aws:UserAgent":"litellm/*"}}}]}',
        }

        # Add ExternalId parameter if provided
        if aws_external_id is not None:
            assume_role_params["ExternalId"] = aws_external_id

        sts_response = sts_client.assume_role_with_web_identity(**assume_role_params)

        iam_creds_dict = {
            "aws_access_key_id": sts_response["Credentials"]["AccessKeyId"],
            "aws_secret_access_key": sts_response["Credentials"]["SecretAccessKey"],
            "aws_session_token": sts_response["Credentials"]["SessionToken"],
            "region_name": aws_region_name,
        }

        if sts_response["PackedPolicySize"] > BEDROCK_MAX_POLICY_SIZE:
            verbose_logger.warning(
                f"The policy size is greater than 75% of the allowed size, PackedPolicySize: {sts_response['PackedPolicySize']}"
            )

        with tracer.trace("boto3.Session(**iam_creds_dict)"):
            session = boto3.Session(**iam_creds_dict)

        iam_creds = session.get_credentials()
        return iam_creds, self._get_default_ttl_for_boto3_credentials()

    def _handle_irsa_cross_account(
        self,
        irsa_role_arn: str,
        aws_role_name: str,
        aws_session_name: str,
        region: str,
        web_identity_token_file: str,
        aws_external_id: Optional[str] = None,
    ) -> dict:
        """Handle cross-account role assumption for IRSA."""
        import boto3

        verbose_logger.debug("Cross-account role assumption detected")

        # Read the web identity token
        with open(web_identity_token_file, "r") as f:
            web_identity_token = f.read().strip()

        # Create an STS client without credentials
        with tracer.trace("boto3.client(sts) for manual IRSA"):
            sts_client = boto3.client("sts", region_name=region)

        # Manually assume the IRSA role with the session name
        verbose_logger.debug(
            f"Manually assuming IRSA role {irsa_role_arn} with session {aws_session_name}"
        )
        irsa_response = sts_client.assume_role_with_web_identity(
            RoleArn=irsa_role_arn,
            RoleSessionName=aws_session_name,
            WebIdentityToken=web_identity_token,
        )

        # Extract the credentials from the IRSA assumption
        irsa_creds = irsa_response["Credentials"]

        # Create a new STS client with the IRSA credentials
        with tracer.trace("boto3.client(sts) with manual IRSA credentials"):
            sts_client_with_creds = boto3.client(
                "sts",
                region_name=region,
                aws_access_key_id=irsa_creds["AccessKeyId"],
                aws_secret_access_key=irsa_creds["SecretAccessKey"],
                aws_session_token=irsa_creds["SessionToken"],
            )

        # Get current caller identity for debugging
        try:
            caller_identity = sts_client_with_creds.get_caller_identity()
            verbose_logger.debug(
                f"Current identity after manual IRSA assumption: {caller_identity.get('Arn', 'unknown')}"
            )
        except Exception as e:
            verbose_logger.debug(f"Failed to get caller identity: {e}")

        # Now assume the target role
        verbose_logger.debug(
            f"Attempting to assume target role: {aws_role_name} with session: {aws_session_name}"
        )
        assume_role_params = {
            "RoleArn": aws_role_name,
            "RoleSessionName": aws_session_name,
        }

        # Add ExternalId parameter if provided
        if aws_external_id is not None:
            assume_role_params["ExternalId"] = aws_external_id

        return sts_client_with_creds.assume_role(**assume_role_params)

    def _handle_irsa_same_account(
        self,
        aws_role_name: str,
        aws_session_name: str,
        region: str,
        aws_external_id: Optional[str] = None,
    ) -> dict:
        """Handle same-account role assumption for IRSA."""
        import boto3

        verbose_logger.debug("Same account role assumption, using automatic IRSA")
        with tracer.trace("boto3.client(sts) with automatic IRSA"):
            sts_client = boto3.client("sts", region_name=region)

        # Get current caller identity for debugging
        try:
            caller_identity = sts_client.get_caller_identity()
            verbose_logger.debug(
                f"Current IRSA identity: {caller_identity.get('Arn', 'unknown')}"
            )
        except Exception as e:
            verbose_logger.debug(f"Failed to get caller identity: {e}")

        # Assume the role
        verbose_logger.debug(
            f"Attempting to assume role: {aws_role_name} with session: {aws_session_name}"
        )
        assume_role_params = {
            "RoleArn": aws_role_name,
            "RoleSessionName": aws_session_name,
        }

        # Add ExternalId parameter if provided
        if aws_external_id is not None:
            assume_role_params["ExternalId"] = aws_external_id

        return sts_client.assume_role(**assume_role_params)

    def _extract_credentials_and_ttl(
        self, sts_response: dict
    ) -> Tuple[Credentials, Optional[int]]:
        """Extract credentials and TTL from STS response."""
        from botocore.credentials import Credentials

        sts_credentials = sts_response["Credentials"]
        credentials = Credentials(
            access_key=sts_credentials["AccessKeyId"],
            secret_key=sts_credentials["SecretAccessKey"],
            token=sts_credentials["SessionToken"],
        )

        expiration_time = sts_credentials["Expiration"]
        ttl = int(
            (expiration_time - datetime.now(expiration_time.tzinfo)).total_seconds()
        )

        return credentials, ttl

    @tracer.wrap()
    def _auth_with_aws_role(
        self,
        aws_access_key_id: Optional[str],
        aws_secret_access_key: Optional[str],
        aws_session_token: Optional[str],
        aws_role_name: str,
        aws_session_name: str,
        aws_external_id: Optional[str] = None,
    ) -> Tuple[Credentials, Optional[int]]:
        """
        Authenticate with AWS Role
        """
        import boto3
        from botocore.credentials import Credentials

        # Check if we're in an EKS/IRSA environment
        web_identity_token_file = os.getenv("AWS_WEB_IDENTITY_TOKEN_FILE")
        irsa_role_arn = os.getenv("AWS_ROLE_ARN")

        # If we have IRSA environment variables and no explicit credentials,
        # we need to use the web identity token flow
        if (
            web_identity_token_file
            and irsa_role_arn
            and aws_access_key_id is None
            and aws_secret_access_key is None
        ):
            # For cross-account role assumption with specific session names,
            # we need to manually assume the IRSA role first with the correct session name
            verbose_logger.debug(
                f"IRSA detected: using web identity token from {web_identity_token_file}"
            )

            try:
                # Get region from environment
                region = (
                    os.getenv("AWS_REGION")
                    or os.getenv("AWS_DEFAULT_REGION")
                    or "us-east-1"
                )

                # Check if we need to do cross-account role assumption
                if aws_role_name != irsa_role_arn:
                    sts_response = self._handle_irsa_cross_account(
                        irsa_role_arn,
                        aws_role_name,
                        aws_session_name,
                        region,
                        web_identity_token_file,
                        aws_external_id,
                    )
                else:
                    sts_response = self._handle_irsa_same_account(
                        aws_role_name, aws_session_name, region, aws_external_id
                    )

                return self._extract_credentials_and_ttl(sts_response)

            except Exception as e:
                verbose_logger.debug(f"Failed to assume role via IRSA: {e}")
                if "AccessDenied" in str(
                    e
                ) and "is not authorized to perform: sts:AssumeRole" in str(e):
                    # Provide a more helpful error message for trust policy issues
                    verbose_logger.error(
                        f"Access denied when trying to assume role {aws_role_name}. "
                        f"Please ensure the trust policy of {aws_role_name} allows "
                        f"the current role to assume it. Current identity: check logs with verbose mode."
                    )
                # Re-raise the exception instead of falling through
                raise

        # In EKS/IRSA environments, use ambient credentials (no explicit keys needed)
        # This allows the web identity token to work automatically
        if aws_access_key_id is None and aws_secret_access_key is None:
            with tracer.trace("boto3.client(sts)"):
                sts_client = boto3.client("sts")
        else:
            with tracer.trace("boto3.client(sts)"):
                sts_client = boto3.client(
                    "sts",
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    aws_session_token=aws_session_token,
                )

        assume_role_params = {
            "RoleArn": aws_role_name,
            "RoleSessionName": aws_session_name,
        }

        # Add ExternalId parameter if provided
        if aws_external_id is not None:
            assume_role_params["ExternalId"] = aws_external_id

        sts_response = sts_client.assume_role(**assume_role_params)

        # Extract the credentials from the response and convert to Session Credentials
        sts_credentials = sts_response["Credentials"]
        credentials = Credentials(
            access_key=sts_credentials["AccessKeyId"],
            secret_key=sts_credentials["SecretAccessKey"],
            token=sts_credentials["SessionToken"],
        )

        sts_expiry = sts_credentials["Expiration"]
        # Convert to timezone-aware datetime for comparison
        current_time = datetime.now(sts_expiry.tzinfo)
        sts_ttl = (sts_expiry - current_time).total_seconds() - 60
        return credentials, sts_ttl

    @tracer.wrap()
    def _auth_with_aws_profile(
        self, aws_profile_name: str
    ) -> Tuple[Credentials, Optional[int]]:
        """
        Authenticate with AWS profile
        """
        import boto3

        # uses auth values from AWS profile usually stored in ~/.aws/credentials
        with tracer.trace("boto3.Session(profile_name=aws_profile_name)"):
            client = boto3.Session(profile_name=aws_profile_name)
            return client.get_credentials(), None

    @tracer.wrap()
    def _auth_with_aws_session_token(
        self,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        aws_session_token: str,
    ) -> Tuple[Credentials, Optional[int]]:
        """
        Authenticate with AWS Session Token
        """
        ### CHECK FOR AWS SESSION TOKEN ###
        from botocore.credentials import Credentials

        credentials = Credentials(
            access_key=aws_access_key_id,
            secret_key=aws_secret_access_key,
            token=aws_session_token,
        )

        return credentials, None

    @tracer.wrap()
    def _auth_with_access_key_and_secret_key(
        self,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        aws_region_name: Optional[str],
    ) -> Tuple[Credentials, Optional[int]]:
        """
        Authenticate with AWS Access Key and Secret Key
        """
        import boto3

        # Check if credentials are already in cache. These credentials have no expiry time.
        with tracer.trace(
            "boto3.Session(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=aws_region_name)"
        ):
            session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=aws_region_name,
            )

        credentials = session.get_credentials()
        return credentials, self._get_default_ttl_for_boto3_credentials()

    @tracer.wrap()
    def _auth_with_env_vars(self) -> Tuple[Credentials, Optional[int]]:
        """
        Authenticate with AWS Environment Variables
        """
        import boto3

        with tracer.trace("boto3.Session()"):
            session = boto3.Session()
            credentials = session.get_credentials()
            return credentials, None

    @tracer.wrap()
    def _get_default_ttl_for_boto3_credentials(self) -> int:
        """
        Get the default TTL for boto3 credentials

        Returns `3600-60` which is 59 minutes
        """
        return 3600 - 60

    def get_runtime_endpoint(
        self,
        api_base: Optional[str],
        aws_bedrock_runtime_endpoint: Optional[str],
        aws_region_name: str,
        endpoint_type: Optional[Literal["runtime", "agent", "agentcore"]] = "runtime",
    ) -> Tuple[str, str]:
        env_aws_bedrock_runtime_endpoint = get_secret("AWS_BEDROCK_RUNTIME_ENDPOINT")
        if api_base is not None:
            endpoint_url = api_base
        elif aws_bedrock_runtime_endpoint is not None and isinstance(
            aws_bedrock_runtime_endpoint, str
        ):
            endpoint_url = aws_bedrock_runtime_endpoint
        elif env_aws_bedrock_runtime_endpoint and isinstance(
            env_aws_bedrock_runtime_endpoint, str
        ):
            endpoint_url = env_aws_bedrock_runtime_endpoint
        else:
            endpoint_url = self._select_default_endpoint_url(
                endpoint_type=endpoint_type,
                aws_region_name=aws_region_name,
            )

        # Determine proxy_endpoint_url
        if aws_bedrock_runtime_endpoint is not None and isinstance(
            aws_bedrock_runtime_endpoint, str
        ):
            proxy_endpoint_url = aws_bedrock_runtime_endpoint
        elif env_aws_bedrock_runtime_endpoint and isinstance(
            env_aws_bedrock_runtime_endpoint, str
        ):
            proxy_endpoint_url = env_aws_bedrock_runtime_endpoint
        else:
            proxy_endpoint_url = endpoint_url

        return endpoint_url, proxy_endpoint_url

    def _select_default_endpoint_url(
        self, endpoint_type: Optional[Literal["runtime", "agent", "agentcore"]], aws_region_name: str
    ) -> str:
        """
        Select the default endpoint url based on the endpoint type

        Default endpoint url is https://bedrock-runtime.{aws_region_name}.amazonaws.com
        """
        if endpoint_type == "agent":
            return f"https://bedrock-agent-runtime.{aws_region_name}.amazonaws.com"
        elif endpoint_type == "agentcore":
            return f"https://bedrock-agentcore.{aws_region_name}.amazonaws.com"
        else:
            return f"https://bedrock-runtime.{aws_region_name}.amazonaws.com"

    def _get_boto_credentials_from_optional_params(
        self, optional_params: dict, model: Optional[str] = None
    ) -> Boto3CredentialsInfo:
        """
        Get boto3 credentials from optional params

        Args:
            optional_params (dict): Optional parameters for the model call

        Returns:
            Credentials: Boto3 credentials object
        """
        try:
            from botocore.credentials import Credentials
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")
        ## CREDENTIALS ##
        # pop aws_secret_access_key, aws_access_key_id, aws_region_name from kwargs, since completion calls fail with them
        aws_secret_access_key = optional_params.pop("aws_secret_access_key", None)
        aws_access_key_id = optional_params.pop("aws_access_key_id", None)
        aws_session_token = optional_params.pop("aws_session_token", None)
        aws_region_name = self._get_aws_region_name(optional_params, model)
        optional_params.pop("aws_region_name", None)
        aws_role_name = optional_params.pop("aws_role_name", None)
        aws_session_name = optional_params.pop("aws_session_name", None)
        aws_profile_name = optional_params.pop("aws_profile_name", None)
        aws_web_identity_token = optional_params.pop("aws_web_identity_token", None)
        aws_sts_endpoint = optional_params.pop("aws_sts_endpoint", None)
        aws_bedrock_runtime_endpoint = optional_params.pop(
            "aws_bedrock_runtime_endpoint", None
        )  # https://bedrock-runtime.{region_name}.amazonaws.com
        aws_external_id = optional_params.pop("aws_external_id", None)

        credentials: Credentials = self.get_credentials(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            aws_region_name=aws_region_name,
            aws_session_name=aws_session_name,
            aws_profile_name=aws_profile_name,
            aws_role_name=aws_role_name,
            aws_web_identity_token=aws_web_identity_token,
            aws_sts_endpoint=aws_sts_endpoint,
            aws_external_id=aws_external_id,
        )

        return Boto3CredentialsInfo(
            credentials=credentials,
            aws_region_name=aws_region_name,
            aws_bedrock_runtime_endpoint=aws_bedrock_runtime_endpoint,
        )

    @tracer.wrap()
    def get_request_headers(
        self,
        credentials: Credentials,
        aws_region_name: str,
        extra_headers: Optional[dict],
        endpoint_url: str,
        data: Union[str, bytes],
        headers: dict,
        api_key: Optional[str] = None,
    ) -> AWSPreparedRequest:
        if api_key is not None:
            aws_bearer_token: Optional[str] = api_key
        else:
            aws_bearer_token = get_secret_str("AWS_BEARER_TOKEN_BEDROCK")

        if aws_bearer_token:
            try:
                from botocore.awsrequest import AWSRequest
            except ImportError:
                raise ImportError(
                    "Missing boto3 to call bedrock. Run 'pip install boto3'."
                )
            headers["Authorization"] = f"Bearer {aws_bearer_token}"
            request = AWSRequest(
                method="POST", url=endpoint_url, data=data, headers=headers
            )
        else:
            try:
                from botocore.auth import SigV4Auth
                from botocore.awsrequest import AWSRequest
            except ImportError:
                raise ImportError(
                    "Missing boto3 to call bedrock. Run 'pip install boto3'."
                )

            # Filter headers for AWS signature calculation
            # AWS SigV4 only includes specific headers in signature calculation
            aws_signature_headers = self._filter_headers_for_aws_signature(headers)
            sigv4 = SigV4Auth(credentials, "bedrock", aws_region_name)
            request = AWSRequest(
                method="POST",
                url=endpoint_url,
                data=data,
                headers=aws_signature_headers,
            )
            sigv4.add_auth(request)

            # Add back all original headers (including forwarded ones) after signature calculation
            for header_name, header_value in headers.items():
                request.headers[header_name] = header_value

            if (
                extra_headers is not None and "Authorization" in extra_headers
            ):  # prevent sigv4 from overwriting the auth header
                request.headers["Authorization"] = extra_headers["Authorization"]
        prepped = request.prepare()

        return prepped

    def _filter_headers_for_aws_signature(self, headers: dict) -> dict:
        """
        Filter headers to only include those that AWS SigV4 includes in signature calculation.
        This Fixes forwarded client headers from breaking the signature calculation.
        """
        aws_signature_headers = {}
        aws_headers = {
            "host",
            "content-type",
            "date",
            "x-amz-date",
            "x-amz-security-token",
            "x-amz-content-sha256",
            "x-amz-algorithm",
            "x-amz-credential",
            "x-amz-signedheaders",
            "x-amz-signature",
        }

        for header_name, header_value in headers.items():
            header_lower = header_name.lower()
            if (
                header_lower in aws_headers
                or header_lower.startswith("x-amz-")
                or header_lower.startswith("x-amzn-")
            ):
                aws_signature_headers[header_name] = header_value

        return aws_signature_headers

    def _sign_request(
        self,
        service_name: Literal["bedrock", "sagemaker", "bedrock-agentcore"],
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        fake_stream: Optional[bool] = None,
        api_key: Optional[str] = None,
    ) -> Tuple[dict, Optional[bytes]]:
        """
        Sign a request for Bedrock or Sagemaker

        Returns:
            Tuple[dict, Optional[str]]: A tuple containing the headers and the json str body of the request
        """
        if api_key is not None:
            aws_bearer_token: Optional[str] = api_key
        else:
            aws_bearer_token = get_secret_str("AWS_BEARER_TOKEN_BEDROCK")

        # If aws bearer token is set, use it directly in the header
        if aws_bearer_token:
            headers = headers or {}
            headers["Content-Type"] = "application/json"
            headers["Authorization"] = f"Bearer {aws_bearer_token}"
            return headers, json.dumps(request_data).encode()

        # If no bearer token is set, proceed with the existing SigV4 authentication
        try:
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
            from botocore.credentials import Credentials
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

        ## CREDENTIALS ##
        # pop aws_secret_access_key, aws_access_key_id, aws_session_token, aws_region_name from kwargs, since completion calls fail with them
        aws_secret_access_key = optional_params.get("aws_secret_access_key", None)
        aws_access_key_id = optional_params.get("aws_access_key_id", None)
        aws_session_token = optional_params.get("aws_session_token", None)
        aws_role_name = optional_params.get("aws_role_name", None)
        aws_session_name = optional_params.get("aws_session_name", None)
        aws_profile_name = optional_params.get("aws_profile_name", None)
        aws_web_identity_token = optional_params.get("aws_web_identity_token", None)
        aws_sts_endpoint = optional_params.get("aws_sts_endpoint", None)
        aws_external_id = optional_params.get("aws_external_id", None)
        aws_region_name = self._get_aws_region_name(
            optional_params=optional_params, model=model
        )

        credentials: Credentials = self.get_credentials(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            aws_region_name=aws_region_name,
            aws_session_name=aws_session_name,
            aws_profile_name=aws_profile_name,
            aws_role_name=aws_role_name,
            aws_web_identity_token=aws_web_identity_token,
            aws_sts_endpoint=aws_sts_endpoint,
            aws_external_id=aws_external_id,
        )

        sigv4 = SigV4Auth(credentials, service_name, aws_region_name)
        if headers is not None:
            headers = {"Content-Type": "application/json", **headers}
        else:
            headers = {"Content-Type": "application/json"}

        request = AWSRequest(
            method="POST",
            url=api_base,
            data=json.dumps(request_data),
            headers=headers,
        )
        sigv4.add_auth(request)

        request_headers_dict = dict(request.headers)
        if (
            headers is not None and "Authorization" in headers
        ):  # prevent sigv4 from overwriting the auth header
            request_headers_dict["Authorization"] = headers["Authorization"]

        return request_headers_dict, request.body
