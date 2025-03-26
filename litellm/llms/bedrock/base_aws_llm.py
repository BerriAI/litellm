import hashlib
import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, cast, get_args

import httpx
from pydantic import BaseModel

from litellm._logging import verbose_logger
from litellm.caching.caching import DualCache
from litellm.constants import BEDROCK_INVOKE_PROVIDERS_LITERAL
from litellm.litellm_core_utils.dd_tracing import tracer
from litellm.secret_managers.main import get_secret

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
                    params_to_check[i] = os.getenv(key)

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
            "aws_sts_endpoint=%s",
            aws_access_key_id,
            aws_secret_access_key,
            aws_session_token,
            aws_region_name,
            aws_session_name,
            aws_profile_name,
            aws_role_name,
            aws_web_identity_token,
            aws_sts_endpoint,
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
            )
        elif aws_role_name is not None and aws_session_name is not None:
            credentials, _cache_ttl = self._auth_with_aws_role(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_role_name=aws_role_name,
                aws_session_name=aws_session_name,
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
        sts_response = sts_client.assume_role_with_web_identity(
            RoleArn=aws_role_name,
            RoleSessionName=aws_session_name,
            WebIdentityToken=oidc_token,
            DurationSeconds=3600,
            Policy='{"Version":"2012-10-17","Statement":[{"Sid":"BedrockLiteLLM","Effect":"Allow","Action":["bedrock:InvokeModel","bedrock:InvokeModelWithResponseStream"],"Resource":"*","Condition":{"Bool":{"aws:SecureTransport":"true"},"StringLike":{"aws:UserAgent":"litellm/*"}}}]}',
        )

        iam_creds_dict = {
            "aws_access_key_id": sts_response["Credentials"]["AccessKeyId"],
            "aws_secret_access_key": sts_response["Credentials"]["SecretAccessKey"],
            "aws_session_token": sts_response["Credentials"]["SessionToken"],
            "region_name": aws_region_name,
        }

        if sts_response["PackedPolicySize"] > 75:
            verbose_logger.warning(
                f"The policy size is greater than 75% of the allowed size, PackedPolicySize: {sts_response['PackedPolicySize']}"
            )

        with tracer.trace("boto3.Session(**iam_creds_dict)"):
            session = boto3.Session(**iam_creds_dict)

        iam_creds = session.get_credentials()
        return iam_creds, self._get_default_ttl_for_boto3_credentials()

    @tracer.wrap()
    def _auth_with_aws_role(
        self,
        aws_access_key_id: Optional[str],
        aws_secret_access_key: Optional[str],
        aws_role_name: str,
        aws_session_name: str,
    ) -> Tuple[Credentials, Optional[int]]:
        """
        Authenticate with AWS Role
        """
        import boto3
        from botocore.credentials import Credentials

        with tracer.trace("boto3.client(sts)"):
            sts_client = boto3.client(
                "sts",
                aws_access_key_id=aws_access_key_id,  # [OPTIONAL]
                aws_secret_access_key=aws_secret_access_key,  # [OPTIONAL]
            )

        sts_response = sts_client.assume_role(
            RoleArn=aws_role_name, RoleSessionName=aws_session_name
        )

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
            endpoint_url = f"https://bedrock-runtime.{aws_region_name}.amazonaws.com"

        # Determine proxy_endpoint_url
        if env_aws_bedrock_runtime_endpoint and isinstance(
            env_aws_bedrock_runtime_endpoint, str
        ):
            proxy_endpoint_url = env_aws_bedrock_runtime_endpoint
        elif aws_bedrock_runtime_endpoint is not None and isinstance(
            aws_bedrock_runtime_endpoint, str
        ):
            proxy_endpoint_url = aws_bedrock_runtime_endpoint
        else:
            proxy_endpoint_url = endpoint_url

        return endpoint_url, proxy_endpoint_url

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
        data: str,
        headers: dict,
    ) -> AWSPreparedRequest:
        try:
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

        sigv4 = SigV4Auth(credentials, "bedrock", aws_region_name)

        request = AWSRequest(
            method="POST", url=endpoint_url, data=data, headers=headers
        )
        sigv4.add_auth(request)
        if (
            extra_headers is not None and "Authorization" in extra_headers
        ):  # prevent sigv4 from overwriting the auth header
            request.headers["Authorization"] = extra_headers["Authorization"]
        prepped = request.prepare()

        return prepped
