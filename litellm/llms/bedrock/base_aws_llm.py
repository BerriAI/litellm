import hashlib
import json
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel

from litellm._logging import verbose_logger
from litellm.caching.caching import DualCache
from litellm.secret_managers.main import get_secret, get_secret_str

if TYPE_CHECKING:
    from botocore.credentials import Credentials
else:
    Credentials = Any


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

    def get_cache_key(self, credential_args: Dict[str, Optional[str]]) -> str:
        """
        Generate a unique cache key based on the credential arguments.
        """
        # Convert credential arguments to a JSON string and hash it to create a unique key
        credential_str = json.dumps(credential_args, sort_keys=True)
        return hashlib.sha256(credential_str.encode()).hexdigest()

    def get_credentials(  # noqa: PLR0915
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

        import boto3
        from botocore.credentials import Credentials

        ## CHECK IS  'os.environ/' passed in
        param_names = [
            "aws_access_key_id",
            "aws_secret_access_key",
            "aws_session_token",
            "aws_region_name",
            "aws_session_name",
            "aws_profile_name",
            "aws_role_name",
            "aws_web_identity_token",
            "aws_sts_endpoint",
        ]
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
                key = param_names[i]
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

        # create cache key for non-expiring auth flows
        args = {k: v for k, v in locals().items() if k.startswith("aws_")}
        cache_key = self.get_cache_key(args)

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

        ### CHECK STS ###
        if (
            aws_web_identity_token is not None
            and aws_role_name is not None
            and aws_session_name is not None
        ):
            verbose_logger.debug(
                f"IN Web Identity Token: {aws_web_identity_token} | Role Name: {aws_role_name} | Session Name: {aws_session_name}"
            )

            if aws_sts_endpoint is None:
                sts_endpoint = f"https://sts.{aws_region_name}.amazonaws.com"
            else:
                sts_endpoint = aws_sts_endpoint

            iam_creds_cache_key = json.dumps(
                {
                    "aws_web_identity_token": aws_web_identity_token,
                    "aws_role_name": aws_role_name,
                    "aws_session_name": aws_session_name,
                }
            )

            iam_creds_dict = self.iam_cache.get_cache(iam_creds_cache_key)
            if iam_creds_dict is None:
                oidc_token = get_secret(aws_web_identity_token)

                if oidc_token is None:
                    raise AwsAuthError(
                        message="OIDC token could not be retrieved from secret manager.",
                        status_code=401,
                    )

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
                    "aws_secret_access_key": sts_response["Credentials"][
                        "SecretAccessKey"
                    ],
                    "aws_session_token": sts_response["Credentials"]["SessionToken"],
                    "region_name": aws_region_name,
                }

                self.iam_cache.set_cache(
                    key=iam_creds_cache_key,
                    value=json.dumps(iam_creds_dict),
                    ttl=3600 - 60,
                )

                if sts_response["PackedPolicySize"] > 75:
                    verbose_logger.warning(
                        f"The policy size is greater than 75% of the allowed size, PackedPolicySize: {sts_response['PackedPolicySize']}"
                    )

            session = boto3.Session(**iam_creds_dict)

            iam_creds = session.get_credentials()

            return iam_creds
        elif aws_role_name is not None and aws_session_name is not None:
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
            return credentials
        elif aws_profile_name is not None:  ### CHECK SESSION ###
            # uses auth values from AWS profile usually stored in ~/.aws/credentials
            client = boto3.Session(profile_name=aws_profile_name)

            return client.get_credentials()
        elif (
            aws_access_key_id is not None
            and aws_secret_access_key is not None
            and aws_session_token is not None
        ):  ### CHECK FOR AWS SESSION TOKEN ###
            from botocore.credentials import Credentials

            credentials = Credentials(
                access_key=aws_access_key_id,
                secret_key=aws_secret_access_key,
                token=aws_session_token,
            )

            return credentials
        elif (
            aws_access_key_id is not None
            and aws_secret_access_key is not None
            and aws_region_name is not None
        ):
            # Check if credentials are already in cache. These credentials have no expiry time.
            cached_credentials: Optional[Credentials] = self.iam_cache.get_cache(
                cache_key
            )
            if cached_credentials:
                return cached_credentials

            session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=aws_region_name,
            )

            credentials = session.get_credentials()

            if (
                credentials.token is None
            ):  # don't cache if session token exists. The expiry time for that is not known.
                self.iam_cache.set_cache(cache_key, credentials, ttl=3600 - 60)

            return credentials
        else:
            # check env var. Do not cache the response from this.
            session = boto3.Session()

            credentials = session.get_credentials()

            return credentials

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
        self, optional_params: dict
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
        aws_region_name = optional_params.pop("aws_region_name", None)
        aws_role_name = optional_params.pop("aws_role_name", None)
        aws_session_name = optional_params.pop("aws_session_name", None)
        aws_profile_name = optional_params.pop("aws_profile_name", None)
        aws_web_identity_token = optional_params.pop("aws_web_identity_token", None)
        aws_sts_endpoint = optional_params.pop("aws_sts_endpoint", None)
        aws_bedrock_runtime_endpoint = optional_params.pop(
            "aws_bedrock_runtime_endpoint", None
        )  # https://bedrock-runtime.{region_name}.amazonaws.com

        ### SET REGION NAME ###
        if aws_region_name is None:
            # check env #
            litellm_aws_region_name = get_secret_str("AWS_REGION_NAME", None)

            if litellm_aws_region_name is not None and isinstance(
                litellm_aws_region_name, str
            ):
                aws_region_name = litellm_aws_region_name

            standard_aws_region_name = get_secret_str("AWS_REGION", None)
            if standard_aws_region_name is not None and isinstance(
                standard_aws_region_name, str
            ):
                aws_region_name = standard_aws_region_name

            if aws_region_name is None:
                aws_region_name = "us-west-2"

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
