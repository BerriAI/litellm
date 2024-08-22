import json
from typing import List, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.caching import DualCache, InMemoryCache
from litellm.utils import get_secret

from .base import BaseLLM


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


class BaseAWSLLM(BaseLLM):
    def __init__(self) -> None:
        self.iam_cache = DualCache()
        super().__init__()

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
        import boto3

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
                    "aws_region_name": aws_region_name,
                    "aws_sts_endpoint": sts_endpoint,
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
            from botocore.credentials import Credentials

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
        else:
            session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=aws_region_name,
            )

            return session.get_credentials()
