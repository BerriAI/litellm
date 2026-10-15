import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import httpx

from litellm._logging import verbose_proxy_logger

_RDS_HOSTNAME_REGION_PATTERN: Final = re.compile(
    r"(?:[^.]+\.)+(?P<region>[a-z]{2}(?:-[a-z]+)+-\d+)\.rds\.amazonaws\.com(?:\.cn)?\.?",
    re.IGNORECASE,
)


def init_rds_client(
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_region_name: str | None = None,
    aws_session_name: str | None = None,
    aws_profile_name: str | None = None,
    aws_role_name: str | None = None,
    aws_web_identity_token: str | None = None,
    timeout: float | httpx.Timeout | None = None,
    aws_session_token: str | None = None,
):
    from litellm.secret_managers.main import get_secret

    # check for custom AWS_REGION_NAME and use it if not passed to init_bedrock_client
    litellm_aws_region_name: Final = get_secret("AWS_REGION_NAME", None)
    standard_aws_region_name: Final = get_secret("AWS_REGION", None)
    ## CHECK IS  'os.environ/' passed in
    # Define the list of parameters to check
    params_to_check: Final = [
        aws_access_key_id,
        aws_secret_access_key,
        aws_region_name,
        aws_session_name,
        aws_profile_name,
        aws_role_name,
        aws_web_identity_token,
    ]

    # Iterate over parameters and update if needed
    for i, param in enumerate(params_to_check):
        if param and param.startswith("os.environ/"):
            params_to_check[i] = get_secret(param)
    # Assign updated values back to parameters
    (
        aws_access_key_id,
        aws_secret_access_key,
        aws_region_name,
        aws_session_name,
        aws_profile_name,
        aws_role_name,
        aws_web_identity_token,
    ) = params_to_check

    ### SET REGION NAME
    region_name = aws_region_name
    if aws_region_name:
        region_name = aws_region_name
    elif litellm_aws_region_name:
        region_name = litellm_aws_region_name
    elif standard_aws_region_name:
        region_name = standard_aws_region_name
    else:
        raise Exception(
            "AWS region not set: set AWS_REGION_NAME or AWS_REGION env variable or in .env file",
        )

    import boto3

    config: Final = boto3.session.Config(**_boto_timeouts(timeout))

    ### CHECK STS ###
    if aws_web_identity_token is not None and aws_role_name is not None and aws_session_name is not None:
        try:
            oidc_token = open(aws_web_identity_token).read()  # check if filepath
        except Exception:
            oidc_token = get_secret(aws_web_identity_token)

        if oidc_token is None:
            raise Exception(
                "OIDC token could not be retrieved from secret manager.",
            )

        web_identity_credentials: Final = _assume_role_with_web_identity(
            oidc_token=oidc_token,
            aws_role_name=aws_role_name,
            aws_session_name=aws_session_name,
            irsa_role_arn=os.getenv("AWS_ROLE_ARN"),
        )

        client = boto3.client(
            service_name="rds",
            aws_access_key_id=web_identity_credentials.access_key_id,
            aws_secret_access_key=web_identity_credentials.secret_access_key,
            aws_session_token=web_identity_credentials.session_token,
            region_name=region_name,
            config=config,
        )

    elif aws_role_name is not None and aws_session_name is not None:
        # use sts if role name passed in
        sts_client = boto3.client(
            "sts",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )

        sts_response = sts_client.assume_role(RoleArn=aws_role_name, RoleSessionName=aws_session_name)

        client = boto3.client(
            service_name="rds",
            aws_access_key_id=sts_response["Credentials"]["AccessKeyId"],
            aws_secret_access_key=sts_response["Credentials"]["SecretAccessKey"],
            aws_session_token=sts_response["Credentials"]["SessionToken"],
            region_name=region_name,
            config=config,
        )
    elif aws_access_key_id is not None:
        # uses auth params passed to completion
        # aws_access_key_id is not None, assume user is trying to auth using litellm.completion

        client = boto3.client(
            service_name="rds",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
            region_name=region_name,
            config=config,
        )
    elif aws_profile_name is not None:
        # uses auth values from AWS profile usually stored in ~/.aws/credentials

        client = boto3.Session(profile_name=aws_profile_name).client(
            service_name="rds",
            region_name=region_name,
            config=config,
        )

    else:
        # aws_access_key_id is None, assume user is trying to auth using env variables
        # boto3 automatically reads env variables

        client = boto3.client(
            service_name="rds",
            region_name=region_name,
            config=config,
        )

    return client


def _boto_timeouts(timeout: float | httpx.Timeout | None) -> Mapping[str, float | None]:
    if timeout is None:
        return MappingProxyType({})
    if isinstance(timeout, httpx.Timeout):
        return MappingProxyType({"connect_timeout": timeout.connect, "read_timeout": timeout.read})
    return MappingProxyType({"connect_timeout": timeout, "read_timeout": timeout})


@dataclass(frozen=True, slots=True)
class _StsCredentials:
    access_key_id: str
    secret_access_key: str
    session_token: str


def _assume_role_with_web_identity(
    *, oidc_token: str, aws_role_name: str, aws_session_name: str, irsa_role_arn: str | None
) -> _StsCredentials:
    """Trade the web identity token for credentials on ``aws_role_name``.

    First tries the direct exchange (the role trusts the OIDC provider). When that is refused and the
    pod runs under IRSA as a different role, chains instead: web identity into the pod role, then
    ``sts:AssumeRole`` into ``aws_role_name`` (the role trusts the pod role, typically cross-account).
    """
    import boto3
    from botocore.exceptions import ClientError

    sts_client: Final = boto3.client("sts")
    try:
        direct: Final = sts_client.assume_role_with_web_identity(
            RoleArn=aws_role_name,
            RoleSessionName=aws_session_name,
            WebIdentityToken=oidc_token,
            DurationSeconds=3600,
        )["Credentials"]
        return _StsCredentials(direct["AccessKeyId"], direct["SecretAccessKey"], direct["SessionToken"])
    except ClientError as direct_error:
        if not irsa_role_arn or irsa_role_arn == aws_role_name:
            raise
        verbose_proxy_logger.info(
            "RDS IAM auth: AssumeRoleWithWebIdentity on %s was refused (%s), chaining through IRSA role %s",
            aws_role_name,
            direct_error.response.get("Error", {}).get("Code", "unknown"),
            irsa_role_arn,
        )

    pod_role: Final = sts_client.assume_role_with_web_identity(
        RoleArn=irsa_role_arn,
        RoleSessionName=aws_session_name,
        WebIdentityToken=oidc_token,
    )["Credentials"]
    chained: Final = boto3.client(
        "sts",
        aws_access_key_id=pod_role["AccessKeyId"],
        aws_secret_access_key=pod_role["SecretAccessKey"],
        aws_session_token=pod_role["SessionToken"],
    ).assume_role(RoleArn=aws_role_name, RoleSessionName=aws_session_name)["Credentials"]
    return _StsCredentials(chained["AccessKeyId"], chained["SecretAccessKey"], chained["SessionToken"])


def rds_region_from_hostname(db_host: str) -> str | None:
    match: Final = _RDS_HOSTNAME_REGION_PATTERN.fullmatch(db_host)
    if match is None:
        return None
    return match.group("region").lower()


def generate_iam_auth_token(
    db_host: str,
    db_port: str,
    db_user: str,
    client: Any | None = None,
    *,
    region: str | None = None,
) -> str:
    from urllib.parse import quote

    if client is None:
        boto_client = init_rds_client(
            aws_region_name=os.getenv("AWS_REGION_NAME"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
            aws_session_name=os.getenv("AWS_SESSION_NAME"),
            aws_profile_name=os.getenv("AWS_PROFILE_NAME"),
            aws_role_name=os.getenv("AWS_ROLE_NAME", os.getenv("AWS_ROLE_ARN")),
            aws_web_identity_token=os.getenv("AWS_WEB_IDENTITY_TOKEN", os.getenv("AWS_WEB_IDENTITY_TOKEN_FILE")),
        )
    else:
        boto_client = client

    token: Final = boto_client.generate_db_auth_token(
        DBHostname=db_host,
        Port=db_port,
        DBUsername=db_user,
        Region=region or rds_region_from_hostname(db_host),
    )
    cleaned_token: Final = quote(token, safe="")

    return cleaned_token
