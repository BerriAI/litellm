import os
from typing import Any, Optional, Union

import httpx


def init_rds_client(
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_region_name: Optional[str] = None,
    aws_session_name: Optional[str] = None,
    aws_profile_name: Optional[str] = None,
    aws_role_name: Optional[str] = None,
    aws_web_identity_token: Optional[str] = None,
    timeout: Optional[Union[float, httpx.Timeout]] = None,
):
    from litellm.secret_managers.main import get_secret

    # check for custom AWS_REGION_NAME and use it if not passed to init_bedrock_client
    litellm_aws_region_name = get_secret("AWS_REGION_NAME", None)
    standard_aws_region_name = get_secret("AWS_REGION", None)
    ## CHECK IS  'os.environ/' passed in
    # Define the list of parameters to check
    params_to_check = [
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
            params_to_check[i] = get_secret(param)  # type: ignore
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
        try:
            oidc_token = open(aws_web_identity_token).read()  # check if filepath
        except Exception:
            oidc_token = get_secret(aws_web_identity_token)

        if oidc_token is None:
            raise Exception(
                "OIDC token could not be retrieved from secret manager.",
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
            service_name="rds",
            aws_access_key_id=sts_response["Credentials"]["AccessKeyId"],
            aws_secret_access_key=sts_response["Credentials"]["SecretAccessKey"],
            aws_session_token=sts_response["Credentials"]["SessionToken"],
            region_name=region_name,
            config=config,
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


def generate_iam_auth_token(
    db_host, db_port, db_user, client: Optional[Any] = None
) -> str:
    from urllib.parse import quote

    if client is None:
        boto_client = init_rds_client(
            aws_region_name=os.getenv("AWS_REGION_NAME"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_session_name=os.getenv("AWS_SESSION_NAME"),
            aws_profile_name=os.getenv("AWS_PROFILE_NAME"),
            aws_role_name=os.getenv("AWS_ROLE_NAME", os.getenv("AWS_ROLE_ARN")),
            aws_web_identity_token=os.getenv(
                "AWS_WEB_IDENTITY_TOKEN", os.getenv("AWS_WEB_IDENTITY_TOKEN_FILE")
            ),
        )
    else:
        boto_client = client

    token = boto_client.generate_db_auth_token(
        DBHostname=db_host, Port=db_port, DBUsername=db_user
    )
    cleaned_token = quote(token, safe="")

    return cleaned_token
