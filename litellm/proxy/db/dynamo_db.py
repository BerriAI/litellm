"""
Deprecated. Only PostgresSQL is supported.
"""

import json
from datetime import datetime
from typing import Any, List, Literal, Optional, Union

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    DynamoDBArgs,
    LiteLLM_Config,
    LiteLLM_UserTable,
    LiteLLM_VerificationToken,
)
from litellm.proxy.db.base_client import CustomDB
from litellm.proxy.utils import hash_token
from litellm.secret_managers.main import get_secret


class DynamoDBWrapper(CustomDB):
    from aiodynamo.credentials import Credentials, StaticCredentials

    credentials: Credentials

    def __init__(self, database_arguments: DynamoDBArgs):
        from aiodynamo.client import Client
        from aiodynamo.credentials import Credentials
        from aiodynamo.expressions import F, UpdateExpression, Value
        from aiodynamo.http.aiohttp import AIOHTTP
        from aiodynamo.http.httpx import HTTPX
        from aiodynamo.models import (
            KeySchema,
            KeySpec,
            KeyType,
            PayPerRequest,
            ReturnValues,
            Throughput,
        )
        from aiohttp import ClientSession
        from yarl import URL

        self.throughput_type = None
        if database_arguments.billing_mode == "PAY_PER_REQUEST":
            self.throughput_type = PayPerRequest()
        elif database_arguments.billing_mode == "PROVISIONED_THROUGHPUT":
            if (
                database_arguments.read_capacity_units is not None
                and isinstance(database_arguments.read_capacity_units, int)
                and database_arguments.write_capacity_units is not None
                and isinstance(database_arguments.write_capacity_units, int)
            ):
                self.throughput_type = Throughput(read=database_arguments.read_capacity_units, write=database_arguments.write_capacity_units)  # type: ignore
            else:
                raise Exception(
                    f"Invalid args passed in. Need to set both read_capacity_units and write_capacity_units. Args passed in - {database_arguments}"
                )
        self.database_arguments = database_arguments
        self.region_name = database_arguments.region_name

    def set_env_vars_based_on_arn(self):
        if self.database_arguments.aws_role_name is None:
            return
        verbose_proxy_logger.debug(
            f"DynamoDB: setting env vars based on arn={self.database_arguments.aws_role_name}"
        )
        import os

        import boto3

        sts_client = boto3.client("sts")

        # call 1
        sts_client.assume_role_with_web_identity(
            RoleArn=self.database_arguments.aws_role_name,
            RoleSessionName=self.database_arguments.aws_session_name,
            WebIdentityToken=self.database_arguments.aws_web_identity_token,
        )

        # call 2
        assumed_role = sts_client.assume_role(
            RoleArn=self.database_arguments.assume_role_aws_role_name,
            RoleSessionName=self.database_arguments.assume_role_aws_session_name,
        )

        aws_access_key_id = assumed_role["Credentials"]["AccessKeyId"]
        aws_secret_access_key = assumed_role["Credentials"]["SecretAccessKey"]
        aws_session_token = assumed_role["Credentials"]["SessionToken"]

        verbose_proxy_logger.debug(
            f"Got STS assumed Role, aws_access_key_id={aws_access_key_id}"
        )
        # set these in the env so aiodynamo can use them
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
        os.environ["AWS_SESSION_TOKEN"] = aws_session_token
