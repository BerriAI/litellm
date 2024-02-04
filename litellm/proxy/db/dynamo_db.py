from litellm.proxy.db.base_client import CustomDB
from litellm.proxy._types import (
    DynamoDBArgs,
    LiteLLM_VerificationToken,
    LiteLLM_Config,
    LiteLLM_UserTable,
)
from litellm.proxy.utils import hash_token
from litellm import get_secret
from typing import Any, List, Literal, Optional, Union
import json
from datetime import datetime
from litellm._logging import verbose_proxy_logger


class DynamoDBWrapper(CustomDB):
    from aiodynamo.credentials import Credentials, StaticCredentials

    credentials: Credentials

    def __init__(self, database_arguments: DynamoDBArgs):
        from aiodynamo.client import Client
        from aiodynamo.credentials import Credentials, StaticCredentials
        from aiodynamo.http.httpx import HTTPX
        from aiodynamo.models import (
            Throughput,
            KeySchema,
            KeySpec,
            KeyType,
            PayPerRequest,
        )
        from yarl import URL
        from aiodynamo.expressions import UpdateExpression, F, Value
        from aiodynamo.models import ReturnValues
        from aiodynamo.http.aiohttp import AIOHTTP
        from aiohttp import ClientSession

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

    async def connect(self):
        """
        Connect to DB, and creating / updating any tables
        """
        from aiodynamo.client import Client
        from aiodynamo.credentials import Credentials, StaticCredentials
        from aiodynamo.http.httpx import HTTPX
        from aiodynamo.models import (
            Throughput,
            KeySchema,
            KeySpec,
            KeyType,
            PayPerRequest,
        )
        from yarl import URL
        from aiodynamo.expressions import UpdateExpression, F, Value
        from aiodynamo.models import ReturnValues
        from aiodynamo.http.aiohttp import AIOHTTP
        from aiohttp import ClientSession
        import aiohttp

        verbose_proxy_logger.debug("DynamoDB Wrapper - Attempting to connect")
        # before making ClientSession check if ssl_verify=False
        if self.database_arguments.ssl_verify == False:
            client_session = ClientSession(connector=aiohttp.TCPConnector(ssl=False))
        else:
            client_session = ClientSession()
        async with client_session as session:
            client = Client(AIOHTTP(session), Credentials.auto(), self.region_name)
            ## User
            try:
                error_occurred = False
                verbose_proxy_logger.debug("DynamoDB Wrapper - Creating User Table")
                table = client.table(self.database_arguments.user_table_name)
                verbose_proxy_logger.debug(f"DynamoDB Wrapper - Created Table, {table}")
                if not await table.exists():
                    verbose_proxy_logger.debug(
                        f"DynamoDB Wrapper - {table} does not exist"
                    )
                    await table.create(
                        self.throughput_type,
                        KeySchema(hash_key=KeySpec("user_id", KeyType.string)),
                    )
            except Exception as e:
                error_occurred = True
            if error_occurred == True:
                raise Exception(
                    f"Failed to create table - {self.database_arguments.user_table_name}.\nPlease create a new table called {self.database_arguments.user_table_name}\nAND set `hash_key` as 'user_id'"
                )
            ## Token
            try:
                error_occurred = False
                table = client.table(self.database_arguments.key_table_name)
                if not await table.exists():
                    await table.create(
                        self.throughput_type,
                        KeySchema(hash_key=KeySpec("token", KeyType.string)),
                    )
            except Exception as e:
                error_occurred = True
            if error_occurred == True:
                raise Exception(
                    f"Failed to create table - {self.database_arguments.key_table_name}.\nPlease create a new table called {self.database_arguments.key_table_name}\nAND set `hash_key` as 'token'"
                )
            ## Config
            try:
                error_occurred = False
                table = client.table(self.database_arguments.config_table_name)
                if not await table.exists():
                    await table.create(
                        self.throughput_type,
                        KeySchema(hash_key=KeySpec("param_name", KeyType.string)),
                    )
            except Exception as e:
                error_occurred = True
            if error_occurred == True:
                raise Exception(
                    f"Failed to create table - {self.database_arguments.config_table_name}.\nPlease create a new table called {self.database_arguments.config_table_name}\nAND set `hash_key` as 'param_name'"
                )

            ## Spend
            try:
                verbose_proxy_logger.debug("DynamoDB Wrapper - Creating Spend Table")
                error_occurred = False
                table = client.table(self.database_arguments.spend_table_name)
                if not await table.exists():
                    await table.create(
                        self.throughput_type,
                        KeySchema(hash_key=KeySpec("request_id", KeyType.string)),
                    )
            except Exception as e:
                error_occurred = True
            if error_occurred == True:
                raise Exception(
                    f"Failed to create table - {self.database_arguments.key_table_name}.\nPlease create a new table called {self.database_arguments.key_table_name}\nAND set `hash_key` as 'token'"
                )
            verbose_proxy_logger.debug("DynamoDB Wrapper - Done connecting()")

    async def insert_data(
        self, value: Any, table_name: Literal["user", "key", "config", "spend"]
    ):
        from aiodynamo.client import Client
        from aiodynamo.credentials import Credentials, StaticCredentials
        from aiodynamo.http.httpx import HTTPX
        from aiodynamo.models import (
            Throughput,
            KeySchema,
            KeySpec,
            KeyType,
            PayPerRequest,
        )
        from yarl import URL
        from aiodynamo.expressions import UpdateExpression, F, Value
        from aiodynamo.models import ReturnValues
        from aiodynamo.http.aiohttp import AIOHTTP
        from aiohttp import ClientSession
        import aiohttp

        if self.database_arguments.ssl_verify == False:
            client_session = ClientSession(connector=aiohttp.TCPConnector(ssl=False))
        else:
            client_session = ClientSession()
        async with client_session as session:
            client = Client(AIOHTTP(session), Credentials.auto(), self.region_name)
            table = None
            if table_name == "user":
                table = client.table(self.database_arguments.user_table_name)
            elif table_name == "key":
                table = client.table(self.database_arguments.key_table_name)
            elif table_name == "config":
                table = client.table(self.database_arguments.config_table_name)
            elif table_name == "spend":
                table = client.table(self.database_arguments.spend_table_name)

            value = value.copy()
            for k, v in value.items():
                if k == "token" and value[k].startswith("sk-"):
                    value[k] = hash_token(token=v)
                if isinstance(v, datetime):
                    value[k] = v.isoformat()

            return await table.put_item(item=value, return_values=ReturnValues.all_old)

    async def get_data(self, key: str, table_name: Literal["user", "key", "config"]):
        from aiodynamo.client import Client
        from aiodynamo.credentials import Credentials, StaticCredentials
        from aiodynamo.http.httpx import HTTPX
        from aiodynamo.models import (
            Throughput,
            KeySchema,
            KeySpec,
            KeyType,
            PayPerRequest,
        )
        from yarl import URL
        from aiodynamo.expressions import UpdateExpression, F, Value
        from aiodynamo.models import ReturnValues
        from aiodynamo.http.aiohttp import AIOHTTP
        from aiohttp import ClientSession
        import aiohttp

        if self.database_arguments.ssl_verify == False:
            client_session = ClientSession(connector=aiohttp.TCPConnector(ssl=False))
        else:
            client_session = ClientSession()

        async with client_session as session:
            client = Client(AIOHTTP(session), Credentials.auto(), self.region_name)
            table = None
            key_name = None
            if table_name == "user":
                table = client.table(self.database_arguments.user_table_name)
                key_name = "user_id"
            elif table_name == "key":
                table = client.table(self.database_arguments.key_table_name)
                key_name = "token"
            elif table_name == "config":
                table = client.table(self.database_arguments.config_table_name)
                key_name = "param_name"

            response = await table.get_item({key_name: key})

            new_response: Any = None
            if table_name == "user":
                new_response = LiteLLM_UserTable(**response)
            elif table_name == "key":
                new_response = {}
                for k, v in response.items():  # handle json string
                    if (
                        (k == "aliases" or k == "config" or k == "metadata")
                        and v is not None
                        and isinstance(v, str)
                    ):
                        new_response[k] = json.loads(v)
                    elif (k == "tpm_limit" or k == "rpm_limit") and isinstance(
                        v, float
                    ):
                        new_response[k] = int(v)
                    else:
                        new_response[k] = v
                new_response = LiteLLM_VerificationToken(**new_response)
            elif table_name == "config":
                new_response = LiteLLM_Config(**response)
            return new_response

    async def update_data(
        self, key: str, value: dict, table_name: Literal["user", "key", "config"]
    ):
        from aiodynamo.client import Client
        from aiodynamo.credentials import Credentials, StaticCredentials
        from aiodynamo.http.httpx import HTTPX
        from aiodynamo.models import (
            Throughput,
            KeySchema,
            KeySpec,
            KeyType,
            PayPerRequest,
        )
        from yarl import URL
        from aiodynamo.expressions import UpdateExpression, F, Value
        from aiodynamo.models import ReturnValues
        from aiodynamo.http.aiohttp import AIOHTTP
        from aiohttp import ClientSession
        import aiohttp

        if self.database_arguments.ssl_verify == False:
            client_session = ClientSession(connector=aiohttp.TCPConnector(ssl=False))
        else:
            client_session = ClientSession()
        async with client_session as session:
            client = Client(AIOHTTP(session), Credentials.auto(), self.region_name)
            table = None
            key_name = None
            try:
                if table_name == "user":
                    table = client.table(self.database_arguments.user_table_name)
                    key_name = "user_id"

                elif table_name == "key":
                    table = client.table(self.database_arguments.key_table_name)
                    key_name = "token"

                elif table_name == "config":
                    table = client.table(self.database_arguments.config_table_name)
                    key_name = "param_name"
                else:
                    raise Exception(
                        f"Invalid table name. Needs to be one of - {self.database_arguments.user_table_name}, {self.database_arguments.key_table_name}, {self.database_arguments.config_table_name}"
                    )
            except Exception as e:
                raise Exception(f"Error connecting to table - {str(e)}")

            # Initialize an empty UpdateExpression

            actions: List = []
            value = value.copy()
            for k, v in value.items():
                # Convert datetime object to ISO8601 string
                if isinstance(v, datetime):
                    v = v.isoformat()
                if k == "token" and value[k].startswith("sk-"):
                    value[k] = hash_token(token=v)

                # Accumulate updates
                actions.append((F(k), Value(value=v)))

            update_expression = UpdateExpression(set_updates=actions)
            # Perform the update in DynamoDB
            result = await table.update_item(
                key={key_name: key},
                update_expression=update_expression,
                return_values=ReturnValues.none,
            )
            return result

    async def delete_data(
        self, keys: List[str], table_name: Literal["user", "key", "config"]
    ):
        """
        Not Implemented yet.
        """
        return super().delete_data(keys, table_name)
