import json
from aiodynamo.client import Client
from aiodynamo.credentials import Credentials, StaticCredentials
from aiodynamo.http.httpx import HTTPX
from aiodynamo.models import Throughput, KeySchema, KeySpec, KeyType, PayPerRequest
from yarl import URL
from litellm.proxy.db.base_client import CustomDB
from litellm.proxy._types import (
    DynamoDBArgs,
    DBTableNames,
    LiteLLM_VerificationToken,
    LiteLLM_Config,
    LiteLLM_UserTable,
)
from litellm import get_secret
from typing import Any, List, Literal, Optional, Union
from aiodynamo.expressions import UpdateExpression, F, Value
from aiodynamo.models import ReturnValues
from aiodynamo.http.aiohttp import AIOHTTP
from aiohttp import ClientSession
from datetime import datetime


class DynamoDBWrapper(CustomDB):
    credentials: Credentials

    def __init__(self, database_arguments: DynamoDBArgs):
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
        self.region_name = database_arguments.region_name

    async def connect(self):
        """
        Connect to DB, and creating / updating any tables
        """
        async with ClientSession() as session:
            client = Client(AIOHTTP(session), Credentials.auto(), self.region_name)
            ## User
            table = client.table(DBTableNames.user.value)
            if not await table.exists():
                try:
                    await table.create(
                        self.throughput_type,
                        KeySchema(hash_key=KeySpec("user_id", KeyType.string)),
                    )
                except:
                    raise Exception(
                        f"Failed to create table - {DBTableNames.user.value}.\nPlease create a new table called {DBTableNames.user.value}\nAND set `hash_key` as 'user_id'"
                    )
            ## Token
            table = client.table(DBTableNames.key.value)
            if not await table.exists():
                try:
                    await table.create(
                        self.throughput_type,
                        KeySchema(hash_key=KeySpec("token", KeyType.string)),
                    )
                except:
                    raise Exception(
                        f"Failed to create table - {DBTableNames.key.value}.\nPlease create a new table called {DBTableNames.key.value}\nAND set `hash_key` as 'token'"
                    )
            ## Config
            table = client.table(DBTableNames.config.value)
            if not await table.exists():
                try:
                    await table.create(
                        self.throughput_type,
                        KeySchema(hash_key=KeySpec("param_name", KeyType.string)),
                    )
                except:
                    raise Exception(
                        f"Failed to create table - {DBTableNames.config.value}.\nPlease create a new table called {DBTableNames.config.value}\nAND set `hash_key` as 'token'"
                    )

    async def insert_data(
        self, value: Any, table_name: Literal["user", "key", "config"]
    ):
        async with ClientSession() as session:
            client = Client(AIOHTTP(session), Credentials.auto(), self.region_name)
            table = None
            if table_name == DBTableNames.user.name:
                table = client.table(DBTableNames.user.value)
            elif table_name == DBTableNames.key.name:
                table = client.table(DBTableNames.key.value)
            elif table_name == DBTableNames.config.name:
                table = client.table(DBTableNames.config.value)

            for k, v in value.items():
                if isinstance(v, datetime):
                    value[k] = v.isoformat()

            await table.put_item(item=value)

    async def get_data(
        self, key: str, value: str, table_name: Literal["user", "key", "config"]
    ):
        async with ClientSession() as session:
            client = Client(AIOHTTP(session), Credentials.auto(), self.region_name)
            table = None
            if table_name == DBTableNames.user.name:
                table = client.table(DBTableNames.user.value)
            elif table_name == DBTableNames.key.name:
                table = client.table(DBTableNames.key.value)
            elif table_name == DBTableNames.config.name:
                table = client.table(DBTableNames.config.value)

            response = await table.get_item({key: value})

            new_response: Any = None
            if table_name == DBTableNames.user.name:
                new_response = LiteLLM_UserTable(**response)
            elif table_name == DBTableNames.key.name:
                new_response = {}
                for k, v in response.items():  # handle json string
                    if (
                        (k == "aliases" or k == "config" or k == "metadata")
                        and v is not None
                        and isinstance(v, str)
                    ):
                        new_response[k] = json.loads(v)
                    else:
                        new_response[k] = v
                new_response = LiteLLM_VerificationToken(**new_response)
            elif table_name == DBTableNames.config.name:
                new_response = LiteLLM_Config(**response)
            return new_response

    async def update_data(
        self, key: str, value: Any, table_name: Literal["user", "key", "config"]
    ):
        async with ClientSession() as session:
            client = Client(AIOHTTP(session), Credentials.auto(), self.region_name)
            table = None
            key_name = None
            data_obj: Optional[
                Union[LiteLLM_Config, LiteLLM_UserTable, LiteLLM_VerificationToken]
            ] = None
            if table_name == DBTableNames.user.name:
                table = client.table(DBTableNames.user.value)
                key_name = "user_id"
                data_obj = LiteLLM_UserTable(user_id=key, **value)

            elif table_name == DBTableNames.key.name:
                table = client.table(DBTableNames.key.value)
                key_name = "token"
                data_obj = LiteLLM_VerificationToken(token=key, **value)

            elif table_name == DBTableNames.config.name:
                table = client.table(DBTableNames.config.value)
                key_name = "param_name"
                data_obj = LiteLLM_Config(param_name=key, **value)

            if data_obj is None:
                raise Exception(
                    f"invalid table name passed in - {table_name}. Unable to load valid data object - {data_obj}."
                )
            # Initialize an empty UpdateExpression

            actions: List = []
            for field in data_obj.fields_set():
                field_value = getattr(data_obj, field)

                # Convert datetime object to ISO8601 string
                if isinstance(field_value, datetime):
                    field_value = field_value.isoformat()

                # Accumulate updates
                actions.append((F(field), Value(value=field_value)))

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
