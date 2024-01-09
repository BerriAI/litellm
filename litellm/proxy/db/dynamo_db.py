import json
from aiodynamo.client import Client
from aiodynamo.credentials import Credentials, StaticCredentials
from aiodynamo.http.httpx import HTTPX
from aiodynamo.models import Throughput, KeySchema, KeySpec, KeyType, PayPerRequest
from yarl import URL
from litellm.proxy.db.base_client import CustomDB
from litellm.proxy._types import DynamoDBArgs, DBTableNames, LiteLLM_VerificationToken, LiteLLM_Config, LiteLLM_UserTable
from litellm import get_secret
from typing import Any, List, Literal, Optional
from aiodynamo.expressions import UpdateExpression, F
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
            self.throughput_type = Throughput(read=database_arguments.read_capacity_units, write=database_arguments.write_capacity_units)
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
                await table.create(
                        self.throughput_type,
                        KeySchema(hash_key=KeySpec("user_id", KeyType.string)),
                    )
            ## Token 
            table = client.table(DBTableNames.key.value)
            if not await table.exists():
                await table.create(
                        self.throughput_type,
                        KeySchema(hash_key=KeySpec("token", KeyType.string)),
                    )
            ## Config  
            table = client.table(DBTableNames.config.value)
            if not await table.exists():
                await table.create(
                        self.throughput_type,
                        KeySchema(hash_key=KeySpec("param_name", KeyType.string)),
                    )

    async def insert_data(self, value: Any, table_name: Literal['user', 'key', 'config']):
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

    async def get_data(self, key: str, value: str, table_name: Literal['user', 'key', 'config']):
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


            if table_name == DBTableNames.user.name:
                new_response = LiteLLM_UserTable(**response)
            elif table_name == DBTableNames.key.name:
                new_response = {}
                for k, v in response.items(): # handle json string 
                    if (k == "aliases" or k == "config" or k == "metadata") and v is not None and isinstance(v, str):
                        new_response[k] = json.loads(v)
                    else: 
                        new_response[k] = v
                new_response = LiteLLM_VerificationToken(**new_response)
            elif table_name == DBTableNames.config.name:
                new_response = LiteLLM_Config(**response)
            return new_response
                    

    async def update_data(self, key: str, value: Any, table_name: Literal['user', 'key', 'config']):
        async with ClientSession() as session:
            client = Client(AIOHTTP(session), Credentials.auto(), self.region_name)
            table = None
            key_name = None
            data_obj = None
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

            # Initialize an empty UpdateExpression
            update_expression = UpdateExpression()

            # Add updates for each field that has been modified
            for field in data_obj.model_fields_set:
                # If a Pydantic model has a __fields_set__ attribute, it's a set of fields that were set when the model was instantiated
                field_value = getattr(data_obj, field)
                if isinstance(field_value, datetime):
                    field_value = field_value.isoformat()
                update_expression = update_expression.set(F(field), field_value)

            # Perform the update in DynamoDB
            result = await table.update_item(
                key={key_name: key},
                update_expression=update_expression,
                return_values=ReturnValues.NONE
            )
            return result
    
    async def delete_data(self, keys: List[str], table_name: Literal['user', 'key', 'config']):
        """
        Not Implemented yet. 
        """
        return super().delete_data(keys, table_name)