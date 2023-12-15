#### What this does ####
#    On success + failure, log events to Supabase

import dotenv, os
import requests

dotenv.load_dotenv()  # Loading env variables using dotenv
import traceback
import datetime, subprocess, sys
import litellm, uuid
from litellm._logging import print_verbose

class DyanmoDBLogger:
    # Class variables or attributes

    def __init__(self, table_name="litellm-server-logs"):
        # Instance variables
        import boto3
        self.dynamodb = boto3.resource('dynamodb', region_name=os.environ["AWS_REGION_NAME"])
        self.table_name = table_name

        # on init check if there is a table with name == self.table_name
        # if not call self.create_dynamodb_table()
        if not self.check_table_exists():
            print_verbose(f"DynamoDB: Table {self.table_name} does not exist. Creating table")
            self.create_dynamodb_table()

    def check_table_exists(self):
        existing_tables = self.dynamodb.meta.client.list_tables()['TableNames']
        print_verbose(f"Dynamo DB: Existing Tables= {existing_tables}")
        return self.table_name in existing_tables


    def create_dynamodb_table(self):
        # for dynamo we can create a table with id attribute, there's no need to define other cols
        table_params = {
            'TableName': self.table_name,
            'KeySchema': [
                {'AttributeName': 'id', 'KeyType': 'HASH'}  # 'id' is the primary key
            ],
            'AttributeDefinitions': [
                {'AttributeName': 'id', 'AttributeType': 'S'}  # 'S' denotes string type
            ],
            'ProvisionedThroughput': {
                'ReadCapacityUnits': 5,   # Adjust based on your read/write capacity needs
                'WriteCapacityUnits': 5
            }
        }

        self.dynamodb.create_table(**table_params)
        print_verbose(f'Table {self.table_name} created successfully')
    
    async def _async_log_event(self, kwargs, response_obj, start_time, end_time, print_verbose):
        self.log_event(kwargs, response_obj, start_time, end_time, print_verbose)
    def log_event(self, kwargs, response_obj, start_time, end_time, print_verbose):
        try:
            print_verbose(
                f"DynamoDB Logging - Enters logging function for model {kwargs}"
            )

            # construct payload to send to DynamoDB
            # follows the same params as langfuse.py
            litellm_params = kwargs.get("litellm_params", {})
            metadata = litellm_params.get("metadata", {}) or {}  # if litellm_params['metadata'] == None
            messages = kwargs.get("messages")
            optional_params = kwargs.get("optional_params", {})
            call_type = kwargs.get("call_type", "litellm.completion")
            usage = response_obj["usage"]
            id = response_obj.get("id", str(uuid.uuid4()))

            # Build the initial payload
            payload = {
                "id": id,
                "call_type": call_type,
                "startTime": start_time,
                "endTime": end_time,
                "model": kwargs.get("model", ""),
                "user": kwargs.get("user", ""),
                "modelParameters": optional_params,
                "messages": messages,
                "response": response_obj,
                "usage": usage,
                "metadata": metadata
            }

            # Ensure everything in the payload is converted to str
            for key, value in payload.items():
                try:
                    payload[key] = str(value)
                except:
                    # non blocking if it can't cast to a str
                    pass


            print_verbose(f"\nDynamoDB Logger - Logging payload = {payload}")
            
            # put data in dyanmo DB
            table = self.dynamodb.Table(self.table_name)
            # Assuming log_data is a dictionary with log information
            response = table.put_item(Item=payload)

            print_verbose(f"Response from DynamoDB:{str(response)}")

            print_verbose(
                f"DynamoDB Layer Logging - final response object: {response_obj}"
            )
            return response
        except:
            traceback.print_exc()
            print_verbose(f"DynamoDB Layer Error - {traceback.format_exc()}")
            pass