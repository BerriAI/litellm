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

    def log_event(self, kwargs, response_obj, start_time, end_time, print_verbose=print):
        try:
            print_verbose(
                f"DynamoDB Logging - Enters logging function for model {kwargs}"
            )

            # construct payload to send to dyanmo DB
            # follows the same params as langfuse.py
            litellm_params = kwargs.get("litellm_params", {})
            metadata = litellm_params.get("metadata", {}) or {} # if litellm_params['metadata'] == None 
            messages = kwargs.get("messages")
            optional_params = kwargs.get("optional_params", {})
            function_name = kwargs.get("function_name", "litellm.completion")
            usage = str(response_obj["usage"])
            id = response_obj.get("id", str(uuid.uuid4()))

 

            # convert all optional params to str
            for param, value in optional_params.items():
                try:
                    optional_params[param] = str(value)
                except:
                    # if casting value to str fails don't block logging
                    pass
            response_obj = str(response_obj)
 
            payload = {
                "id": id,
                "function_name": function_name,                  # str
                "startTime": str(start_time),                # str
                "endTime": str(end_time),                    # str
                "model": kwargs.get("model", ""),              # str
                "user": kwargs.get("user", ""),                 # str
                "modelParameters": optional_params,     # dict[str]
                "messages": [{"role": "user", "content": "hit"}],                   # List[dict[str, str]]
                "response": response_obj,               # litellm.ModelResponse
                "usage" : usage,        # dict[str, int]
                "metadata": metadata                    # dict[Any, Any]
            }
            print_verbose(f"\nDynamoDB Logger - Logging payload = {payload}")
            
            # put data in dyanmo DB
            table = self.dynamodb.Table(self.table_name)
            # Assuming log_data is a dictionary with log information
            response = table.put_item(Item=payload)
            print(f'Log data added to {self.table_name} successfully:', response)

            print_verbose(
                f"DynamoDB Layer Logging - final response object: {response_obj}"
            )
            return response
        except:
            traceback.print_exc()
            print_verbose(f"DynamoDB Layer Error - {traceback.format_exc()}")
            pass