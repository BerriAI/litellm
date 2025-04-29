#### What this does ####
#    On success + failure, log events to Supabase

import os
import traceback
import uuid
from typing import Any

import litellm


class DyanmoDBLogger:
    # Class variables or attributes

    def __init__(self):
        # Instance variables
        import boto3

        self.dynamodb: Any = boto3.resource(
            "dynamodb", region_name=os.environ["AWS_REGION_NAME"]
        )
        if litellm.dynamodb_table_name is None:
            raise ValueError(
                "LiteLLM Error, trying to use DynamoDB but not table name passed. Create a table and set `litellm.dynamodb_table_name=<your-table>`"
            )
        self.table_name = litellm.dynamodb_table_name

    async def _async_log_event(
        self, kwargs, response_obj, start_time, end_time, print_verbose
    ):
        self.log_event(kwargs, response_obj, start_time, end_time, print_verbose)

    def log_event(self, kwargs, response_obj, start_time, end_time, print_verbose):
        try:
            print_verbose(
                f"DynamoDB Logging - Enters logging function for model {kwargs}"
            )

            # construct payload to send to DynamoDB
            # follows the same params as langfuse.py
            litellm_params = kwargs.get("litellm_params", {})
            metadata = (
                litellm_params.get("metadata", {}) or {}
            )  # if litellm_params['metadata'] == None
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
                "metadata": metadata,
            }

            # Ensure everything in the payload is converted to str
            for key, value in payload.items():
                try:
                    payload[key] = str(value)
                except Exception:
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
        except Exception:
            print_verbose(f"DynamoDB Layer Error - {traceback.format_exc()}")
            pass
