#### What this does ####
#    On success + failure, log events to Supabase

import os
import traceback
from decimal import Decimal
from typing import Any, Optional, TypedDict, cast

import litellm
from litellm.types.utils import StandardLoggingPayload


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

    @classmethod
    def convert_for_dynamodb(cls, value):
        if value is None:
            return None  # DynamoDB will reject this, but we'll filter None values later
        elif isinstance(value, float):
            return Decimal(str(value))
        elif isinstance(value, (list, tuple)):
            return [cls.convert_for_dynamodb(item) for item in value if item is not None]
        elif isinstance(value, dict):
            return {
                k: cls.convert_for_dynamodb(v) 
                for k, v in value.items() 
                if v is not None
            }
        elif isinstance(value, str) and not value:
            return None  # Empty strings aren't allowed in DynamoDB
        return value

    @classmethod
    def prepare_for_dynamodb(cls, data) -> dict[str, Any]:
        # First convert all floats to Decimal and handle nested structures
        converted = {
            k: cls.convert_for_dynamodb(v)
            for k, v in data.items()
            if v is not None  # Filter out None values
        }
        
        # Remove any remaining empty lists, dicts, or None values
        return {
            k: v for k, v in converted.items()
            if v is not None and not (isinstance(v, (list, dict)) and len(v) == 0)
        }

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

            # Clean Metadata before logging - never log raw metadata
            # the raw metadata can contain circular references which leads to infinite recursion
            # we clean out all extra litellm metadata params before logging
            clean_metadata = {}
            if isinstance(metadata, dict):
                for key, value in metadata.items():
                    # clean litellm metadata before logging
                    if key in [
                        "headers",
                        "endpoint",
                        "caching_groups",
                        "previous_models",
                    ]:
                        continue
                    else:
                        clean_metadata[key] = value


            # Ensure everything in the payload is converted to str
            payload: Optional[StandardLoggingPayload] = cast(
                Optional[StandardLoggingPayload],
                kwargs.get("standard_logging_object", None),
            )

            if payload is None:
                return

            print_verbose(f"\nDynamoDB Logger - Logging payload = {payload}")

            dynamodb_payload = self.prepare_for_dynamodb(payload)

            # put data in dyanmo DB
            table = self.dynamodb.Table(self.table_name)
            response = table.put_item(Item=dynamodb_payload)

            print_verbose(f"Response from DynamoDB:{str(response)}")

            print_verbose(
                f"DynamoDB Layer Logging - final response object: {response_obj}"
            )
            return response

        except Exception:
            print_verbose(f"DynamoDB Layer Error - {traceback.format_exc()}")
            pass
