#### What this does ####
#    On success + failure, log events to Supabase

import dotenv, os
import requests

dotenv.load_dotenv()  # Loading env variables using dotenv
import traceback
import datetime, subprocess, sys
import litellm, uuid
from litellm._logging import print_verbose


class S3Logger:
    # Class variables or attributes
    def __init__(
        self,
        s3_bucket_name="litellm-logs",
        s3_region_name=None,
        s3_api_version=None,
        s3_use_ssl=True,
        s3_verify=None,
        s3_endpoint_url=None,
        s3_aws_access_key_id=None,
        s3_aws_secret_access_key=None,
        s3_aws_session_token=None,
        s3_config=None,
        **kwargs,
    ):
        import boto3

        self.bucket_name = s3_bucket_name
        # Create an S3 client with custom endpoint URL
        self.s3_client = boto3.client(
            "s3",
            region_name=s3_region_name,
            endpoint_url=s3_endpoint_url,
            api_version=s3_api_version,
            use_ssl=s3_use_ssl,
            verify=s3_verify,
            aws_access_key_id=s3_aws_access_key_id,
            aws_secret_access_key=s3_aws_secret_access_key,
            aws_session_token=s3_aws_session_token,
            config=s3_config,
            **kwargs,
        )

    async def _async_log_event(
        self, kwargs, response_obj, start_time, end_time, print_verbose
    ):
        self.log_event(kwargs, response_obj, start_time, end_time, print_verbose)

    def log_event(self, kwargs, response_obj, start_time, end_time, print_verbose):
        try:
            print_verbose(f"s3 Logging - Enters logging function for model {kwargs}")

            # construct payload to send to s3
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
                except:
                    # non blocking if it can't cast to a str
                    pass
            s3_object_key = payload["id"]

            import json

            payload = json.dumps(payload)

            print_verbose(f"\ns3 Logger - Logging payload = {payload}")

            response = self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_object_key,
                Body=payload,
                ContentType="application/json",
                ContentLanguage="en",
                ContentDisposition=f'inline; filename="{key}.json"',
            )

            print_verbose(f"Response from s3:{str(response)}")

            print_verbose(f"s3 Layer Logging - final response object: {response_obj}")
            return response
        except:
            traceback.print_exc()
            print_verbose(f"s3 Layer Error - {traceback.format_exc()}")
            pass
