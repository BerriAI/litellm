#### What this does ####
#    On success + failure, log events to Supabase

import dotenv, os
import requests

dotenv.load_dotenv()  # Loading env variables using dotenv
import traceback
import datetime, subprocess, sys
import litellm, uuid
from litellm._logging import print_verbose, verbose_logger


class S3Logger:
    # Class variables or attributes
    def __init__(
        self,
        s3_bucket_name=None,
        s3_path=None,
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

        try:
            verbose_logger.debug(
                f"in init s3 logger - s3_callback_params {litellm.s3_callback_params}"
            )

            if litellm.s3_callback_params is not None:
                # read in .env variables - example os.environ/AWS_BUCKET_NAME
                for key, value in litellm.s3_callback_params.items():
                    if type(value) is str and value.startswith("os.environ/"):
                        litellm.s3_callback_params[key] = litellm.get_secret(value)
                # now set s3 params from litellm.s3_logger_params
                s3_bucket_name = litellm.s3_callback_params.get("s3_bucket_name")
                s3_region_name = litellm.s3_callback_params.get("s3_region_name")
                s3_api_version = litellm.s3_callback_params.get("s3_api_version")
                s3_use_ssl = litellm.s3_callback_params.get("s3_use_ssl", True)
                s3_verify = litellm.s3_callback_params.get("s3_verify")
                s3_endpoint_url = litellm.s3_callback_params.get("s3_endpoint_url")
                s3_aws_access_key_id = litellm.s3_callback_params.get(
                    "s3_aws_access_key_id"
                )
                s3_aws_secret_access_key = litellm.s3_callback_params.get(
                    "s3_aws_secret_access_key"
                )
                s3_aws_session_token = litellm.s3_callback_params.get(
                    "s3_aws_session_token"
                )
                s3_config = litellm.s3_callback_params.get("s3_config")
                # done reading litellm.s3_callback_params

            self.bucket_name = s3_bucket_name
            self.s3_path = s3_path
            verbose_logger.debug(f"s3 logger using endpoint url {s3_endpoint_url}")
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
        except Exception as e:
            print_verbose(f"Got exception on init s3 client {str(e)}")
            raise e

    async def _async_log_event(
        self, kwargs, response_obj, start_time, end_time, print_verbose
    ):
        self.log_event(kwargs, response_obj, start_time, end_time, print_verbose)

    def log_event(self, kwargs, response_obj, start_time, end_time, print_verbose):
        try:
            verbose_logger.debug(
                f"s3 Logging - Enters logging function for model {kwargs}"
            )

            # construct payload to send to s3
            # follows the same params as langfuse.py
            litellm_params = kwargs.get("litellm_params", {})
            metadata = (
                litellm_params.get("metadata", {}) or {}
            )  # if litellm_params['metadata'] == None
            messages = kwargs.get("messages")
            optional_params = kwargs.get("optional_params", {})
            call_type = kwargs.get("call_type", "litellm.completion")
            cache_hit = kwargs.get("cache_hit", False)
            usage = response_obj["usage"]
            id = response_obj.get("id", str(uuid.uuid4()))

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

            # Build the initial payload
            payload = {
                "id": id,
                "call_type": call_type,
                "cache_hit": cache_hit,
                "startTime": start_time,
                "endTime": end_time,
                "model": kwargs.get("model", ""),
                "user": kwargs.get("user", ""),
                "modelParameters": optional_params,
                "messages": messages,
                "response": response_obj,
                "usage": usage,
                "metadata": clean_metadata,
            }

            # Ensure everything in the payload is converted to str
            for key, value in payload.items():
                try:
                    payload[key] = str(value)
                except:
                    # non blocking if it can't cast to a str
                    pass

            s3_file_name = litellm.utils.get_logging_id(start_time, payload) or ""
            s3_object_key = (
                (self.s3_path.rstrip("/") + "/" if self.s3_path else "")
                + start_time.strftime("%Y-%m-%d")
                + "/"
                + s3_file_name
            )  # we need the s3 key to include the time, so we log cache hits too
            s3_object_key += ".json"

            s3_object_download_filename = (
                "time-"
                + start_time.strftime("%Y-%m-%dT%H-%M-%S-%f")
                + "_"
                + payload["id"]
                + ".json"
            )

            import json

            payload = json.dumps(payload)

            print_verbose(f"\ns3 Logger - Logging payload = {payload}")

            response = self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_object_key,
                Body=payload,
                ContentType="application/json",
                ContentLanguage="en",
                ContentDisposition=f'inline; filename="{s3_object_download_filename}"',
                CacheControl="private, immutable, max-age=31536000, s-maxage=0",
            )

            print_verbose(f"Response from s3:{str(response)}")

            print_verbose(f"s3 Layer Logging - final response object: {response_obj}")
            return response
        except Exception as e:
            traceback.print_exc()
            verbose_logger.debug(f"s3 Layer Error - {str(e)}\n{traceback.format_exc()}")
            pass
