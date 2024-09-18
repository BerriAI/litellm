import datetime
import os
import boto3
import json
import openai
from typing import Optional
from botocore.exceptions import ClientError

import litellm
from litellm._logging import print_verbose, verbose_logger
from litellm.types.utils import StandardLoggingPayload


class CloudWatchLogger:
    def __init__(
        self,
        log_group_name=None,
        log_stream_name=None,
        aws_region=None,
    ):
        try:
            verbose_logger.debug(
                f"in init cloudwatch logger - cloudwatch_callback_params {litellm.cloudwatch_callback_params}"
            )

            if litellm.cloudwatch_callback_params is not None:
                # read in .env variables - example os.environ/CLOUDWATCH_LOG_GROUP_NAME
                for key, value in litellm.cloudwatch_callback_params.items():
                    if isinstance(value, str) and value.startswith("os.environ/"):
                        litellm.cloudwatch_callback_params[key] = litellm.get_secret(value)
                # now set cloudwatch params from litellm.cloudwatch_callback_params
                log_group_name = litellm.cloudwatch_callback_params.get("log_group_name", log_group_name)
                log_stream_name = litellm.cloudwatch_callback_params.get("log_stream_name", log_stream_name)
                aws_region = litellm.cloudwatch_callback_params.get("aws_region", aws_region)

            self.log_group_name = log_group_name or os.getenv("CLOUDWATCH_LOG_GROUP_NAME")
            self.log_stream_name = log_stream_name or os.getenv("CLOUDWATCH_LOG_STREAM_NAME")
            self.aws_region = aws_region or os.getenv("AWS_REGION")

            if self.log_group_name is None:
                raise ValueError("log_group_name must be provided either through parameters, cloudwatch_callback_params, or environment variables.")

            # Initialize CloudWatch Logs client
            self.logs_client = boto3.client("logs", region_name=self.aws_region)

            # Ensure the log group exists
            self._ensure_log_group()
            self.sequence_token = None

        except Exception as e:
            print_verbose(f"Got exception while initializing CloudWatch Logs client: {str(e)}")
            raise e

    def _ensure_log_group(self):
        try:
            self.logs_client.create_log_group(logGroupName=self.log_group_name)
            print_verbose(f"Created log group: {self.log_group_name}")
        except self.logs_client.exceptions.ResourceAlreadyExistsException:
            print_verbose(f"Log group already exists: {self.log_group_name}")

    def _ensure_log_stream(self):
        try:
            self.logs_client.create_log_stream(
                logGroupName=self.log_group_name, logStreamName=self.log_stream_name
            )
            print_verbose(f"Created log stream: {self.log_stream_name}")
        except self.logs_client.exceptions.ResourceAlreadyExistsException:
            print_verbose(f"Log stream already exists: {self.log_stream_name}")

    async def _async_log_event(
        self, kwargs, response_obj, start_time, end_time, print_verbose
    ):
        self.log_event(kwargs, response_obj, start_time, end_time, print_verbose)

    def log_event(self, kwargs, response_obj, start_time, end_time, print_verbose):
        try:
            # Ensure log group and stream exist before logging
            self._ensure_log_group()

            verbose_logger.debug(
                f"CloudWatch Logging - Enters logging function for model {kwargs}"
            )


            # Construct payload
            payload: Optional[StandardLoggingPayload] = kwargs.get("standard_logging_object", None)
            if payload is None:
                litellm_params = kwargs.get("litellm_params", {})
                metadata = litellm_params.get("metadata", {}) or {}
                payload = {key: value for key, value in metadata.items() if key not in ["headers", "endpoint", "caching_groups", "previous_models"]}

            if isinstance(response_obj, openai.lib.streaming._assistants.AsyncAssistantEventHandler):
                current_run = response_obj.current_run
                payload["id"] = current_run.id
                payload["assistant_id"] = current_run.assistant_id
                self.log_stream_name = payload["thread_id"] = current_run.thread_id
                payload["completion_tokens"] = current_run.usage.completion_tokens
                payload["prompt_tokens"] = current_run.usage.prompt_tokens
                payload["total_tokens"] = current_run.usage.total_tokens
                payload["created_at"] = current_run.created_at
                payload["completed_at"] = current_run.completed_at
                payload["failed_at"] = current_run.failed_at
                payload["cancelled_at"] = current_run.cancelled_at
                payload["assistant_message"] = str(response_obj.current_message_snapshot.content)
                payload.pop("response", None) # remove response from payload as it's not json serializable

            log_event_message = json.dumps(payload)

            timestamp = int(datetime.datetime.now().timestamp() * 1000)
            
            if self.log_stream_name is None:
                self.log_stream_name = payload["id"]
            self._ensure_log_stream()

            # Prepare the log event parameters
            log_event_params = {
                'logGroupName': self.log_group_name,
                'logStreamName': self.log_stream_name,
                'logEvents': [
                    {
                        'timestamp': timestamp,
                        'message': log_event_message
                    }
                ]
            }

            if self.sequence_token:
                log_event_params['sequenceToken'] = self.sequence_token

            response = self.logs_client.put_log_events(**log_event_params)

            self.sequence_token = response['nextSequenceToken']

            print_verbose(f"Logged to CloudWatch: {log_event_message}")
        except Exception as e:
            verbose_logger.exception(f"CloudWatch Logs Error: {str(e)}")
