"""
Send logs to Argilla for annotation
"""

import asyncio
import json
import os
import random
import types
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel  # type: ignore

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.integrations.argilla import (
    SUPPORTED_PAYLOAD_FIELDS,
    ArgillaCredentialsObject,
    ArgillaItem,
)
from litellm.types.utils import StandardLoggingPayload


def is_serializable(value):
    non_serializable_types = (
        types.CoroutineType,
        types.FunctionType,
        types.GeneratorType,
        BaseModel,
    )
    return not isinstance(value, non_serializable_types)


class ArgillaLogger(CustomBatchLogger):
    def __init__(
        self,
        argilla_api_key: Optional[str] = None,
        argilla_dataset_name: Optional[str] = None,
        argilla_base_url: Optional[str] = None,
        **kwargs,
    ):
        if litellm.argilla_transformation_object is None:
            raise Exception(
                "'litellm.argilla_transformation_object' is required, to log your payload to Argilla."
            )
        self.validate_argilla_transformation_object(
            litellm.argilla_transformation_object
        )
        self.argilla_transformation_object = litellm.argilla_transformation_object
        self.default_credentials = self.get_credentials_from_env(
            argilla_api_key=argilla_api_key,
            argilla_dataset_name=argilla_dataset_name,
            argilla_base_url=argilla_base_url,
        )
        self.sampling_rate: float = (
            float(os.getenv("ARGILLA_SAMPLING_RATE"))  # type: ignore
            if os.getenv("ARGILLA_SAMPLING_RATE") is not None
            and os.getenv("ARGILLA_SAMPLING_RATE").strip().isdigit()  # type: ignore
            else 1.0
        )

        self.async_httpx_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        _batch_size = (
            os.getenv("ARGILLA_BATCH_SIZE", None) or litellm.argilla_batch_size
        )
        if _batch_size:
            self.batch_size = int(_batch_size)
        asyncio.create_task(self.periodic_flush())
        self.flush_lock = asyncio.Lock()
        super().__init__(**kwargs, flush_lock=self.flush_lock)

    def validate_argilla_transformation_object(
        self, argilla_transformation_object: Dict[str, Any]
    ):
        if not isinstance(argilla_transformation_object, dict):
            raise Exception(
                "'argilla_transformation_object' must be a dictionary, to log your payload to Argilla."
            )

        for v in argilla_transformation_object.values():
            if v not in SUPPORTED_PAYLOAD_FIELDS:
                raise Exception(
                    f"All values in argilla_transformation_object must be a key in SUPPORTED_PAYLOAD_FIELDS, {v} is not a valid key."
                )

    def get_credentials_from_env(
        self,
        argilla_api_key: Optional[str],
        argilla_dataset_name: Optional[str],
        argilla_base_url: Optional[str],
    ) -> ArgillaCredentialsObject:

        _credentials_api_key = argilla_api_key or os.getenv("ARGILLA_API_KEY")
        if _credentials_api_key is None:
            raise Exception("Invalid Argilla API Key given. _credentials_api_key=None.")

        _credentials_base_url = (
            argilla_base_url
            or os.getenv("ARGILLA_BASE_URL")
            or "http://localhost:6900/"
        )
        if _credentials_base_url is None:
            raise Exception(
                "Invalid Argilla Base URL given. _credentials_base_url=None."
            )

        _credentials_dataset_name = (
            argilla_dataset_name
            or os.getenv("ARGILLA_DATASET_NAME")
            or "litellm-completion"
        )
        if _credentials_dataset_name is None:
            raise Exception("Invalid Argilla Dataset give. Value=None.")
        else:
            dataset_response = litellm.module_level_client.get(
                url=f"{_credentials_base_url}/api/v1/me/datasets?name={_credentials_dataset_name}",
                headers={"X-Argilla-Api-Key": _credentials_api_key},
            )
            json_response = dataset_response.json()
            if (
                "items" in json_response
                and isinstance(json_response["items"], list)
                and len(json_response["items"]) > 0
            ):
                _credentials_dataset_name = json_response["items"][0]["id"]

        return ArgillaCredentialsObject(
            ARGILLA_API_KEY=_credentials_api_key,
            ARGILLA_BASE_URL=_credentials_base_url,
            ARGILLA_DATASET_NAME=_credentials_dataset_name,
        )

    def get_chat_messages(
        self, payload: StandardLoggingPayload
    ) -> List[Dict[str, Any]]:
        payload_messages = payload.get("messages", None)

        if payload_messages is None:
            raise Exception("No chat messages found in payload.")

        if (
            isinstance(payload_messages, list)
            and len(payload_messages) > 0
            and isinstance(payload_messages[0], dict)
        ):
            return payload_messages
        elif isinstance(payload_messages, dict):
            return [payload_messages]
        else:
            raise Exception(f"Invalid chat messages format: {payload_messages}")

    def get_str_response(self, payload: StandardLoggingPayload) -> str:
        response = payload["response"]

        if response is None:
            raise Exception("No response found in payload.")

        if isinstance(response, str):
            return response
        elif isinstance(response, dict):
            return (
                response.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
        else:
            raise Exception(f"Invalid response format: {response}")

    def _prepare_log_data(
        self, kwargs, response_obj, start_time, end_time
    ) -> Optional[ArgillaItem]:
        try:
            # Ensure everything in the payload is converted to str
            payload: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object", None
            )

            if payload is None:
                raise Exception("Error logging request payload. Payload=none.")

            argilla_message = self.get_chat_messages(payload)
            argilla_response = self.get_str_response(payload)
            argilla_item: ArgillaItem = {"fields": {}}
            for k, v in self.argilla_transformation_object.items():
                if v == "messages":
                    argilla_item["fields"][k] = argilla_message
                elif v == "response":
                    argilla_item["fields"][k] = argilla_response
                else:
                    argilla_item["fields"][k] = payload.get(v, None)

            return argilla_item
        except Exception:
            raise

    def _send_batch(self):
        if not self.log_queue:
            return

        argilla_api_base = self.default_credentials["ARGILLA_BASE_URL"]
        argilla_dataset_name = self.default_credentials["ARGILLA_DATASET_NAME"]

        url = f"{argilla_api_base}/api/v1/datasets/{argilla_dataset_name}/records/bulk"

        argilla_api_key = self.default_credentials["ARGILLA_API_KEY"]

        headers = {"X-Argilla-Api-Key": argilla_api_key}

        try:
            response = litellm.module_level_client.post(
                url=url,
                json=self.log_queue,
                headers=headers,
            )

            if response.status_code >= 300:
                verbose_logger.error(
                    f"Argilla Error: {response.status_code} - {response.text}"
                )
            else:
                verbose_logger.debug(
                    f"Batch of {len(self.log_queue)} runs successfully created"
                )

            self.log_queue.clear()
        except Exception:
            verbose_logger.exception("Argilla Layer Error - Error sending batch.")

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            sampling_rate = (
                float(os.getenv("LANGSMITH_SAMPLING_RATE"))  # type: ignore
                if os.getenv("LANGSMITH_SAMPLING_RATE") is not None
                and os.getenv("LANGSMITH_SAMPLING_RATE").strip().isdigit()  # type: ignore
                else 1.0
            )
            random_sample = random.random()
            if random_sample > sampling_rate:
                verbose_logger.info(
                    "Skipping Langsmith logging. Sampling rate={}, random_sample={}".format(
                        sampling_rate, random_sample
                    )
                )
                return  # Skip logging
            verbose_logger.debug(
                "Langsmith Sync Layer Logging - kwargs: %s, response_obj: %s",
                kwargs,
                response_obj,
            )
            data = self._prepare_log_data(kwargs, response_obj, start_time, end_time)
            if data is None:
                return

            self.log_queue.append(data)
            verbose_logger.debug(
                f"Langsmith, event added to queue. Will flush in {self.flush_interval} seconds..."
            )

            if len(self.log_queue) >= self.batch_size:
                self._send_batch()

        except Exception:
            verbose_logger.exception("Langsmith Layer Error - log_success_event error")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            sampling_rate = self.sampling_rate
            random_sample = random.random()
            if random_sample > sampling_rate:
                verbose_logger.info(
                    "Skipping Langsmith logging. Sampling rate={}, random_sample={}".format(
                        sampling_rate, random_sample
                    )
                )
                return  # Skip logging
            verbose_logger.debug(
                "Langsmith Async Layer Logging - kwargs: %s, response_obj: %s",
                kwargs,
                response_obj,
            )
            payload: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object", None
            )

            data = self._prepare_log_data(kwargs, response_obj, start_time, end_time)

            ## ALLOW CUSTOM LOGGERS TO MODIFY / FILTER DATA BEFORE LOGGING
            for callback in litellm.callbacks:
                if isinstance(callback, CustomLogger):
                    try:
                        if data is None:
                            break
                        data = await callback.async_dataset_hook(data, payload)
                    except NotImplementedError:
                        pass

            if data is None:
                return

            self.log_queue.append(data)
            verbose_logger.debug(
                "Langsmith logging: queue length %s, batch size %s",
                len(self.log_queue),
                self.batch_size,
            )
            if len(self.log_queue) >= self.batch_size:
                await self.flush_queue()
        except Exception:
            verbose_logger.exception(
                "Argilla Layer Error - error logging async success event."
            )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        sampling_rate = self.sampling_rate
        random_sample = random.random()
        if random_sample > sampling_rate:
            verbose_logger.info(
                "Skipping Langsmith logging. Sampling rate={}, random_sample={}".format(
                    sampling_rate, random_sample
                )
            )
            return  # Skip logging
        verbose_logger.info("Langsmith Failure Event Logging!")
        try:
            data = self._prepare_log_data(kwargs, response_obj, start_time, end_time)
            self.log_queue.append(data)
            verbose_logger.debug(
                "Langsmith logging: queue length %s, batch size %s",
                len(self.log_queue),
                self.batch_size,
            )
            if len(self.log_queue) >= self.batch_size:
                await self.flush_queue()
        except Exception:
            verbose_logger.exception(
                "Langsmith Layer Error - error logging async failure event."
            )

    async def async_send_batch(self):
        """
        sends runs to /batch endpoint

        Sends runs from self.log_queue

        Returns: None

        Raises: Does not raise an exception, will only verbose_logger.exception()
        """
        if not self.log_queue:
            return

        argilla_api_base = self.default_credentials["ARGILLA_BASE_URL"]
        argilla_dataset_name = self.default_credentials["ARGILLA_DATASET_NAME"]

        url = f"{argilla_api_base}/api/v1/datasets/{argilla_dataset_name}/records/bulk"

        argilla_api_key = self.default_credentials["ARGILLA_API_KEY"]

        headers = {"X-Argilla-Api-Key": argilla_api_key}

        try:
            response = await self.async_httpx_client.put(
                url=url,
                data=json.dumps(
                    {
                        "items": self.log_queue,
                    }
                ),
                headers=headers,
                timeout=60000,
            )
            response.raise_for_status()

            if response.status_code >= 300:
                verbose_logger.error(
                    f"Argilla Error: {response.status_code} - {response.text}"
                )
            else:
                verbose_logger.debug(
                    "Batch of %s runs successfully created", len(self.log_queue)
                )
        except httpx.HTTPStatusError:
            verbose_logger.exception("Argilla HTTP Error")
        except Exception:
            verbose_logger.exception("Argilla Layer Error")
