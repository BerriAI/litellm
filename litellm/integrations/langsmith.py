#### What this does ####
#    On success, logs events to Langsmith
import asyncio
import os
import random
import traceback
import types
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel  # type: ignore

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.integrations.langsmith import *
from litellm.types.utils import StandardCallbackDynamicParams, StandardLoggingPayload


def is_serializable(value):
    non_serializable_types = (
        types.CoroutineType,
        types.FunctionType,
        types.GeneratorType,
        BaseModel,
    )
    return not isinstance(value, non_serializable_types)


class LangsmithLogger(CustomBatchLogger):
    def __init__(
        self,
        langsmith_api_key: Optional[str] = None,
        langsmith_project: Optional[str] = None,
        langsmith_base_url: Optional[str] = None,
        **kwargs,
    ):
        self.default_credentials = self.get_credentials_from_env(
            langsmith_api_key=langsmith_api_key,
            langsmith_project=langsmith_project,
            langsmith_base_url=langsmith_base_url,
        )
        self.sampling_rate: float = (
            float(os.getenv("LANGSMITH_SAMPLING_RATE"))  # type: ignore
            if os.getenv("LANGSMITH_SAMPLING_RATE") is not None
            and os.getenv("LANGSMITH_SAMPLING_RATE").strip().isdigit()  # type: ignore
            else 1.0
        )
        self.langsmith_default_run_name = os.getenv(
            "LANGSMITH_DEFAULT_RUN_NAME", "LLMRun"
        )
        self.async_httpx_client = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        _batch_size = (
            os.getenv("LANGSMITH_BATCH_SIZE", None) or litellm.langsmith_batch_size
        )
        if _batch_size:
            self.batch_size = int(_batch_size)
        self.log_queue: List[LangsmithQueueObject] = []
        asyncio.create_task(self.periodic_flush())
        self.flush_lock = asyncio.Lock()

        super().__init__(**kwargs, flush_lock=self.flush_lock)

    def get_credentials_from_env(
        self,
        langsmith_api_key: Optional[str] = None,
        langsmith_project: Optional[str] = None,
        langsmith_base_url: Optional[str] = None,
    ) -> LangsmithCredentialsObject:

        _credentials_api_key = langsmith_api_key or os.getenv("LANGSMITH_API_KEY")
        if _credentials_api_key is None:
            raise Exception(
                "Invalid Langsmith API Key given. _credentials_api_key=None."
            )
        _credentials_project = (
            langsmith_project or os.getenv("LANGSMITH_PROJECT") or "litellm-completion"
        )
        if _credentials_project is None:
            raise Exception(
                "Invalid Langsmith API Key given. _credentials_project=None."
            )
        _credentials_base_url = (
            langsmith_base_url
            or os.getenv("LANGSMITH_BASE_URL")
            or "https://api.smith.langchain.com"
        )
        if _credentials_base_url is None:
            raise Exception(
                "Invalid Langsmith API Key given. _credentials_base_url=None."
            )

        return LangsmithCredentialsObject(
            LANGSMITH_API_KEY=_credentials_api_key,
            LANGSMITH_BASE_URL=_credentials_base_url,
            LANGSMITH_PROJECT=_credentials_project,
        )

    def _prepare_log_data(
        self,
        kwargs,
        response_obj,
        start_time,
        end_time,
        credentials: LangsmithCredentialsObject,
    ):
        try:
            _litellm_params = kwargs.get("litellm_params", {}) or {}
            metadata = _litellm_params.get("metadata", {}) or {}
            project_name = metadata.get(
                "project_name", credentials["LANGSMITH_PROJECT"]
            )
            run_name = metadata.get("run_name", self.langsmith_default_run_name)
            run_id = metadata.get("id", metadata.get("run_id", None))
            parent_run_id = metadata.get("parent_run_id", None)
            trace_id = metadata.get("trace_id", None)
            session_id = metadata.get("session_id", None)
            dotted_order = metadata.get("dotted_order", None)
            verbose_logger.debug(
                f"Langsmith Logging - project_name: {project_name}, run_name {run_name}"
            )

            # Ensure everything in the payload is converted to str
            payload: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object", None
            )

            if payload is None:
                raise Exception("Error logging request payload. Payload=none.")

            metadata = payload[
                "metadata"
            ]  # ensure logged metadata is json serializable

            data = {
                "name": run_name,
                "run_type": "llm",  # this should always be llm, since litellm always logs llm calls. Langsmith allow us to log "chain"
                "inputs": payload,
                "outputs": payload["response"],
                "session_name": project_name,
                "start_time": payload["startTime"],
                "end_time": payload["endTime"],
                "tags": payload["request_tags"],
                "extra": metadata,
            }

            if payload["error_str"] is not None and payload["status"] == "failure":
                data["error"] = payload["error_str"]

            if run_id:
                data["id"] = run_id

            if parent_run_id:
                data["parent_run_id"] = parent_run_id

            if trace_id:
                data["trace_id"] = trace_id

            if session_id:
                data["session_id"] = session_id

            if dotted_order:
                data["dotted_order"] = dotted_order

            run_id: Optional[str] = data.get("id")  # type: ignore
            if "id" not in data or data["id"] is None:
                """
                for /batch langsmith requires id, trace_id and dotted_order passed as params
                """
                run_id = str(uuid.uuid4())

                data["id"] = run_id

            if (
                "trace_id" not in data
                or data["trace_id"] is None
                and (run_id is not None and isinstance(run_id, str))
            ):
                data["trace_id"] = run_id

            if (
                "dotted_order" not in data
                or data["dotted_order"] is None
                and (run_id is not None and isinstance(run_id, str))
            ):
                data["dotted_order"] = self.make_dot_order(run_id=run_id)  # type: ignore

            verbose_logger.debug("Langsmith Logging data on langsmith: %s", data)

            return data
        except Exception:
            raise

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
            credentials = self._get_credentials_to_use_for_request(kwargs=kwargs)
            data = self._prepare_log_data(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
                credentials=credentials,
            )
            self.log_queue.append(
                LangsmithQueueObject(
                    data=data,
                    credentials=credentials,
                )
            )
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
            credentials = self._get_credentials_to_use_for_request(kwargs=kwargs)
            data = self._prepare_log_data(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
                credentials=credentials,
            )
            self.log_queue.append(
                LangsmithQueueObject(
                    data=data,
                    credentials=credentials,
                )
            )
            verbose_logger.debug(
                "Langsmith logging: queue length %s, batch size %s",
                len(self.log_queue),
                self.batch_size,
            )
            if len(self.log_queue) >= self.batch_size:
                await self.flush_queue()
        except Exception:
            verbose_logger.exception(
                "Langsmith Layer Error - error logging async success event."
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
            credentials = self._get_credentials_to_use_for_request(kwargs=kwargs)
            data = self._prepare_log_data(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
                credentials=credentials,
            )
            self.log_queue.append(
                LangsmithQueueObject(
                    data=data,
                    credentials=credentials,
                )
            )
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
        Handles sending batches of runs to Langsmith

        self.log_queue contains LangsmithQueueObjects
            Each LangsmithQueueObject has the following:
                - "credentials" - credentials to use for the request (langsmith_api_key, langsmith_project, langsmith_base_url)
                - "data" - data to log on to langsmith for the request


        This function
         - groups the queue objects by credentials
         - loops through each unique credentials and sends batches to Langsmith


        This was added to support key/team based logging on langsmith
        """
        if not self.log_queue:
            return

        batch_groups = self._group_batches_by_credentials()
        for batch_group in batch_groups.values():
            await self._log_batch_on_langsmith(
                credentials=batch_group.credentials,
                queue_objects=batch_group.queue_objects,
            )

    async def _log_batch_on_langsmith(
        self,
        credentials: LangsmithCredentialsObject,
        queue_objects: List[LangsmithQueueObject],
    ):
        """
        Logs a batch of runs to Langsmith
        sends runs to /batch endpoint for the given credentials

        Args:
            credentials: LangsmithCredentialsObject
            queue_objects: List[LangsmithQueueObject]

        Returns: None

        Raises: Does not raise an exception, will only verbose_logger.exception()
        """
        langsmith_api_base = credentials["LANGSMITH_BASE_URL"]
        langsmith_api_key = credentials["LANGSMITH_API_KEY"]
        url = f"{langsmith_api_base}/runs/batch"
        headers = {"x-api-key": langsmith_api_key}
        elements_to_log = [queue_object["data"] for queue_object in queue_objects]

        try:
            verbose_logger.debug(
                "Sending batch of %s runs to Langsmith", len(elements_to_log)
            )
            response = await self.async_httpx_client.post(
                url=url,
                json={"post": elements_to_log},
                headers=headers,
            )
            response.raise_for_status()

            if response.status_code >= 300:
                verbose_logger.error(
                    f"Langsmith Error: {response.status_code} - {response.text}"
                )
            else:
                verbose_logger.debug(
                    f"Batch of {len(self.log_queue)} runs successfully created"
                )
        except httpx.HTTPStatusError as e:
            verbose_logger.exception(
                f"Langsmith HTTP Error: {e.response.status_code} - {e.response.text}"
            )
        except Exception:
            verbose_logger.exception(
                f"Langsmith Layer Error - {traceback.format_exc()}"
            )

    def _group_batches_by_credentials(self) -> Dict[CredentialsKey, BatchGroup]:
        """Groups queue objects by credentials using a proper key structure"""
        log_queue_by_credentials: Dict[CredentialsKey, BatchGroup] = {}

        for queue_object in self.log_queue:
            credentials = queue_object["credentials"]
            key = CredentialsKey(
                api_key=credentials["LANGSMITH_API_KEY"],
                project=credentials["LANGSMITH_PROJECT"],
                base_url=credentials["LANGSMITH_BASE_URL"],
            )

            if key not in log_queue_by_credentials:
                log_queue_by_credentials[key] = BatchGroup(
                    credentials=credentials, queue_objects=[]
                )

            log_queue_by_credentials[key].queue_objects.append(queue_object)

        return log_queue_by_credentials

    def _get_credentials_to_use_for_request(
        self, kwargs: Dict[str, Any]
    ) -> LangsmithCredentialsObject:
        """
        Handles key/team based logging

        If standard_callback_dynamic_params are provided, use those credentials.

        Otherwise, use the default credentials.
        """
        standard_callback_dynamic_params: Optional[StandardCallbackDynamicParams] = (
            kwargs.get("standard_callback_dynamic_params", None)
        )
        if standard_callback_dynamic_params is not None:
            credentials = self.get_credentials_from_env(
                langsmith_api_key=standard_callback_dynamic_params.get(
                    "langsmith_api_key", None
                ),
                langsmith_project=standard_callback_dynamic_params.get(
                    "langsmith_project", None
                ),
                langsmith_base_url=standard_callback_dynamic_params.get(
                    "langsmith_base_url", None
                ),
            )
        else:
            credentials = self.default_credentials
        return credentials

    def _send_batch(self):
        """Calls async_send_batch in an event loop"""
        if not self.log_queue:
            return

        try:
            # Try to get the existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an event loop, create a task
                asyncio.create_task(self.async_send_batch())
            else:
                # If no event loop is running, run the coroutine directly
                loop.run_until_complete(self.async_send_batch())
        except RuntimeError:
            # If we can't get an event loop, create a new one
            asyncio.run(self.async_send_batch())

    def get_run_by_id(self, run_id):

        langsmith_api_key = self.default_credentials["LANGSMITH_API_KEY"]

        langsmith_api_base = self.default_credentials["LANGSMITH_BASE_URL"]

        url = f"{langsmith_api_base}/runs/{run_id}"
        response = litellm.module_level_client.get(
            url=url,
            headers={"x-api-key": langsmith_api_key},
        )

        return response.json()

    def make_dot_order(self, run_id: str):
        st = datetime.now(timezone.utc)
        id_ = run_id
        return st.strftime("%Y%m%dT%H%M%S%fZ") + str(id_)
