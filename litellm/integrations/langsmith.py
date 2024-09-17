#### What this does ####
#    On success, logs events to Langsmith
import asyncio
import os
import random
import time
import traceback
import types
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict, Union

import dotenv  # type: ignore
import httpx
import requests  # type: ignore
from pydantic import BaseModel  # type: ignore

import litellm
from litellm._logging import verbose_logger
from litellm.integrations.custom_batch_logger import CustomBatchLogger
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.types.utils import StandardLoggingPayload


class LangsmithInputs(BaseModel):
    model: Optional[str] = None
    messages: Optional[List[Any]] = None
    stream: Optional[bool] = None
    call_type: Optional[str] = None
    litellm_call_id: Optional[str] = None
    completion_start_time: Optional[datetime] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    custom_llm_provider: Optional[str] = None
    input: Optional[List[Any]] = None
    log_event_type: Optional[str] = None
    original_response: Optional[Any] = None
    response_cost: Optional[float] = None

    # LiteLLM Virtual Key specific fields
    user_api_key: Optional[str] = None
    user_api_key_user_id: Optional[str] = None
    user_api_key_team_alias: Optional[str] = None


class LangsmithCredentialsObject(TypedDict):
    LANGSMITH_API_KEY: str
    LANGSMITH_PROJECT: str
    LANGSMITH_BASE_URL: str


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
        asyncio.create_task(self.periodic_flush())
        self.flush_lock = asyncio.Lock()
        super().__init__(**kwargs, flush_lock=self.flush_lock)

    def get_credentials_from_env(
        self,
        langsmith_api_key: Optional[str],
        langsmith_project: Optional[str],
        langsmith_base_url: Optional[str],
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

    def _prepare_log_data(self, kwargs, response_obj, start_time, end_time):
        import json
        from datetime import datetime as dt

        try:
            _litellm_params = kwargs.get("litellm_params", {}) or {}
            metadata = _litellm_params.get("metadata", {}) or {}
            new_metadata = {}
            for key, value in metadata.items():
                if (
                    isinstance(value, list)
                    or isinstance(value, str)
                    or isinstance(value, int)
                    or isinstance(value, float)
                ):
                    new_metadata[key] = value
                elif isinstance(value, BaseModel):
                    new_metadata[key] = value.model_dump_json()
                elif isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, dt):
                            value[k] = v.isoformat()
                    new_metadata[key] = value

            metadata = new_metadata

            kwargs["user_api_key"] = metadata.get("user_api_key", None)
            kwargs["user_api_key_user_id"] = metadata.get("user_api_key_user_id", None)
            kwargs["user_api_key_team_alias"] = metadata.get(
                "user_api_key_team_alias", None
            )

            project_name = metadata.get(
                "project_name", self.default_credentials["LANGSMITH_PROJECT"]
            )
            run_name = metadata.get("run_name", self.langsmith_default_run_name)
            run_id = metadata.get("id", None)
            parent_run_id = metadata.get("parent_run_id", None)
            trace_id = metadata.get("trace_id", None)
            session_id = metadata.get("session_id", None)
            dotted_order = metadata.get("dotted_order", None)
            tags = metadata.get("tags", []) or []
            verbose_logger.debug(
                f"Langsmith Logging - project_name: {project_name}, run_name {run_name}"
            )

            # filter out kwargs to not include any dicts, langsmith throws an erros when trying to log kwargs
            # logged_kwargs = LangsmithInputs(**kwargs)
            # kwargs = logged_kwargs.model_dump()

            # new_kwargs = {}
            # Ensure everything in the payload is converted to str
            payload: Optional[StandardLoggingPayload] = kwargs.get(
                "standard_logging_object", None
            )

            if payload is None:
                raise Exception("Error logging request payload. Payload=none.")

            new_kwargs = payload
            metadata = payload[
                "metadata"
            ]  # ensure logged metadata is json serializable

            data = {
                "name": run_name,
                "run_type": "llm",  # this should always be llm, since litellm always logs llm calls. Langsmith allow us to log "chain"
                "inputs": new_kwargs,
                "outputs": new_kwargs["response"],
                "session_name": project_name,
                "start_time": new_kwargs["startTime"],
                "end_time": new_kwargs["endTime"],
                "tags": tags,
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

            if "id" not in data or data["id"] is None:
                """
                for /batch langsmith requires id, trace_id and dotted_order passed as params
                """
                run_id = str(uuid.uuid4())
                data["id"] = str(run_id)
                data["trace_id"] = str(run_id)
                data["dotted_order"] = self.make_dot_order(run_id=run_id)

            verbose_logger.debug("Langsmith Logging data on langsmith: %s", data)

            return data
        except Exception:
            raise

    def _send_batch(self):
        if not self.log_queue:
            return

        langsmith_api_key = self.default_credentials["LANGSMITH_API_KEY"]
        langsmith_api_base = self.default_credentials["LANGSMITH_BASE_URL"]

        url = f"{langsmith_api_base}/runs/batch"

        headers = {"x-api-key": langsmith_api_key}

        try:
            response = requests.post(
                url=url,
                json=self.log_queue,
                headers=headers,
            )

            if response.status_code >= 300:
                verbose_logger.error(
                    f"Langsmith Error: {response.status_code} - {response.text}"
                )
            else:
                verbose_logger.debug(
                    f"Batch of {len(self.log_queue)} runs successfully created"
                )

            self.log_queue.clear()
        except Exception:
            verbose_logger.exception("Langsmith Layer Error - Error sending batch.")

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

        langsmith_api_base = self.default_credentials["LANGSMITH_BASE_URL"]

        url = f"{langsmith_api_base}/runs/batch"

        langsmith_api_key = self.default_credentials["LANGSMITH_API_KEY"]

        headers = {"x-api-key": langsmith_api_key}

        try:
            response = await self.async_httpx_client.post(
                url=url,
                json={
                    "post": self.log_queue,
                },
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
        except Exception as e:
            verbose_logger.exception(
                f"Langsmith Layer Error - {traceback.format_exc()}"
            )

    def get_run_by_id(self, run_id):

        langsmith_api_key = self.default_credentials["LANGSMITH_API_KEY"]

        langsmith_api_base = self.default_credentials["LANGSMITH_BASE_URL"]

        url = f"{langsmith_api_base}/runs/{run_id}"
        response = requests.get(
            url=url,
            headers={"x-api-key": langsmith_api_key},
        )

        return response.json()

    def make_dot_order(self, run_id: str):
        st = datetime.now(timezone.utc)
        id_ = run_id
        return st.strftime("%Y%m%dT%H%M%S%fZ") + str(id_)
