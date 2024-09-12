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
from typing import Any, List, Optional, Union

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


def is_serializable(value):
    non_serializable_types = (
        types.CoroutineType,
        types.FunctionType,
        types.GeneratorType,
        BaseModel,
    )
    return not isinstance(value, non_serializable_types)


class LangsmithLogger(CustomBatchLogger):
    def __init__(self, **kwargs):
        self.langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        self.langsmith_project = os.getenv("LANGSMITH_PROJECT", "litellm-completion")
        self.langsmith_default_run_name = os.getenv(
            "LANGSMITH_DEFAULT_RUN_NAME", "LLMRun"
        )
        self.langsmith_base_url = os.getenv(
            "LANGSMITH_BASE_URL", "https://api.smith.langchain.com"
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

    def _prepare_log_data(self, kwargs, response_obj, start_time, end_time):
        import datetime
        from datetime import datetime as dt
        from datetime import timezone

        metadata = kwargs.get("litellm_params", {}).get("metadata", {}) or {}
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

        project_name = metadata.get("project_name", self.langsmith_project)
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

        try:
            start_time = kwargs["start_time"].astimezone(timezone.utc).isoformat()
            end_time = kwargs["end_time"].astimezone(timezone.utc).isoformat()
        except:
            start_time = datetime.datetime.utcnow().isoformat()
            end_time = datetime.datetime.utcnow().isoformat()

        # filter out kwargs to not include any dicts, langsmith throws an erros when trying to log kwargs
        logged_kwargs = LangsmithInputs(**kwargs)
        kwargs = logged_kwargs.model_dump()

        new_kwargs = {}
        for key in kwargs:
            value = kwargs[key]
            if key == "start_time" or key == "end_time" or value is None:
                pass
            elif key == "original_response" and not isinstance(value, str):
                new_kwargs[key] = str(value)
            elif type(value) == datetime.datetime:
                new_kwargs[key] = value.isoformat()
            elif type(value) != dict and is_serializable(value=value):
                new_kwargs[key] = value
            elif not is_serializable(value=value):
                continue

        if isinstance(response_obj, BaseModel):
            try:
                response_obj = response_obj.model_dump()
            except:
                response_obj = response_obj.dict()  # type: ignore

        data = {
            "name": run_name,
            "run_type": "llm",  # this should always be llm, since litellm always logs llm calls. Langsmith allow us to log "chain"
            "inputs": new_kwargs,
            "outputs": response_obj,
            "session_name": project_name,
            "start_time": start_time,
            "end_time": end_time,
            "tags": tags,
            "extra": metadata,
        }

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
            run_id = uuid.uuid4()
            data["id"] = str(run_id)
            data["trace_id"] = str(run_id)
            data["dotted_order"] = self.make_dot_order(run_id=run_id)

        verbose_logger.debug("Langsmith Logging data on langsmith: %s", data)

        return data

    def _send_batch(self):
        if not self.log_queue:
            return

        url = f"{self.langsmith_base_url}/runs/batch"
        headers = {"x-api-key": self.langsmith_api_key}

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
        except Exception as e:
            verbose_logger.error(f"Langsmith Layer Error - {traceback.format_exc()}")

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            sampling_rate = (
                float(os.getenv("LANGSMITH_SAMPLING_RATE"))
                if os.getenv("LANGSMITH_SAMPLING_RATE") is not None
                and os.getenv("LANGSMITH_SAMPLING_RATE").strip().isdigit()
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

        except:
            verbose_logger.error(f"Langsmith Layer Error - {traceback.format_exc()}")

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            sampling_rate = (
                float(os.getenv("LANGSMITH_SAMPLING_RATE"))
                if os.getenv("LANGSMITH_SAMPLING_RATE") is not None
                and os.getenv("LANGSMITH_SAMPLING_RATE").strip().isdigit()
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
        except:
            verbose_logger.error(f"Langsmith Layer Error - {traceback.format_exc()}")

    async def async_send_batch(self):
        """
        sends runs to /batch endpoint

        Sends runs from self.log_queue

        Returns: None

        Raises: Does not raise an exception, will only verbose_logger.exception()
        """
        import json

        if not self.log_queue:
            return

        url = f"{self.langsmith_base_url}/runs/batch"
        headers = {"x-api-key": self.langsmith_api_key}

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

        url = f"{self.langsmith_base_url}/runs/{run_id}"
        response = requests.get(
            url=url,
            headers={"x-api-key": self.langsmith_api_key},
        )

        return response.json()

    def make_dot_order(self, run_id: str):
        st = datetime.now(timezone.utc)
        id_ = run_id
        return st.strftime("%Y%m%dT%H%M%S%fZ") + str(id_)
