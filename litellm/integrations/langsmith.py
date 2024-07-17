#### What this does ####
#    On success, logs events to Langsmith
import asyncio
import os
import traceback
import types
from datetime import datetime
from typing import Any, List, Optional

import dotenv  # type: ignore
import requests  # type: ignore
from pydantic import BaseModel  # type: ignore

from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


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
    original_response: Optional[str] = None
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


class LangsmithLogger(CustomLogger):
    # Class variables or attributes
    def __init__(self):
        self.langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        self.langsmith_project = os.getenv("LANGSMITH_PROJECT", "litellm-completion")
        self.langsmith_default_run_name = os.getenv(
            "LANGSMITH_DEFAULT_RUN_NAME", "LLMRun"
        )
        self.langsmith_base_url = os.getenv(
            "LANGSMITH_BASE_URL", "https://api.smith.langchain.com"
        )
        self.async_httpx_client = AsyncHTTPHandler()

    def _prepare_log_data(
        self, kwargs, response_obj, start_time, end_time, print_verbose
    ):
        import datetime
        from datetime import timezone

        metadata = kwargs.get("litellm_params", {}).get("metadata", {}) or {}

        kwargs["user_api_key"] = metadata.get("user_api_key", None)
        kwargs["user_api_key_user_id"] = metadata.get("user_api_key_user_id", None)
        kwargs["user_api_key_team_alias"] = metadata.get(
            "user_api_key_team_alias", None
        )

        project_name = metadata.get("project_name", self.langsmith_project)
        run_name = metadata.get("run_name", self.langsmith_default_run_name)
        run_id = metadata.get("id", None)
        print_verbose(
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
            elif type(value) == datetime.datetime:
                new_kwargs[key] = value.isoformat()
            elif type(value) != dict and is_serializable(value=value):
                new_kwargs[key] = value

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
            "id": run_id,
        }

        return data

    async def async_log_event(
        self, kwargs, response_obj, start_time, end_time, print_verbose
    ):
        try:
            data = self._prepare_log_data(
                kwargs, response_obj, start_time, end_time, print_verbose
            )
            url = f"{self.langsmith_base_url}/runs"
            print_verbose(f"Langsmith Logging - About to send data to {url} ...")

            headers = {"x-api-key": self.langsmith_api_key}
            response = await self.async_httpx_client.post(
                url=url, json=data, headers=headers
            )

            if response.status_code >= 300:
                print_verbose(f"Error: {response.status_code}")
            else:
                print_verbose("Run successfully created")
            print_verbose(
                f"Langsmith Layer Logging - final response object: {response_obj}. Response text from langsmith={response.text}"
            )
        except:
            print_verbose(f"Langsmith Layer Error - {traceback.format_exc()}")

    def log_event(self, kwargs, response_obj, start_time, end_time, print_verbose):
        try:
            data = self._prepare_log_data(
                kwargs, response_obj, start_time, end_time, print_verbose
            )
            url = f"{self.langsmith_base_url}/runs"
            print_verbose(f"Langsmith Logging - About to send data to {url} ...")

            response = requests.post(
                url=url,
                json=data,
                headers={"x-api-key": self.langsmith_api_key},
            )

            if response.status_code >= 300:
                print_verbose(f"Error: {response.status_code}")
            else:
                print_verbose("Run successfully created")
            print_verbose(
                f"Langsmith Layer Logging - final response object: {response_obj}. Response text from langsmith={response.text}"
            )
        except:
            print_verbose(f"Langsmith Layer Error - {traceback.format_exc()}")

    def get_run_by_id(self, run_id):

        url = f"{self.langsmith_base_url}/runs/{run_id}"
        response = requests.get(
            url=url,
            headers={"x-api-key": self.langsmith_api_key},
        )

        return response.json()
