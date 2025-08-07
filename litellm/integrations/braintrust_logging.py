# What is this?
## Log success + failure events to Braintrust

import copy
import os
from datetime import datetime
from typing import Dict, Optional

import httpx
from pydantic import BaseModel

import litellm
from litellm import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import (
    HTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.utils import print_verbose

global_braintrust_http_handler = get_async_httpx_client(
    llm_provider=httpxSpecialProvider.LoggingCallback
)
global_braintrust_sync_http_handler = HTTPHandler()
API_BASE = "https://api.braintrustdata.com/v1"


def get_utc_datetime():
    import datetime as dt
    from datetime import datetime

    if hasattr(dt, "UTC"):
        return datetime.now(dt.UTC)  # type: ignore
    else:
        return datetime.utcnow()  # type: ignore


class BraintrustLogger(CustomLogger):
    def __init__(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None
    ) -> None:
        super().__init__()
        self.validate_environment(api_key=api_key)
        self.api_base = api_base or os.getenv("BRAINTRUST_API_BASE") or API_BASE
        self.default_project_id = None
        self.api_key: str = api_key or os.getenv("BRAINTRUST_API_KEY")  # type: ignore
        self.headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        }
        self._project_id_cache: Dict[
            str, str
        ] = {}  # Cache mapping project names to IDs

    def validate_environment(self, api_key: Optional[str]):
        """
        Expects
        BRAINTRUST_API_KEY

        in the environment
        """
        missing_keys = []
        if api_key is None and os.getenv("BRAINTRUST_API_KEY", None) is None:
            missing_keys.append("BRAINTRUST_API_KEY")

        if len(missing_keys) > 0:
            raise Exception("Missing keys={} in environment.".format(missing_keys))

    def get_project_id_sync(self, project_name: str) -> str:
        """
        Get project ID from name, using cache if available.
        If project doesn't exist, creates it.
        """
        if project_name in self._project_id_cache:
            return self._project_id_cache[project_name]

        try:
            response = global_braintrust_sync_http_handler.post(
                f"{self.api_base}/project",
                headers=self.headers,
                json={"name": project_name},
            )
            project_dict = response.json()
            project_id = project_dict["id"]
            self._project_id_cache[project_name] = project_id
            return project_id
        except httpx.HTTPStatusError as e:
            raise Exception(f"Failed to register project: {e.response.text}")

    async def get_project_id_async(self, project_name: str) -> str:
        """
        Async version of get_project_id_sync
        """
        if project_name in self._project_id_cache:
            return self._project_id_cache[project_name]

        try:
            response = await global_braintrust_http_handler.post(
                f"{self.api_base}/project/register",
                headers=self.headers,
                json={"name": project_name},
            )
            project_dict = response.json()
            project_id = project_dict["id"]
            self._project_id_cache[project_name] = project_id
            return project_id
        except httpx.HTTPStatusError as e:
            raise Exception(f"Failed to register project: {e.response.text}")

    @staticmethod
    def add_metadata_from_header(litellm_params: dict, metadata: dict) -> dict:
        """
        Adds metadata from proxy request headers to Braintrust logging if keys start with "braintrust_"
        and overwrites litellm_params.metadata if already included.

        For example if you want to append your trace to an existing `trace_id` via header, send
        `headers: { ..., langfuse_existing_trace_id: your-existing-trace-id }` via proxy request.
        """
        if litellm_params is None:
            return metadata

        if litellm_params.get("proxy_server_request") is None:
            return metadata

        if metadata is None:
            metadata = {}

        proxy_headers = (
            litellm_params.get("proxy_server_request", {}).get("headers", {}) or {}
        )

        for metadata_param_key in proxy_headers:
            if metadata_param_key.startswith("braintrust"):
                trace_param_key = metadata_param_key.replace("braintrust", "", 1)
                if trace_param_key in metadata:
                    verbose_logger.warning(
                        f"Overwriting Braintrust `{trace_param_key}` from request header"
                    )
                else:
                    verbose_logger.debug(
                        f"Found Braintrust `{trace_param_key}` in request header"
                    )
                metadata[trace_param_key] = proxy_headers.get(metadata_param_key)

        return metadata

    async def create_default_project_and_experiment(self):
        project = await global_braintrust_http_handler.post(
            f"{self.api_base}/project", headers=self.headers, json={"name": "litellm"}
        )

        project_dict = project.json()

        self.default_project_id = project_dict["id"]

    def create_sync_default_project_and_experiment(self):
        project = global_braintrust_sync_http_handler.post(
            f"{self.api_base}/project", headers=self.headers, json={"name": "litellm"}
        )

        project_dict = project.json()

        self.default_project_id = project_dict["id"]

    def log_success_event(  # noqa: PLR0915
        self, kwargs, response_obj, start_time, end_time
    ):
        verbose_logger.debug("REACHES BRAINTRUST SUCCESS")
        try:
            litellm_call_id = kwargs.get("litellm_call_id")
            prompt = {"messages": kwargs.get("messages")}
            output = None
            choices = []
            if response_obj is not None and (
                kwargs.get("call_type", None) == "embedding"
                or isinstance(response_obj, litellm.EmbeddingResponse)
            ):
                output = None
            elif response_obj is not None and isinstance(
                response_obj, litellm.ModelResponse
            ):
                output = response_obj["choices"][0]["message"].json()
                choices = response_obj["choices"]
            elif response_obj is not None and isinstance(
                response_obj, litellm.TextCompletionResponse
            ):
                output = response_obj.choices[0].text
                choices = response_obj.choices
            elif response_obj is not None and isinstance(
                response_obj, litellm.ImageResponse
            ):
                output = response_obj["data"]

            litellm_params = kwargs.get("litellm_params", {})
            metadata = (
                litellm_params.get("metadata", {}) or {}
            )  # if litellm_params['metadata'] == None
            metadata = self.add_metadata_from_header(litellm_params, metadata)
            clean_metadata = {}
            try:
                metadata = copy.deepcopy(
                    metadata
                )  # Avoid modifying the original metadata
            except Exception:
                new_metadata = {}
                for key, value in metadata.items():
                    if (
                        isinstance(value, list)
                        or isinstance(value, dict)
                        or isinstance(value, str)
                        or isinstance(value, int)
                        or isinstance(value, float)
                    ):
                        new_metadata[key] = copy.deepcopy(value)
                metadata = new_metadata

            # Get project_id from metadata or create default if needed
            project_id = metadata.get("project_id")
            if project_id is None:
                project_name = metadata.get("project_name")
                project_id = (
                    self.get_project_id_sync(project_name) if project_name else None
                )

            if project_id is None:
                if self.default_project_id is None:
                    self.create_sync_default_project_and_experiment()
                project_id = self.default_project_id

            tags = []
            if isinstance(metadata, dict):
                for key, value in metadata.items():
                    # generate langfuse tags - Default Tags sent to Langfuse from LiteLLM Proxy
                    if (
                        litellm.langfuse_default_tags is not None
                        and isinstance(litellm.langfuse_default_tags, list)
                        and key in litellm.langfuse_default_tags
                    ):
                        tags.append(f"{key}:{value}")

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

            cost = kwargs.get("response_cost", None)
            if cost is not None:
                clean_metadata["litellm_response_cost"] = cost

            # metadata.model is required for braintrust to calculate the "Estimated cost" metric
            litellm_model = kwargs.get("model", None)
            if litellm_model is not None:
                clean_metadata["model"] = litellm_model

            metrics: Optional[dict] = None
            usage_obj = getattr(response_obj, "usage", None)
            if usage_obj and isinstance(usage_obj, litellm.Usage):
                litellm.utils.get_logging_id(start_time, response_obj)
                metrics = {
                    "prompt_tokens": usage_obj.prompt_tokens,
                    "completion_tokens": usage_obj.completion_tokens,
                    "total_tokens": usage_obj.total_tokens,
                    "total_cost": cost,
                    "time_to_first_token": end_time.timestamp()
                    - start_time.timestamp(),
                    "start": start_time.timestamp(),
                    "end": end_time.timestamp(),
                }

            request_data = {
                "id": litellm_call_id,
                "input": prompt["messages"],
                "metadata": clean_metadata,
                "tags": tags,
                "span_attributes": {"name": "Chat Completion", "type": "llm"},
            }
            if choices is not None:
                request_data["output"] = [choice.dict() for choice in choices]
            else:
                request_data["output"] = output

            if metrics is not None:
                request_data["metrics"] = metrics

            try:
                print_verbose(
                    f"global_braintrust_sync_http_handler.post: {global_braintrust_sync_http_handler.post}"
                )
                global_braintrust_sync_http_handler.post(
                    url=f"{self.api_base}/project_logs/{project_id}/insert",
                    json={"events": [request_data]},
                    headers=self.headers,
                )
            except httpx.HTTPStatusError as e:
                raise Exception(e.response.text)
        except Exception as e:
            raise e  # don't use verbose_logger.exception, if exception is raised

    async def async_log_success_event(  # noqa: PLR0915
        self, kwargs, response_obj, start_time, end_time
    ):
        verbose_logger.debug("REACHES BRAINTRUST SUCCESS")
        try:
            litellm_call_id = kwargs.get("litellm_call_id")
            prompt = {"messages": kwargs.get("messages")}
            output = None
            choices = []
            if response_obj is not None and (
                kwargs.get("call_type", None) == "embedding"
                or isinstance(response_obj, litellm.EmbeddingResponse)
            ):
                output = None
            elif response_obj is not None and isinstance(
                response_obj, litellm.ModelResponse
            ):
                output = response_obj["choices"][0]["message"].json()
                choices = response_obj["choices"]
            elif response_obj is not None and isinstance(
                response_obj, litellm.TextCompletionResponse
            ):
                output = response_obj.choices[0].text
                choices = response_obj.choices
            elif response_obj is not None and isinstance(
                response_obj, litellm.ImageResponse
            ):
                output = response_obj["data"]

            litellm_params = kwargs.get("litellm_params", {})
            metadata = (
                litellm_params.get("metadata", {}) or {}
            )  # if litellm_params['metadata'] == None
            metadata = self.add_metadata_from_header(litellm_params, metadata)
            clean_metadata = {}
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
                        if isinstance(v, datetime):
                            value[k] = v.isoformat()
                    new_metadata[key] = value

            # Get project_id from metadata or create default if needed
            project_id = metadata.get("project_id")
            if project_id is None:
                project_name = metadata.get("project_name")
                project_id = (
                    await self.get_project_id_async(project_name)
                    if project_name
                    else None
                )

            if project_id is None:
                if self.default_project_id is None:
                    await self.create_default_project_and_experiment()
                project_id = self.default_project_id

            tags = []
            if isinstance(metadata, dict):
                for key, value in metadata.items():
                    # generate langfuse tags - Default Tags sent to Langfuse from LiteLLM Proxy
                    if (
                        litellm.langfuse_default_tags is not None
                        and isinstance(litellm.langfuse_default_tags, list)
                        and key in litellm.langfuse_default_tags
                    ):
                        tags.append(f"{key}:{value}")

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

            cost = kwargs.get("response_cost", None)
            if cost is not None:
                clean_metadata["litellm_response_cost"] = cost

            # metadata.model is required for braintrust to calculate the "Estimated cost" metric
            litellm_model = kwargs.get("model", None)
            if litellm_model is not None:
                clean_metadata["model"] = litellm_model

            metrics: Optional[dict] = None
            usage_obj = getattr(response_obj, "usage", None)
            if usage_obj and isinstance(usage_obj, litellm.Usage):
                litellm.utils.get_logging_id(start_time, response_obj)
                metrics = {
                    "prompt_tokens": usage_obj.prompt_tokens,
                    "completion_tokens": usage_obj.completion_tokens,
                    "total_tokens": usage_obj.total_tokens,
                    "total_cost": cost,
                    "start": start_time.timestamp(),
                    "end": end_time.timestamp(),
                }

                api_call_start_time = kwargs.get("api_call_start_time")
                completion_start_time = kwargs.get("completion_start_time")

                if (
                    api_call_start_time is not None
                    and completion_start_time is not None
                ):
                    metrics["time_to_first_token"] = (
                        completion_start_time.timestamp()
                        - api_call_start_time.timestamp()
                    )

            request_data = {
                "id": litellm_call_id,
                "input": prompt["messages"],
                "output": output,
                "metadata": clean_metadata,
                "tags": tags,
                "span_attributes": {"name": "Chat Completion", "type": "llm"},
            }
            if choices is not None:
                request_data["output"] = [choice.dict() for choice in choices]
            else:
                request_data["output"] = output

            if metrics is not None:
                request_data["metrics"] = metrics

            if metrics is not None:
                request_data["metrics"] = metrics

            try:
                await global_braintrust_http_handler.post(
                    url=f"{self.api_base}/project_logs/{project_id}/insert",
                    json={"events": [request_data]},
                    headers=self.headers,
                )
            except httpx.HTTPStatusError as e:
                raise Exception(e.response.text)
        except Exception as e:
            raise e  # don't use verbose_logger.exception, if exception is raised

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        return super().log_failure_event(kwargs, response_obj, start_time, end_time)
