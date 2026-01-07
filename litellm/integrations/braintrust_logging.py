# What is this?
## Log success + failure events to Braintrust

import os
from datetime import datetime
from typing import Dict, Optional

import httpx

import litellm
from litellm import verbose_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.custom_httpx.http_handler import (
    HTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)
from litellm.utils import print_verbose

API_BASE = "https://api.braintrustdata.com/v1"


def get_utc_datetime():
    import datetime as dt

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
        self._project_id_cache: Dict[str, str] = (
            {}
        )  # Cache mapping project names to IDs
        self.global_braintrust_http_handler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )
        self.global_braintrust_sync_http_handler = HTTPHandler()

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
            response = self.global_braintrust_sync_http_handler.post(
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
            response = await self.global_braintrust_http_handler.post(
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

    async def create_default_project_and_experiment(self):
        project = await self.global_braintrust_http_handler.post(
            f"{self.api_base}/project", headers=self.headers, json={"name": "litellm"}
        )

        project_dict = project.json()

        self.default_project_id = project_dict["id"]

    def create_sync_default_project_and_experiment(self):
        project = self.global_braintrust_sync_http_handler.post(
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
            standard_logging_object = kwargs.get("standard_logging_object", {})
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

            litellm_params = kwargs.get("litellm_params", {}) or {}
            dynamic_metadata = litellm_params.get("metadata", {}) or {}

            # Get project_id from metadata or create default if needed
            project_id = dynamic_metadata.get("project_id")
            if project_id is None:
                project_name = dynamic_metadata.get("project_name")
                project_id = (
                    self.get_project_id_sync(project_name) if project_name else None
                )

            if project_id is None:
                if self.default_project_id is None:
                    self.create_sync_default_project_and_experiment()
                project_id = self.default_project_id

            tags = []

            if isinstance(dynamic_metadata, dict):
                for key, value in dynamic_metadata.items():
                    # generate langfuse tags - Default Tags sent to Langfuse from LiteLLM Proxy
                    if (
                        litellm.langfuse_default_tags is not None
                        and isinstance(litellm.langfuse_default_tags, list)
                        and key in litellm.langfuse_default_tags
                    ):
                        tags.append(f"{key}:{value}")

                    if (
                        isinstance(value, str) and key not in standard_logging_object
                    ):  # support logging dynamic metadata to braintrust
                        standard_logging_object[key] = value

            cost = kwargs.get("response_cost", None)

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

            # Allow metadata override for span name
            span_name = dynamic_metadata.get("span_name", "Chat Completion")
            
            # Span parents is a special case
            span_parents = dynamic_metadata.get("span_parents")

            # Convert comma-separated string to list if present
            if span_parents:
                span_parents = [s.strip() for s in span_parents.split(",") if s.strip()]

            # Add optional span attributes only if present
            span_attributes = {
                "span_id": dynamic_metadata.get("span_id"),
                "root_span_id": dynamic_metadata.get("root_span_id"),
                "span_parents": span_parents,
            }

            request_data = {
                "id": litellm_call_id,
                "input": prompt["messages"],
                "metadata": standard_logging_object,
                "tags": tags,
                "span_attributes": {"name": span_name, "type": "llm"},
            }
            
            # Only add those that are not None (or falsy)
            for key, value in span_attributes.items():
                if value:
                    request_data[key] = value

            if choices is not None:
                request_data["output"] = [choice.dict() for choice in choices]
            else:
                request_data["output"] = output

            if metrics is not None:
                request_data["metrics"] = metrics

            try:
                print_verbose(
                    f"self.global_braintrust_sync_http_handler.post: {self.global_braintrust_sync_http_handler.post}"
                )
                self.global_braintrust_sync_http_handler.post(
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
            standard_logging_object = kwargs.get("standard_logging_object", {})
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
            dynamic_metadata = litellm_params.get("metadata", {}) or {}

            # Get project_id from metadata or create default if needed
            project_id = dynamic_metadata.get("project_id")
            if project_id is None:
                project_name = dynamic_metadata.get("project_name")
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

            if isinstance(dynamic_metadata, dict):
                for key, value in dynamic_metadata.items():
                    # generate langfuse tags - Default Tags sent to Langfuse from LiteLLM Proxy
                    if (
                        litellm.langfuse_default_tags is not None
                        and isinstance(litellm.langfuse_default_tags, list)
                        and key in litellm.langfuse_default_tags
                    ):
                        tags.append(f"{key}:{value}")

                    if (
                        isinstance(value, str) and key not in standard_logging_object
                    ):  # support logging dynamic metadata to braintrust
                        standard_logging_object[key] = value

            cost = kwargs.get("response_cost", None)

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

            # Allow metadata override for span name
            span_name = dynamic_metadata.get("span_name", "Chat Completion")

            request_data = {
                "id": litellm_call_id,
                "input": prompt["messages"],
                "output": output,
                "metadata": standard_logging_object,
                "tags": tags,
                "span_attributes": {"name": span_name, "type": "llm"},
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
                await self.global_braintrust_http_handler.post(
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
