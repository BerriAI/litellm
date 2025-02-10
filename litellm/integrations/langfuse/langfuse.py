#### What this does ####
#    On success, logs events to Langfuse
import copy
import os
import traceback
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast

from packaging.version import Version

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.redact_messages import redact_user_api_key_info
from litellm.llms.custom_httpx.http_handler import _get_httpx_client
from litellm.secret_managers.main import str_to_bool
from litellm.types.integrations.langfuse import *
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.types.utils import (
    EmbeddingResponse,
    ImageResponse,
    ModelResponse,
    RerankResponse,
    StandardLoggingPayload,
    StandardLoggingPromptManagementMetadata,
    TextCompletionResponse,
    TranscriptionResponse,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import DynamicLoggingCache
else:
    DynamicLoggingCache = Any


class LangFuseLogger:
    # Class variables or attributes
    def __init__(
        self,
        langfuse_public_key=None,
        langfuse_secret=None,
        langfuse_host=None,
        flush_interval=1,
    ):
        try:
            import langfuse
            from langfuse import Langfuse
        except Exception as e:
            raise Exception(
                f"\033[91mLangfuse not installed, try running 'pip install langfuse' to fix this error: {e}\n{traceback.format_exc()}\033[0m"
            )
        # Instance variables
        self.secret_key = langfuse_secret or os.getenv("LANGFUSE_SECRET_KEY")
        self.public_key = langfuse_public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
        self.langfuse_host = langfuse_host or os.getenv(
            "LANGFUSE_HOST", "https://cloud.langfuse.com"
        )
        if not (
            self.langfuse_host.startswith("http://")
            or self.langfuse_host.startswith("https://")
        ):
            # add http:// if unset, assume communicating over private network - e.g. render
            self.langfuse_host = "http://" + self.langfuse_host
        self.langfuse_release = os.getenv("LANGFUSE_RELEASE")
        self.langfuse_debug = os.getenv("LANGFUSE_DEBUG")
        self.langfuse_flush_interval = LangFuseLogger._get_langfuse_flush_interval(
            flush_interval
        )
        http_client = _get_httpx_client()
        self.langfuse_client = http_client.client

        parameters = {
            "public_key": self.public_key,
            "secret_key": self.secret_key,
            "host": self.langfuse_host,
            "release": self.langfuse_release,
            "debug": self.langfuse_debug,
            "flush_interval": self.langfuse_flush_interval,  # flush interval in seconds
            "httpx_client": self.langfuse_client,
        }
        self.langfuse_sdk_version: str = langfuse.version.__version__

        if Version(self.langfuse_sdk_version) >= Version("2.6.0"):
            parameters["sdk_integration"] = "litellm"

        self.Langfuse = Langfuse(**parameters)

        # set the current langfuse project id in the environ
        # this is used by Alerting to link to the correct project
        try:
            project_id = self.Langfuse.client.projects.get().data[0].id
            os.environ["LANGFUSE_PROJECT_ID"] = project_id
        except Exception:
            project_id = None

        if os.getenv("UPSTREAM_LANGFUSE_SECRET_KEY") is not None:
            upstream_langfuse_debug = (
                str_to_bool(self.upstream_langfuse_debug)
                if self.upstream_langfuse_debug is not None
                else None
            )
            self.upstream_langfuse_secret_key = os.getenv(
                "UPSTREAM_LANGFUSE_SECRET_KEY"
            )
            self.upstream_langfuse_public_key = os.getenv(
                "UPSTREAM_LANGFUSE_PUBLIC_KEY"
            )
            self.upstream_langfuse_host = os.getenv("UPSTREAM_LANGFUSE_HOST")
            self.upstream_langfuse_release = os.getenv("UPSTREAM_LANGFUSE_RELEASE")
            self.upstream_langfuse_debug = os.getenv("UPSTREAM_LANGFUSE_DEBUG")
            self.upstream_langfuse = Langfuse(
                public_key=self.upstream_langfuse_public_key,
                secret_key=self.upstream_langfuse_secret_key,
                host=self.upstream_langfuse_host,
                release=self.upstream_langfuse_release,
                debug=(
                    upstream_langfuse_debug
                    if upstream_langfuse_debug is not None
                    else False
                ),
            )
        else:
            self.upstream_langfuse = None

    @staticmethod
    def add_metadata_from_header(litellm_params: dict, metadata: dict) -> dict:
        """
        Adds metadata from proxy request headers to Langfuse logging if keys start with "langfuse_"
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
            if metadata_param_key.startswith("langfuse_"):
                trace_param_key = metadata_param_key.replace("langfuse_", "", 1)
                if trace_param_key in metadata:
                    verbose_logger.warning(
                        f"Overwriting Langfuse `{trace_param_key}` from request header"
                    )
                else:
                    verbose_logger.debug(
                        f"Found Langfuse `{trace_param_key}` in request header"
                    )
                metadata[trace_param_key] = proxy_headers.get(metadata_param_key)

        return metadata

    def log_event_on_langfuse(
        self,
        kwargs: dict,
        response_obj: Union[
            None,
            dict,
            EmbeddingResponse,
            ModelResponse,
            TextCompletionResponse,
            ImageResponse,
            TranscriptionResponse,
            RerankResponse,
            HttpxBinaryResponseContent,
        ],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        level: str = "DEFAULT",
        status_message: Optional[str] = None,
    ) -> dict:
        """
        Logs a success or error event on Langfuse
        """
        try:
            verbose_logger.debug(
                f"Langfuse Logging - Enters logging function for model {kwargs}"
            )

            # set default values for input/output for langfuse logging
            input = None
            output = None

            litellm_params = kwargs.get("litellm_params", {})
            litellm_call_id = kwargs.get("litellm_call_id", None)
            metadata = (
                litellm_params.get("metadata", {}) or {}
            )  # if litellm_params['metadata'] == None
            metadata = self.add_metadata_from_header(litellm_params, metadata)
            optional_params = copy.deepcopy(kwargs.get("optional_params", {}))

            prompt = {"messages": kwargs.get("messages")}

            functions = optional_params.pop("functions", None)
            tools = optional_params.pop("tools", None)
            if functions is not None:
                prompt["functions"] = functions
            if tools is not None:
                prompt["tools"] = tools

            # langfuse only accepts str, int, bool, float for logging
            for param, value in optional_params.items():
                if not isinstance(value, (str, int, bool, float)):
                    try:
                        optional_params[param] = str(value)
                    except Exception:
                        # if casting value to str fails don't block logging
                        pass

            input, output = self._get_langfuse_input_output_content(
                kwargs=kwargs,
                response_obj=response_obj,
                prompt=prompt,
                level=level,
                status_message=status_message,
            )
            verbose_logger.debug(
                f"OUTPUT IN LANGFUSE: {output}; original: {response_obj}"
            )
            trace_id = None
            generation_id = None
            if self._is_langfuse_v2():
                trace_id, generation_id = self._log_langfuse_v2(
                    user_id=user_id,
                    metadata=metadata,
                    litellm_params=litellm_params,
                    output=output,
                    start_time=start_time,
                    end_time=end_time,
                    kwargs=kwargs,
                    optional_params=optional_params,
                    input=input,
                    response_obj=response_obj,
                    level=level,
                    litellm_call_id=litellm_call_id,
                )
            elif response_obj is not None:
                self._log_langfuse_v1(
                    user_id=user_id,
                    metadata=metadata,
                    output=output,
                    start_time=start_time,
                    end_time=end_time,
                    kwargs=kwargs,
                    optional_params=optional_params,
                    input=input,
                    response_obj=response_obj,
                )
            verbose_logger.debug(
                f"Langfuse Layer Logging - final response object: {response_obj}"
            )
            verbose_logger.info("Langfuse Layer Logging - logging success")

            return {"trace_id": trace_id, "generation_id": generation_id}
        except Exception as e:
            verbose_logger.exception(
                "Langfuse Layer Error(): Exception occured - {}".format(str(e))
            )
            return {"trace_id": None, "generation_id": None}

    def _get_langfuse_input_output_content(
        self,
        kwargs: dict,
        response_obj: Union[
            None,
            dict,
            EmbeddingResponse,
            ModelResponse,
            TextCompletionResponse,
            ImageResponse,
            TranscriptionResponse,
            RerankResponse,
            HttpxBinaryResponseContent,
        ],
        prompt: dict,
        level: str,
        status_message: Optional[str],
    ) -> Tuple[Optional[dict], Optional[Union[str, dict, list]]]:
        """
        Get the input and output content for Langfuse logging

        Args:
            kwargs: The keyword arguments passed to the function
            response_obj: The response object returned by the function
            prompt: The prompt used to generate the response
            level: The level of the log message
            status_message: The status message of the log message

        Returns:
            input: The input content for Langfuse logging
            output: The output content for Langfuse logging
        """
        input = None
        output: Optional[Union[str, dict, List[Any]]] = None
        if (
            level == "ERROR"
            and status_message is not None
            and isinstance(status_message, str)
        ):
            input = prompt
            output = status_message
        elif response_obj is not None and (
            kwargs.get("call_type", None) == "embedding"
            or isinstance(response_obj, litellm.EmbeddingResponse)
        ):
            input = prompt
            output = None
        elif response_obj is not None and isinstance(
            response_obj, litellm.ModelResponse
        ):
            input = prompt
            output = self._get_chat_content_for_langfuse(response_obj)
        elif response_obj is not None and isinstance(
            response_obj, litellm.HttpxBinaryResponseContent
        ):
            input = prompt
            output = "speech-output"
        elif response_obj is not None and isinstance(
            response_obj, litellm.TextCompletionResponse
        ):
            input = prompt
            output = self._get_text_completion_content_for_langfuse(response_obj)
        elif response_obj is not None and isinstance(
            response_obj, litellm.ImageResponse
        ):
            input = prompt
            output = response_obj.get("data", None)
        elif response_obj is not None and isinstance(
            response_obj, litellm.TranscriptionResponse
        ):
            input = prompt
            output = response_obj.get("text", None)
        elif response_obj is not None and isinstance(
            response_obj, litellm.RerankResponse
        ):
            input = prompt
            output = response_obj.results
        elif (
            kwargs.get("call_type") is not None
            and kwargs.get("call_type") == "_arealtime"
            and response_obj is not None
            and isinstance(response_obj, list)
        ):
            input = kwargs.get("input")
            output = response_obj
        elif (
            kwargs.get("call_type") is not None
            and kwargs.get("call_type") == "pass_through_endpoint"
            and response_obj is not None
            and isinstance(response_obj, dict)
        ):
            input = prompt
            output = response_obj.get("response", "")
        return input, output

    async def _async_log_event(
        self, kwargs, response_obj, start_time, end_time, user_id
    ):
        """
        Langfuse SDK uses a background thread to log events

        This approach does not impact latency and runs in the background
        """

    def _is_langfuse_v2(self):
        import langfuse

        return Version(langfuse.version.__version__) >= Version("2.0.0")

    def _log_langfuse_v1(
        self,
        user_id,
        metadata,
        output,
        start_time,
        end_time,
        kwargs,
        optional_params,
        input,
        response_obj,
    ):
        from langfuse.model import CreateGeneration, CreateTrace  # type: ignore

        verbose_logger.warning(
            "Please upgrade langfuse to v2.0.0 or higher: https://github.com/langfuse/langfuse-python/releases/tag/v2.0.1"
        )

        trace = self.Langfuse.trace(  # type: ignore
            CreateTrace(  # type: ignore
                name=metadata.get("generation_name", "litellm-completion"),
                input=input,
                output=output,
                userId=user_id,
            )
        )

        trace.generation(
            CreateGeneration(
                name=metadata.get("generation_name", "litellm-completion"),
                startTime=start_time,
                endTime=end_time,
                model=kwargs["model"],
                modelParameters=optional_params,
                prompt=input,
                completion=output,
                usage={
                    "prompt_tokens": response_obj.usage.prompt_tokens,
                    "completion_tokens": response_obj.usage.completion_tokens,
                },
                metadata=metadata,
            )
        )

    def _log_langfuse_v2(  # noqa: PLR0915
        self,
        user_id: Optional[str],
        metadata: dict,
        litellm_params: dict,
        output: Optional[Union[str, dict, list]],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        kwargs: dict,
        optional_params: dict,
        input: Optional[dict],
        response_obj,
        level: str,
        litellm_call_id: Optional[str],
    ) -> tuple:
        verbose_logger.debug("Langfuse Layer Logging - logging to langfuse v2")

        try:
            metadata = metadata or {}
            standard_logging_object: Optional[StandardLoggingPayload] = cast(
                Optional[StandardLoggingPayload],
                kwargs.get("standard_logging_object", None),
            )
            tags = (
                self._get_langfuse_tags(standard_logging_object=standard_logging_object)
                if self._supports_tags()
                else []
            )

            if standard_logging_object is None:
                end_user_id = None
                prompt_management_metadata: Optional[
                    StandardLoggingPromptManagementMetadata
                ] = None
            else:
                end_user_id = standard_logging_object["metadata"].get(
                    "user_api_key_end_user_id", None
                )

                prompt_management_metadata = cast(
                    Optional[StandardLoggingPromptManagementMetadata],
                    standard_logging_object["metadata"].get(
                        "prompt_management_metadata", None
                    ),
                )

            # Clean Metadata before logging - never log raw metadata
            # the raw metadata can contain circular references which leads to infinite recursion
            # we clean out all extra litellm metadata params before logging
            clean_metadata: Dict[str, Any] = {}
            if prompt_management_metadata is not None:
                clean_metadata["prompt_management_metadata"] = (
                    prompt_management_metadata
                )
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

            # Add default langfuse tags
            tags = self.add_default_langfuse_tags(
                tags=tags, kwargs=kwargs, metadata=metadata
            )

            session_id = clean_metadata.pop("session_id", None)
            trace_name = cast(Optional[str], clean_metadata.pop("trace_name", None))
            trace_id = clean_metadata.pop("trace_id", litellm_call_id)
            existing_trace_id = clean_metadata.pop("existing_trace_id", None)
            update_trace_keys = cast(list, clean_metadata.pop("update_trace_keys", []))
            debug = clean_metadata.pop("debug_langfuse", None)
            mask_input = clean_metadata.pop("mask_input", False)
            mask_output = clean_metadata.pop("mask_output", False)

            clean_metadata = redact_user_api_key_info(metadata=clean_metadata)

            if trace_name is None and existing_trace_id is None:
                # just log `litellm-{call_type}` as the trace name
                ## DO NOT SET TRACE_NAME if trace-id set. this can lead to overwriting of past traces.
                trace_name = f"litellm-{kwargs.get('call_type', 'completion')}"

            if existing_trace_id is not None:
                trace_params: Dict[str, Any] = {"id": existing_trace_id}

                # Update the following keys for this trace
                for metadata_param_key in update_trace_keys:
                    trace_param_key = metadata_param_key.replace("trace_", "")
                    if trace_param_key not in trace_params:
                        updated_trace_value = clean_metadata.pop(
                            metadata_param_key, None
                        )
                        if updated_trace_value is not None:
                            trace_params[trace_param_key] = updated_trace_value

                # Pop the trace specific keys that would have been popped if there were a new trace
                for key in list(
                    filter(lambda key: key.startswith("trace_"), clean_metadata.keys())
                ):
                    clean_metadata.pop(key, None)

                # Special keys that are found in the function arguments and not the metadata
                if "input" in update_trace_keys:
                    trace_params["input"] = (
                        input if not mask_input else "redacted-by-litellm"
                    )
                if "output" in update_trace_keys:
                    trace_params["output"] = (
                        output if not mask_output else "redacted-by-litellm"
                    )
            else:  # don't overwrite an existing trace
                trace_params = {
                    "id": trace_id,
                    "name": trace_name,
                    "session_id": session_id,
                    "input": input if not mask_input else "redacted-by-litellm",
                    "version": clean_metadata.pop(
                        "trace_version", clean_metadata.get("version", None)
                    ),  # If provided just version, it will applied to the trace as well, if applied a trace version it will take precedence
                    "user_id": end_user_id,
                }
                for key in list(
                    filter(lambda key: key.startswith("trace_"), clean_metadata.keys())
                ):
                    trace_params[key.replace("trace_", "")] = clean_metadata.pop(
                        key, None
                    )

                if level == "ERROR":
                    trace_params["status_message"] = output
                else:
                    trace_params["output"] = (
                        output if not mask_output else "redacted-by-litellm"
                    )

            if debug is True or (isinstance(debug, str) and debug.lower() == "true"):
                if "metadata" in trace_params:
                    # log the raw_metadata in the trace
                    trace_params["metadata"]["metadata_passed_to_litellm"] = metadata
                else:
                    trace_params["metadata"] = {"metadata_passed_to_litellm": metadata}

            cost = kwargs.get("response_cost", None)
            verbose_logger.debug(f"trace: {cost}")

            clean_metadata["litellm_response_cost"] = cost
            if standard_logging_object is not None:
                clean_metadata["hidden_params"] = standard_logging_object[
                    "hidden_params"
                ]

            if (
                litellm.langfuse_default_tags is not None
                and isinstance(litellm.langfuse_default_tags, list)
                and "proxy_base_url" in litellm.langfuse_default_tags
            ):
                proxy_base_url = os.environ.get("PROXY_BASE_URL", None)
                if proxy_base_url is not None:
                    tags.append(f"proxy_base_url:{proxy_base_url}")

            api_base = litellm_params.get("api_base", None)
            if api_base:
                clean_metadata["api_base"] = api_base

            vertex_location = kwargs.get("vertex_location", None)
            if vertex_location:
                clean_metadata["vertex_location"] = vertex_location

            aws_region_name = kwargs.get("aws_region_name", None)
            if aws_region_name:
                clean_metadata["aws_region_name"] = aws_region_name

            if self._supports_tags():
                if "cache_hit" in kwargs:
                    if kwargs["cache_hit"] is None:
                        kwargs["cache_hit"] = False
                    clean_metadata["cache_hit"] = kwargs["cache_hit"]
                if existing_trace_id is None:
                    trace_params.update({"tags": tags})

            proxy_server_request = litellm_params.get("proxy_server_request", None)
            if proxy_server_request:
                proxy_server_request.get("method", None)
                proxy_server_request.get("url", None)
                headers = proxy_server_request.get("headers", None)
                clean_headers = {}
                if headers:
                    for key, value in headers.items():
                        # these headers can leak our API keys and/or JWT tokens
                        if key.lower() not in ["authorization", "cookie", "referer"]:
                            clean_headers[key] = value

                # clean_metadata["request"] = {
                #     "method": method,
                #     "url": url,
                #     "headers": clean_headers,
                # }
            trace = self.Langfuse.trace(**trace_params)

            # Log provider specific information as a span
            log_provider_specific_information_as_span(trace, clean_metadata)

            generation_id = None
            usage = None
            if response_obj is not None:
                if (
                    hasattr(response_obj, "id")
                    and response_obj.get("id", None) is not None
                ):
                    generation_id = litellm.utils.get_logging_id(
                        start_time, response_obj
                    )
                _usage_obj = getattr(response_obj, "usage", None)

                if _usage_obj:
                    usage = {
                        "prompt_tokens": _usage_obj.prompt_tokens,
                        "completion_tokens": _usage_obj.completion_tokens,
                        "total_cost": cost if self._supports_costs() else None,
                    }
            generation_name = clean_metadata.pop("generation_name", None)
            if generation_name is None:
                # if `generation_name` is None, use sensible default values
                # If using litellm proxy user `key_alias` if not None
                # If `key_alias` is None, just log `litellm-{call_type}` as the generation name
                _user_api_key_alias = cast(
                    Optional[str], clean_metadata.get("user_api_key_alias", None)
                )
                generation_name = (
                    f"litellm-{cast(str, kwargs.get('call_type', 'completion'))}"
                )
                if _user_api_key_alias is not None:
                    generation_name = f"litellm:{_user_api_key_alias}"

            if response_obj is not None:
                system_fingerprint = getattr(response_obj, "system_fingerprint", None)
            else:
                system_fingerprint = None

            if system_fingerprint is not None:
                optional_params["system_fingerprint"] = system_fingerprint

            generation_params = {
                "name": generation_name,
                "id": clean_metadata.pop("generation_id", generation_id),
                "start_time": start_time,
                "end_time": end_time,
                "model": kwargs["model"],
                "model_parameters": optional_params,
                "input": input if not mask_input else "redacted-by-litellm",
                "output": output if not mask_output else "redacted-by-litellm",
                "usage": usage,
                "metadata": log_requester_metadata(clean_metadata),
                "level": level,
                "version": clean_metadata.pop("version", None),
            }

            parent_observation_id = metadata.get("parent_observation_id", None)
            if parent_observation_id is not None:
                generation_params["parent_observation_id"] = parent_observation_id

            if self._supports_prompt():
                generation_params = _add_prompt_to_generation_params(
                    generation_params=generation_params,
                    clean_metadata=clean_metadata,
                    prompt_management_metadata=prompt_management_metadata,
                    langfuse_client=self.Langfuse,
                )
            if output is not None and isinstance(output, str) and level == "ERROR":
                generation_params["status_message"] = output

            if self._supports_completion_start_time():
                generation_params["completion_start_time"] = kwargs.get(
                    "completion_start_time", None
                )

            generation_client = trace.generation(**generation_params)

            return generation_client.trace_id, generation_id
        except Exception:
            verbose_logger.error(f"Langfuse Layer Error - {traceback.format_exc()}")
            return None, None

    @staticmethod
    def _get_chat_content_for_langfuse(
        response_obj: ModelResponse,
    ):
        """
        Get the chat content for Langfuse logging
        """
        if response_obj.choices and len(response_obj.choices) > 0:
            output = response_obj["choices"][0]["message"].json()
            return output
        else:
            return None

    @staticmethod
    def _get_text_completion_content_for_langfuse(
        response_obj: TextCompletionResponse,
    ):
        """
        Get the text completion content for Langfuse logging
        """
        if response_obj.choices and len(response_obj.choices) > 0:
            return response_obj.choices[0].text
        else:
            return None

    @staticmethod
    def _get_langfuse_tags(
        standard_logging_object: Optional[StandardLoggingPayload],
    ) -> List[str]:
        if standard_logging_object is None:
            return []
        return standard_logging_object.get("request_tags", []) or []

    def add_default_langfuse_tags(self, tags, kwargs, metadata):
        """
        Helper function to add litellm default langfuse tags

        - Special LiteLLM tags:
            - cache_hit
            - cache_key

        """
        if litellm.langfuse_default_tags is not None and isinstance(
            litellm.langfuse_default_tags, list
        ):
            if "cache_hit" in litellm.langfuse_default_tags:
                _cache_hit_value = kwargs.get("cache_hit", False)
                tags.append(f"cache_hit:{_cache_hit_value}")
            if "cache_key" in litellm.langfuse_default_tags:
                _hidden_params = metadata.get("hidden_params", {}) or {}
                _cache_key = _hidden_params.get("cache_key", None)
                if _cache_key is None and litellm.cache is not None:
                    # fallback to using "preset_cache_key"
                    _preset_cache_key = litellm.cache._get_preset_cache_key_from_kwargs(
                        **kwargs
                    )
                    _cache_key = _preset_cache_key
                tags.append(f"cache_key:{_cache_key}")
        return tags

    def _supports_tags(self):
        """Check if current langfuse version supports tags"""
        return Version(self.langfuse_sdk_version) >= Version("2.6.3")

    def _supports_prompt(self):
        """Check if current langfuse version supports prompt"""
        return Version(self.langfuse_sdk_version) >= Version("2.7.3")

    def _supports_costs(self):
        """Check if current langfuse version supports costs"""
        return Version(self.langfuse_sdk_version) >= Version("2.7.3")

    def _supports_completion_start_time(self):
        """Check if current langfuse version supports completion start time"""
        return Version(self.langfuse_sdk_version) >= Version("2.7.3")

    @staticmethod
    def _get_langfuse_flush_interval(flush_interval: int) -> int:
        """
        Get the langfuse flush interval to initialize the Langfuse client

        Reads `LANGFUSE_FLUSH_INTERVAL` from the environment variable.
        If not set, uses the flush interval passed in as an argument.

        Args:
            flush_interval: The flush interval to use if LANGFUSE_FLUSH_INTERVAL is not set

        Returns:
            [int] The flush interval to use to initialize the Langfuse client
        """
        return int(os.getenv("LANGFUSE_FLUSH_INTERVAL") or flush_interval)


def _add_prompt_to_generation_params(
    generation_params: dict,
    clean_metadata: dict,
    prompt_management_metadata: Optional[StandardLoggingPromptManagementMetadata],
    langfuse_client: Any,
) -> dict:
    from langfuse import Langfuse
    from langfuse.model import (
        ChatPromptClient,
        Prompt_Chat,
        Prompt_Text,
        TextPromptClient,
    )

    langfuse_client = cast(Langfuse, langfuse_client)

    user_prompt = clean_metadata.pop("prompt", None)
    if user_prompt is None and prompt_management_metadata is None:
        pass
    elif isinstance(user_prompt, dict):
        if user_prompt.get("type", "") == "chat":
            _prompt_chat = Prompt_Chat(**user_prompt)
            generation_params["prompt"] = ChatPromptClient(prompt=_prompt_chat)
        elif user_prompt.get("type", "") == "text":
            _prompt_text = Prompt_Text(**user_prompt)
            generation_params["prompt"] = TextPromptClient(prompt=_prompt_text)
        elif "version" in user_prompt and "prompt" in user_prompt:
            # prompts
            if isinstance(user_prompt["prompt"], str):
                prompt_text_params = getattr(
                    Prompt_Text, "model_fields", Prompt_Text.__fields__
                )
                _data = {
                    "name": user_prompt["name"],
                    "prompt": user_prompt["prompt"],
                    "version": user_prompt["version"],
                    "config": user_prompt.get("config", None),
                }
                if "labels" in prompt_text_params and "tags" in prompt_text_params:
                    _data["labels"] = user_prompt.get("labels", []) or []
                    _data["tags"] = user_prompt.get("tags", []) or []
                _prompt_obj = Prompt_Text(**_data)  # type: ignore
                generation_params["prompt"] = TextPromptClient(prompt=_prompt_obj)

            elif isinstance(user_prompt["prompt"], list):
                prompt_chat_params = getattr(
                    Prompt_Chat, "model_fields", Prompt_Chat.__fields__
                )
                _data = {
                    "name": user_prompt["name"],
                    "prompt": user_prompt["prompt"],
                    "version": user_prompt["version"],
                    "config": user_prompt.get("config", None),
                }
                if "labels" in prompt_chat_params and "tags" in prompt_chat_params:
                    _data["labels"] = user_prompt.get("labels", []) or []
                    _data["tags"] = user_prompt.get("tags", []) or []

                _prompt_obj = Prompt_Chat(**_data)  # type: ignore

                generation_params["prompt"] = ChatPromptClient(prompt=_prompt_obj)
            else:
                verbose_logger.error(
                    "[Non-blocking] Langfuse Logger: Invalid prompt format"
                )
        else:
            verbose_logger.error(
                "[Non-blocking] Langfuse Logger: Invalid prompt format. No prompt logged to Langfuse"
            )
    elif (
        prompt_management_metadata is not None
        and prompt_management_metadata["prompt_integration"] == "langfuse"
    ):
        try:
            generation_params["prompt"] = langfuse_client.get_prompt(
                prompt_management_metadata["prompt_id"]
            )
        except Exception as e:
            verbose_logger.debug(
                f"[Non-blocking] Langfuse Logger: Error getting prompt client for logging: {e}"
            )
            pass

    else:
        generation_params["prompt"] = user_prompt

    return generation_params


def log_provider_specific_information_as_span(
    trace,
    clean_metadata,
):
    """
    Logs provider-specific information as spans.

    Parameters:
        trace: The tracing object used to log spans.
        clean_metadata: A dictionary containing metadata to be logged.

    Returns:
        None
    """

    _hidden_params = clean_metadata.get("hidden_params", None)
    if _hidden_params is None:
        return

    vertex_ai_grounding_metadata = _hidden_params.get(
        "vertex_ai_grounding_metadata", None
    )

    if vertex_ai_grounding_metadata is not None:
        if isinstance(vertex_ai_grounding_metadata, list):
            for elem in vertex_ai_grounding_metadata:
                if isinstance(elem, dict):
                    for key, value in elem.items():
                        trace.span(
                            name=key,
                            input=value,
                        )
                else:
                    trace.span(
                        name="vertex_ai_grounding_metadata",
                        input=elem,
                    )
        else:
            trace.span(
                name="vertex_ai_grounding_metadata",
                input=vertex_ai_grounding_metadata,
            )


def log_requester_metadata(clean_metadata: dict):
    returned_metadata = {}
    requester_metadata = clean_metadata.get("requester_metadata") or {}
    for k, v in clean_metadata.items():
        if k not in requester_metadata:
            returned_metadata[k] = v

    returned_metadata.update({"requester_metadata": requester_metadata})

    return returned_metadata
