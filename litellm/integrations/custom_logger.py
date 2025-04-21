#### What this does ####
#    On success, logs events to Promptlayer
import traceback
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

from pydantic import BaseModel

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.integrations.argilla import ArgillaItem
from litellm.types.llms.openai import AllMessageValues, ChatCompletionRequest
from litellm.types.utils import (
    AdapterCompletionStreamWrapper,
    LLMResponseTypes,
    ModelResponse,
    ModelResponseStream,
    StandardCallbackDynamicParams,
    StandardLoggingPayload,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = Union[_Span, Any]
else:
    Span = Any


class CustomLogger:  # https://docs.litellm.ai/docs/observability/custom_callback#callback-class
    # Class variables or attributes
    def __init__(self, message_logging: bool = True) -> None:
        self.message_logging = message_logging
        pass

    def log_pre_api_call(self, model, messages, kwargs):
        pass

    def log_post_api_call(self, kwargs, response_obj, start_time, end_time):
        pass

    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        pass

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        pass

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        pass

    #### ASYNC ####

    async def async_log_stream_event(self, kwargs, response_obj, start_time, end_time):
        pass

    async def async_log_pre_api_call(self, model, messages, kwargs):
        pass

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        pass

    #### PROMPT MANAGEMENT HOOKS ####

    async def async_get_chat_completion_prompt(
        self,
        model: str,
        messages: List[AllMessageValues],
        non_default_params: dict,
        prompt_id: str,
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
    ) -> Tuple[str, List[AllMessageValues], dict]:
        """
        Returns:
        - model: str - the model to use (can be pulled from prompt management tool)
        - messages: List[AllMessageValues] - the messages to use (can be pulled from prompt management tool)
        - non_default_params: dict - update with any optional params (e.g. temperature, max_tokens, etc.) to use (can be pulled from prompt management tool)
        """
        return model, messages, non_default_params

    def get_chat_completion_prompt(
        self,
        model: str,
        messages: List[AllMessageValues],
        non_default_params: dict,
        prompt_id: Optional[str],
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
    ) -> Tuple[str, List[AllMessageValues], dict]:
        """
        Returns:
        - model: str - the model to use (can be pulled from prompt management tool)
        - messages: List[AllMessageValues] - the messages to use (can be pulled from prompt management tool)
        - non_default_params: dict - update with any optional params (e.g. temperature, max_tokens, etc.) to use (can be pulled from prompt management tool)
        """
        return model, messages, non_default_params

    #### PRE-CALL CHECKS - router/proxy only ####
    """
    Allows usage-based-routing-v2 to run pre-call rpm checks within the picked deployment's semaphore (concurrency-safe tpm/rpm checks).
    """

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: List,
        messages: Optional[List[AllMessageValues]],
        request_kwargs: Optional[dict] = None,
        parent_otel_span: Optional[Span] = None,
    ) -> List[dict]:
        return healthy_deployments

    async def async_pre_call_check(
        self, deployment: dict, parent_otel_span: Optional[Span]
    ) -> Optional[dict]:
        pass

    def pre_call_check(self, deployment: dict) -> Optional[dict]:
        pass

    #### Fallback Events - router/proxy only ####
    async def log_model_group_rate_limit_error(
        self, exception: Exception, original_model_group: Optional[str], kwargs: dict
    ):
        pass

    async def log_success_fallback_event(
        self, original_model_group: str, kwargs: dict, original_exception: Exception
    ):
        pass

    async def log_failure_fallback_event(
        self, original_model_group: str, kwargs: dict, original_exception: Exception
    ):
        pass

    #### ADAPTERS #### Allow calling 100+ LLMs in custom format - https://github.com/BerriAI/litellm/pulls

    def translate_completion_input_params(
        self, kwargs
    ) -> Optional[ChatCompletionRequest]:
        """
        Translates the input params, from the provider's native format to the litellm.completion() format.
        """
        pass

    def translate_completion_output_params(
        self, response: ModelResponse
    ) -> Optional[BaseModel]:
        """
        Translates the output params, from the OpenAI format to the custom format.
        """
        pass

    def translate_completion_output_params_streaming(
        self, completion_stream: Any
    ) -> Optional[AdapterCompletionStreamWrapper]:
        """
        Translates the streaming chunk, from the OpenAI format to the custom format.
        """
        pass

    ### DATASET HOOKS #### - currently only used for Argilla

    async def async_dataset_hook(
        self,
        logged_item: ArgillaItem,
        standard_logging_payload: Optional[StandardLoggingPayload],
    ) -> Optional[ArgillaItem]:
        """
        - Decide if the result should be logged to Argilla.
        - Modify the result before logging to Argilla.
        - Return None if the result should not be logged to Argilla.
        """
        raise NotImplementedError("async_dataset_hook not implemented")

    #### CALL HOOKS - proxy only ####
    """
    Control the modify incoming / outgoung data before calling the model
    """

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
        ],
    ) -> Optional[
        Union[Exception, str, dict]
    ]:  # raise exception if invalid, return a str for the user to receive - if rejected, or return a modified dictionary for passing into litellm
        pass

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
    ):
        pass

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: LLMResponseTypes,
    ) -> Any:
        pass

    async def async_logging_hook(
        self, kwargs: dict, result: Any, call_type: str
    ) -> Tuple[dict, Any]:
        """For masking logged request/response. Return a modified version of the request/result."""
        return kwargs, result

    def logging_hook(
        self, kwargs: dict, result: Any, call_type: str
    ) -> Tuple[dict, Any]:
        """For masking logged request/response. Return a modified version of the request/result."""
        return kwargs, result

    async def async_moderation_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        call_type: Literal[
            "completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "responses",
        ],
    ) -> Any:
        pass

    async def async_post_call_streaming_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: str,
    ) -> Any:
        pass

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        async for item in response:
            yield item

    #### SINGLE-USE #### - https://docs.litellm.ai/docs/observability/custom_callback#using-your-custom-callback-function

    def log_input_event(self, model, messages, kwargs, print_verbose, callback_func):
        try:
            kwargs["model"] = model
            kwargs["messages"] = messages
            kwargs["log_event_type"] = "pre_api_call"
            callback_func(
                kwargs,
            )
            print_verbose(f"Custom Logger - model call details: {kwargs}")
        except Exception:
            print_verbose(f"Custom Logger Error - {traceback.format_exc()}")

    async def async_log_input_event(
        self, model, messages, kwargs, print_verbose, callback_func
    ):
        try:
            kwargs["model"] = model
            kwargs["messages"] = messages
            kwargs["log_event_type"] = "pre_api_call"
            await callback_func(
                kwargs,
            )
            print_verbose(f"Custom Logger - model call details: {kwargs}")
        except Exception:
            print_verbose(f"Custom Logger Error - {traceback.format_exc()}")

    def log_event(
        self, kwargs, response_obj, start_time, end_time, print_verbose, callback_func
    ):
        # Method definition
        try:
            kwargs["log_event_type"] = "post_api_call"
            callback_func(
                kwargs,  # kwargs to func
                response_obj,
                start_time,
                end_time,
            )
        except Exception:
            print_verbose(f"Custom Logger Error - {traceback.format_exc()}")
            pass

    async def async_log_event(
        self, kwargs, response_obj, start_time, end_time, print_verbose, callback_func
    ):
        # Method definition
        try:
            kwargs["log_event_type"] = "post_api_call"
            await callback_func(
                kwargs,  # kwargs to func
                response_obj,
                start_time,
                end_time,
            )
        except Exception:
            print_verbose(f"Custom Logger Error - {traceback.format_exc()}")
            pass

    # Useful helpers for custom logger classes

    def truncate_standard_logging_payload_content(
        self,
        standard_logging_object: StandardLoggingPayload,
    ):
        """
        Truncate error strings and message content in logging payload

        Some loggers like DataDog/ GCS Bucket have a limit on the size of the payload. (1MB)

        This function truncates the error string and the message content if they exceed a certain length.
        """
        MAX_STR_LENGTH = 10_000

        # Truncate fields that might exceed max length
        fields_to_truncate = ["error_str", "messages", "response"]
        for field in fields_to_truncate:
            self._truncate_field(
                standard_logging_object=standard_logging_object,
                field_name=field,
                max_length=MAX_STR_LENGTH,
            )

    def _truncate_field(
        self,
        standard_logging_object: StandardLoggingPayload,
        field_name: str,
        max_length: int,
    ) -> None:
        """
        Helper function to truncate a field in the logging payload

        This converts the field to a string and then truncates it if it exceeds the max length.

        Why convert to string ?
        1. User was sending a poorly formatted list for `messages` field, we could not predict where they would send content
            - Converting to string and then truncating the logged content catches this
        2. We want to avoid modifying the original `messages`, `response`, and `error_str` in the logging payload since these are in kwargs and could be returned to the user
        """
        field_value = standard_logging_object.get(field_name)  # type: ignore
        if field_value:
            str_value = str(field_value)
            if len(str_value) > max_length:
                standard_logging_object[field_name] = self._truncate_text(  # type: ignore
                    text=str_value, max_length=max_length
                )

    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text if it exceeds max_length"""
        return (
            text[:max_length]
            + "...truncated by litellm, this logger does not support large content"
            if len(text) > max_length
            else text
        )
