from typing import TYPE_CHECKING, Any, Dict, Optional, Union, cast

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import *
from litellm.types.responses.main import *
from litellm.types.router import GenericLiteLLMParams

from ..common_utils import OpenAIError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class OpenAIResponsesAPIConfig(BaseResponsesAPIConfig):
    def get_supported_openai_params(self, model: str) -> list:
        """
        All OpenAI Responses API params are supported
        """
        return [
            "input",
            "model",
            "include",
            "instructions",
            "max_output_tokens",
            "metadata",
            "parallel_tool_calls",
            "previous_response_id",
            "reasoning",
            "store",
            "background",
            "stream",
            "prompt",
            "temperature",
            "text",
            "tool_choice",
            "tools",
            "top_p",
            "truncation",
            "user",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        ]

    def map_openai_params(
        self,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        """No mapping applied since inputs are in OpenAI spec already"""
        return dict(response_api_optional_params)

    def transform_responses_api_request(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        response_api_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        """No transform applied since inputs are in OpenAI spec already"""
        return dict(
            ResponsesAPIRequestParams(
                model=model, input=input, **response_api_optional_request_params
            )
        )

    def transform_response_api_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIResponse:
        """No transform applied since outputs are in OpenAI spec already"""
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise OpenAIError(
                message=raw_response.text, status_code=raw_response.status_code
            )
        return ResponsesAPIResponse(**raw_response_json)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        api_key = (
            api_key
            or litellm.api_key
            or litellm.openai_key
            or get_secret_str("OPENAI_API_KEY")
        )
        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
            }
        )
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the endpoint for OpenAI responses API
        """
        api_base = (
            api_base
            or litellm.api_base
            or get_secret_str("OPENAI_BASE_URL")
            or get_secret_str("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        )

        # Remove trailing slashes
        api_base = api_base.rstrip("/")

        return f"{api_base}/responses"

    def transform_streaming_response(
        self,
        model: str,
        parsed_chunk: dict,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIStreamingResponse:
        """
        Transform a parsed streaming response chunk into a ResponsesAPIStreamingResponse
        """
        # Convert the dictionary to a properly typed ResponsesAPIStreamingResponse
        verbose_logger.debug("Raw OpenAI Chunk=%s", parsed_chunk)
        event_type = str(parsed_chunk.get("type"))
        event_pydantic_model = OpenAIResponsesAPIConfig.get_event_model_class(
            event_type=event_type
        )
        return event_pydantic_model(**parsed_chunk)

    @staticmethod
    def get_event_model_class(event_type: str) -> Any:
        """
        Returns the appropriate event model class based on the event type.

        Args:
            event_type (str): The type of event from the response chunk

        Returns:
            Any: The corresponding event model class

        Raises:
            ValueError: If the event type is unknown
        """
        event_models = {
            ResponsesAPIStreamEvents.RESPONSE_CREATED: ResponseCreatedEvent,
            ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS: ResponseInProgressEvent,
            ResponsesAPIStreamEvents.RESPONSE_COMPLETED: ResponseCompletedEvent,
            ResponsesAPIStreamEvents.RESPONSE_FAILED: ResponseFailedEvent,
            ResponsesAPIStreamEvents.RESPONSE_INCOMPLETE: ResponseIncompleteEvent,
            ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED: OutputItemAddedEvent,
            ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE: OutputItemDoneEvent,
            ResponsesAPIStreamEvents.CONTENT_PART_ADDED: ContentPartAddedEvent,
            ResponsesAPIStreamEvents.CONTENT_PART_DONE: ContentPartDoneEvent,
            ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA: OutputTextDeltaEvent,
            ResponsesAPIStreamEvents.OUTPUT_TEXT_ANNOTATION_ADDED: OutputTextAnnotationAddedEvent,
            ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE: OutputTextDoneEvent,
            ResponsesAPIStreamEvents.REFUSAL_DELTA: RefusalDeltaEvent,
            ResponsesAPIStreamEvents.REFUSAL_DONE: RefusalDoneEvent,
            ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DELTA: FunctionCallArgumentsDeltaEvent,
            ResponsesAPIStreamEvents.FUNCTION_CALL_ARGUMENTS_DONE: FunctionCallArgumentsDoneEvent,
            ResponsesAPIStreamEvents.FILE_SEARCH_CALL_IN_PROGRESS: FileSearchCallInProgressEvent,
            ResponsesAPIStreamEvents.FILE_SEARCH_CALL_SEARCHING: FileSearchCallSearchingEvent,
            ResponsesAPIStreamEvents.FILE_SEARCH_CALL_COMPLETED: FileSearchCallCompletedEvent,
            ResponsesAPIStreamEvents.WEB_SEARCH_CALL_IN_PROGRESS: WebSearchCallInProgressEvent,
            ResponsesAPIStreamEvents.WEB_SEARCH_CALL_SEARCHING: WebSearchCallSearchingEvent,
            ResponsesAPIStreamEvents.WEB_SEARCH_CALL_COMPLETED: WebSearchCallCompletedEvent,
            ResponsesAPIStreamEvents.ERROR: ErrorEvent,
        }

        model_class = event_models.get(cast(ResponsesAPIStreamEvents, event_type))
        if not model_class:
            return GenericEvent

        return model_class

    def should_fake_stream(
        self,
        model: Optional[str],
        stream: Optional[bool],
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        if stream is not True:
            return False
        if model is not None:
            try:
                if (
                    litellm.utils.supports_native_streaming(
                        model=model,
                        custom_llm_provider=custom_llm_provider,
                    )
                    is False
                ):
                    return True
            except Exception as e:
                verbose_logger.debug(
                    f"Error getting model info in OpenAIResponsesAPIConfig: {e}"
                )
        return False

    #########################################################
    ########## DELETE RESPONSE API TRANSFORMATION ##############
    #########################################################
    def transform_delete_response_api_request(
        self,
        response_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform the delete response API request into a URL and data

        OpenAI API expects the following request
        - DELETE /v1/responses/{response_id}
        """
        url = f"{api_base}/{response_id}"
        data: Dict = {}
        return url, data

    def transform_delete_response_api_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> DeleteResponseResult:
        """
        Transform the delete response API response into a DeleteResponseResult
        """
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise OpenAIError(
                message=raw_response.text, status_code=raw_response.status_code
            )
        return DeleteResponseResult(**raw_response_json)

    #########################################################
    ########## GET RESPONSE API TRANSFORMATION ###############
    #########################################################
    def transform_get_response_api_request(
        self,
        response_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform the get response API request into a URL and data

        OpenAI API expects the following request
        - GET /v1/responses/{response_id}
        """
        url = f"{api_base}/{response_id}"
        data: Dict = {}
        return url, data

    def transform_get_response_api_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIResponse:
        """
        Transform the get response API response into a ResponsesAPIResponse
        """
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise OpenAIError(
                message=raw_response.text, status_code=raw_response.status_code
            )
        return ResponsesAPIResponse(**raw_response_json)

    #########################################################
    ########## LIST INPUT ITEMS TRANSFORMATION #############
    #########################################################
    def transform_list_input_items_request(
        self,
        response_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: Optional[str] = None,
        before: Optional[str] = None,
        include: Optional[List[str]] = None,
        limit: int = 20,
        order: Literal["asc", "desc"] = "desc",
    ) -> Tuple[str, Dict]:
        url = f"{api_base}/{response_id}/input_items"
        params: Dict[str, Any] = {}
        if after is not None:
            params["after"] = after
        if before is not None:
            params["before"] = before
        if include:
            params["include"] = ",".join(include)
        if limit is not None:
            params["limit"] = limit
        if order is not None:
            params["order"] = order
        return url, params

    def transform_list_input_items_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Dict:
        try:
            return raw_response.json()
        except Exception:
            raise OpenAIError(
                message=raw_response.text, status_code=raw_response.status_code
            )
