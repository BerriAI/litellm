from copy import deepcopy
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import httpx
from openai.types.responses import ResponseReasoningItem

from litellm._logging import verbose_logger
from litellm.llms.azure.common_utils import BaseAzureLLM
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.llms.openai import *
from litellm.types.responses.main import *
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class AzureOpenAIResponsesAPIConfig(OpenAIResponsesAPIConfig):

    # Parameters not supported by Azure Responses API
    AZURE_UNSUPPORTED_PARAMS = ["context_management"]

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.AZURE

    def get_supported_openai_params(self, model: str) -> list:
        """
        Azure Responses API does not support context_management (compaction).
        """
        base_supported_params = super().get_supported_openai_params(model)
        return [
            param
            for param in base_supported_params
            if param not in self.AZURE_UNSUPPORTED_PARAMS
        ]

    def validate_environment(
        self, headers: dict, model: str, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        return BaseAzureLLM._base_validate_azure_environment(
            headers=headers, litellm_params=litellm_params
        )

    def get_stripped_model_name(self, model: str) -> str:
        # if "responses/" is in the model name, remove it
        if "responses/" in model:
            model = model.replace("responses/", "")
        if "o_series" in model:
            model = model.replace("o_series/", "")
        return model

    def _handle_reasoning_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle reasoning items to filter out the status field.
        Issue: https://github.com/BerriAI/litellm/issues/13484

        Azure OpenAI API does not accept 'status' field in reasoning input items.
        """
        if item.get("type") == "reasoning":
            try:
                # Ensure required fields are present for ResponseReasoningItem
                item_data = dict(item)
                if "summary" not in item_data:
                    item_data["summary"] = (
                        item_data.get("reasoning_content", "")[:100] + "..."
                        if len(item_data.get("reasoning_content", "")) > 100
                        else item_data.get("reasoning_content", "")
                    )

                # Create ResponseReasoningItem object from the item data
                reasoning_item = ResponseReasoningItem(**item_data)

                # Convert back to dict with exclude_none=True to exclude None fields
                dict_reasoning_item = reasoning_item.model_dump(exclude_none=True)
                dict_reasoning_item.pop("status", None)

                return dict_reasoning_item
            except Exception as e:
                verbose_logger.debug(
                    f"Failed to create ResponseReasoningItem, falling back to manual filtering: {e}"
                )
                # Fallback: manually filter out known None fields
                filtered_item = {
                    k: v
                    for k, v in item.items()
                    if v is not None
                    or k not in {"status", "content", "encrypted_content"}
                }
                return filtered_item
        return item

    def _validate_input_param(
        self, input: Union[str, ResponseInputParam]
    ) -> Union[str, ResponseInputParam]:
        """
        Override parent method to also filter out 'status' field from message items.
        Azure OpenAI API does not accept 'status' field in input messages.
        """
        from typing import cast

        # First call parent's validation
        validated_input = super()._validate_input_param(input)

        # Then filter out status from message items
        if isinstance(validated_input, list):
            filtered_input: List[Any] = []
            for item in validated_input:
                if isinstance(item, dict) and item.get("type") == "message":
                    # Filter out status field from message items
                    filtered_item = {k: v for k, v in item.items() if k != "status"}
                    filtered_input.append(filtered_item)
                else:
                    filtered_input.append(item)
            return cast(ResponseInputParam, filtered_input)

        return validated_input

    def transform_responses_api_request(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        response_api_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        """No transform applied since inputs are in OpenAI spec already"""
        stripped_model_name = self.get_stripped_model_name(model)

        # Azure Responses API requires flattened tools (params at top level, not nested in 'function')
        if "tools" in response_api_optional_request_params and isinstance(
            response_api_optional_request_params["tools"], list
        ):
            new_tools: List[Dict[str, Any]] = []
            for tool in response_api_optional_request_params["tools"]:
                if isinstance(tool, dict) and "function" in tool:
                    new_tool: Dict[str, Any] = deepcopy(tool)
                    function_data = new_tool.pop("function")
                    new_tool.update(function_data)
                    new_tools.append(new_tool)
                else:
                    new_tools.append(tool)
            response_api_optional_request_params["tools"] = new_tools

        return super().transform_responses_api_request(
            model=stripped_model_name,
            input=input,
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Constructs a complete URL for the API request.

        Args:
        - api_base: Base URL, e.g.,
            "https://litellm8397336933.openai.azure.com"
            OR
            "https://litellm8397336933.openai.azure.com/openai/responses?api-version=2024-05-01-preview"
        - model: Model name.
        - optional_params: Additional query parameters, including "api_version".
        - stream: If streaming is required (optional).

        Returns:
        - A complete URL string, e.g.,
        "https://litellm8397336933.openai.azure.com/openai/responses?api-version=2024-05-01-preview"
        """
        from litellm.constants import AZURE_DEFAULT_RESPONSES_API_VERSION

        return BaseAzureLLM._get_base_azure_url(
            api_base=api_base,
            litellm_params=litellm_params,
            route="/openai/responses",
            default_api_version=AZURE_DEFAULT_RESPONSES_API_VERSION,
        )

    #########################################################
    ########## DELETE RESPONSE API TRANSFORMATION ##############
    #########################################################
    def _construct_url_for_response_id_in_path(
        self, api_base: str, response_id: str
    ) -> str:
        """
        Constructs a URL for the API request with the response_id in the path.
        """
        from urllib.parse import urlparse, urlunparse

        # Parse the URL to separate its components
        parsed_url = urlparse(api_base)

        # Insert the response_id at the end of the path component
        # Remove trailing slash if present to avoid double slashes
        path = parsed_url.path.rstrip("/")
        new_path = f"{path}/{response_id}"

        # Reconstruct the URL with all original components but with the modified path
        constructed_url = urlunparse(
            (
                parsed_url.scheme,  # http, https
                parsed_url.netloc,  # domain name, port
                new_path,  # path with response_id added
                parsed_url.params,  # parameters
                parsed_url.query,  # query string
                parsed_url.fragment,  # fragment
            )
        )
        return constructed_url

    def transform_delete_response_api_request(
        self,
        response_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform the delete response API request into a URL and data

        Azure OpenAI API expects the following request:
        - DELETE /openai/responses/{response_id}?api-version=xxx

        This function handles URLs with query parameters by inserting the response_id
        at the correct location (before any query parameters).
        """
        delete_url = self._construct_url_for_response_id_in_path(
            api_base=api_base, response_id=response_id
        )

        data: Dict = {}
        verbose_logger.debug(f"delete response url={delete_url}")
        return delete_url, data

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
        get_url = self._construct_url_for_response_id_in_path(
            api_base=api_base, response_id=response_id
        )
        data: Dict = {}
        verbose_logger.debug(f"get response url={get_url}")
        return get_url, data

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
        url = (
            self._construct_url_for_response_id_in_path(
                api_base=api_base, response_id=response_id
            )
            + "/input_items"
        )
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
        verbose_logger.debug(f"list input items url={url}")
        return url, params

    #########################################################
    ########## CANCEL RESPONSE API TRANSFORMATION ##########
    #########################################################
    def transform_cancel_response_api_request(
        self,
        response_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform the cancel response API request into a URL and data

        Azure OpenAI API expects the following request:
        - POST /openai/responses/{response_id}/cancel?api-version=xxx

        This function handles URLs with query parameters by inserting the response_id
        at the correct location (before any query parameters).
        """
        from urllib.parse import urlparse, urlunparse

        # Parse the URL to separate its components
        parsed_url = urlparse(api_base)

        # Insert the response_id and /cancel at the end of the path component
        # Remove trailing slash if present to avoid double slashes
        path = parsed_url.path.rstrip("/")
        new_path = f"{path}/{response_id}/cancel"

        # Reconstruct the URL with all original components but with the modified path
        cancel_url = urlunparse(
            (
                parsed_url.scheme,  # http, https
                parsed_url.netloc,  # domain name, port
                new_path,  # path with response_id and /cancel added
                parsed_url.params,  # parameters
                parsed_url.query,  # query string
                parsed_url.fragment,  # fragment
            )
        )

        data: Dict = {}
        verbose_logger.debug(f"cancel response url={cancel_url}")
        return cancel_url, data

    def transform_cancel_response_api_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ResponsesAPIResponse:
        """
        Transform the cancel response API response into a ResponsesAPIResponse
        """
        try:
            raw_response_json = raw_response.json()
        except Exception:
            from litellm.llms.azure.chat.gpt_transformation import AzureOpenAIError

            raise AzureOpenAIError(
                message=raw_response.text, status_code=raw_response.status_code
            )
        return ResponsesAPIResponse(**raw_response_json)
