import base64
from typing import Any, Dict, Optional, Union, cast, get_type_hints

import litellm
from litellm._logging import verbose_logger
from litellm.llms.base_llm.responses.transformation import BaseResponsesAPIConfig
from litellm.types.llms.openai import (
    ResponseAPIUsage,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
)
from litellm.types.responses.main import DecodedResponseId
from litellm.types.utils import SpecialEnums, Usage


class ResponsesAPIRequestUtils:
    """Helper utils for constructing ResponseAPI requests"""

    @staticmethod
    def get_optional_params_responses_api(
        model: str,
        responses_api_provider_config: BaseResponsesAPIConfig,
        response_api_optional_params: ResponsesAPIOptionalRequestParams,
    ) -> Dict:
        """
        Get optional parameters for the responses API.

        Args:
            params: Dictionary of all parameters
            model: The model name
            responses_api_provider_config: The provider configuration for responses API

        Returns:
            A dictionary of supported parameters for the responses API
        """
        # Remove None values and internal parameters

        # Get supported parameters for the model
        supported_params = responses_api_provider_config.get_supported_openai_params(
            model
        )

        # Check for unsupported parameters
        unsupported_params = [
            param
            for param in response_api_optional_params
            if param not in supported_params
        ]

        if unsupported_params:
            raise litellm.UnsupportedParamsError(
                model=model,
                message=f"The following parameters are not supported for model {model}: {', '.join(unsupported_params)}",
            )

        # Map parameters to provider-specific format
        mapped_params = responses_api_provider_config.map_openai_params(
            response_api_optional_params=response_api_optional_params,
            model=model,
            drop_params=litellm.drop_params,
        )

        return mapped_params

    @staticmethod
    def get_requested_response_api_optional_param(
        params: Dict[str, Any],
    ) -> ResponsesAPIOptionalRequestParams:
        """
        Filter parameters to only include those defined in ResponsesAPIOptionalRequestParams.

        Args:
            params: Dictionary of parameters to filter

        Returns:
            ResponsesAPIOptionalRequestParams instance with only the valid parameters
        """
        valid_keys = get_type_hints(ResponsesAPIOptionalRequestParams).keys()
        filtered_params = {
            k: v for k, v in params.items() if k in valid_keys and v is not None
        }

        # decode previous_response_id if it's a litellm encoded id
        if "previous_response_id" in filtered_params:
            decoded_previous_response_id = ResponsesAPIRequestUtils.decode_previous_response_id_to_original_previous_response_id(
                filtered_params["previous_response_id"]
            )
            filtered_params["previous_response_id"] = decoded_previous_response_id

        if "metadata" in filtered_params:
            from litellm.utils import add_openai_metadata

            filtered_params["metadata"] = add_openai_metadata(
                filtered_params["metadata"]
            )

        return cast(ResponsesAPIOptionalRequestParams, filtered_params)

    @staticmethod
    def _update_responses_api_response_id_with_model_id(
        responses_api_response: ResponsesAPIResponse,
        custom_llm_provider: Optional[str],
        litellm_metadata: Optional[Dict[str, Any]] = None,
    ) -> ResponsesAPIResponse:
        """
        Update the responses_api_response_id with model_id and custom_llm_provider

        This builds a composite ID containing the custom LLM provider, model ID, and original response ID
        """
        litellm_metadata = litellm_metadata or {}
        model_info: Dict[str, Any] = litellm_metadata.get("model_info", {}) or {}
        model_id = model_info.get("id")
        updated_id = ResponsesAPIRequestUtils._build_responses_api_response_id(
            model_id=model_id,
            custom_llm_provider=custom_llm_provider,
            response_id=responses_api_response.id,
        )

        responses_api_response.id = updated_id
        return responses_api_response

    @staticmethod
    def _build_responses_api_response_id(
        custom_llm_provider: Optional[str],
        model_id: Optional[str],
        response_id: str,
    ) -> str:
        """Build the responses_api_response_id"""
        assembled_id: str = str(
            SpecialEnums.LITELLM_MANAGED_RESPONSE_COMPLETE_STR.value
        ).format(custom_llm_provider, model_id, response_id)
        base64_encoded_id: str = base64.b64encode(assembled_id.encode("utf-8")).decode(
            "utf-8"
        )
        return f"resp_{base64_encoded_id}"

    @staticmethod
    def _decode_responses_api_response_id(
        response_id: str,
    ) -> DecodedResponseId:
        """
        Decode the responses_api_response_id

        Returns:
            DecodedResponseId: Structured tuple with custom_llm_provider, model_id, and response_id
        """
        try:
            # Remove prefix and decode
            cleaned_id = response_id.replace("resp_", "")
            decoded_id = base64.b64decode(cleaned_id.encode("utf-8")).decode("utf-8")

            # Parse components using known prefixes
            if ";" not in decoded_id:
                return DecodedResponseId(
                    custom_llm_provider=None,
                    model_id=None,
                    response_id=response_id,
                )

            parts = decoded_id.split(";")

            # Format: litellm:custom_llm_provider:{};model_id:{};response_id:{}
            custom_llm_provider = None
            model_id = None

            if (
                len(parts) >= 3
            ):  # Full format with custom_llm_provider, model_id, and response_id
                custom_llm_provider_part = parts[0]
                model_id_part = parts[1]
                response_part = parts[2]

                custom_llm_provider = custom_llm_provider_part.replace(
                    "litellm:custom_llm_provider:", ""
                )
                model_id = model_id_part.replace("model_id:", "")
                decoded_response_id = response_part.replace("response_id:", "")
            else:
                decoded_response_id = response_id

            return DecodedResponseId(
                custom_llm_provider=custom_llm_provider,
                model_id=model_id,
                response_id=decoded_response_id,
            )
        except Exception as e:
            verbose_logger.debug(f"Error decoding response_id '{response_id}': {e}")
            return DecodedResponseId(
                custom_llm_provider=None,
                model_id=None,
                response_id=response_id,
            )

    @staticmethod
    def get_model_id_from_response_id(response_id: Optional[str]) -> Optional[str]:
        """Get the model_id from the response_id"""
        if response_id is None:
            return None
        decoded_response_id = (
            ResponsesAPIRequestUtils._decode_responses_api_response_id(response_id)
        )
        return decoded_response_id.get("model_id") or None

    @staticmethod
    def decode_previous_response_id_to_original_previous_response_id(
        previous_response_id: str,
    ) -> str:
        """
        Decode the previous_response_id to the original previous_response_id

        Why?
            - LiteLLM encodes the `custom_llm_provider` and `model_id` into the `previous_response_id` this helps with maintaining session consistency when load balancing multiple deployments of the same model.
            - We cannot send the litellm encoded b64 to the upstream llm api, hence we decode it to the original `previous_response_id`

        Args:
            previous_response_id: The previous_response_id to decode

        Returns:
            The original previous_response_id
        """
        decoded_response_id = (
            ResponsesAPIRequestUtils._decode_responses_api_response_id(
                previous_response_id
            )
        )
        return decoded_response_id.get("response_id", previous_response_id)


class ResponseAPILoggingUtils:
    @staticmethod
    def _is_response_api_usage(usage: Union[dict, ResponseAPIUsage]) -> bool:
        """returns True if usage is from OpenAI Response API"""
        if isinstance(usage, ResponseAPIUsage):
            return True
        if "input_tokens" in usage and "output_tokens" in usage:
            return True
        return False

    @staticmethod
    def _transform_response_api_usage_to_chat_usage(
        usage: Optional[Union[dict, ResponseAPIUsage]],
    ) -> Usage:
        """Tranforms the ResponseAPIUsage object to a Usage object"""
        if usage is None:
            return Usage(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
            )
        response_api_usage: ResponseAPIUsage = (
            ResponseAPIUsage(**usage) if isinstance(usage, dict) else usage
        )
        prompt_tokens: int = response_api_usage.input_tokens or 0
        completion_tokens: int = response_api_usage.output_tokens or 0
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
