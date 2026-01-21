import json
from typing import TYPE_CHECKING, List, Optional, Tuple, cast

from httpx import Response

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig

from ..base_aws_llm import BaseAWSLLM
from ..common_utils import BedrockEventStreamDecoderBase, BedrockModelInfo

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.utils import CostResponseTypes


if TYPE_CHECKING:
    from httpx import URL


class BedrockPassthroughConfig(
    BaseAWSLLM, BedrockModelInfo, BedrockEventStreamDecoderBase, BasePassthroughConfig
):
    def is_streaming_request(self, endpoint: str, request_data: dict) -> bool:
        return "stream" in endpoint

    def _encode_model_id_for_endpoint(self, model_id: str) -> str:
        """
        Encode model_id (especially ARNs) for use in Bedrock endpoints.
        
        ARNs contain special characters like colons and slashes that need to be
        properly URL-encoded when used in HTTP request paths. For example:
        arn:aws:bedrock:us-east-1:123:application-inference-profile/abc123
        becomes:
        arn:aws:bedrock:us-east-1:123:application-inference-profile%2Fabc123
        
        Args:
            model_id: The model ID or ARN to encode
            
        Returns:
            The encoded model_id suitable for use in endpoint URLs
        """
        from litellm.passthrough.utils import CommonUtils
        import re
        
        # Create a temporary endpoint with the model_id to check if encoding is needed
        temp_endpoint = f"/model/{model_id}/converse"
        encoded_temp_endpoint = CommonUtils.encode_bedrock_runtime_modelid_arn(temp_endpoint)
        
        # Extract the encoded model_id from the temporary endpoint
        encoded_model_id_match = re.search(r'/model/([^/]+)/', encoded_temp_endpoint)
        if encoded_model_id_match:
            return encoded_model_id_match.group(1)
        else:
            # Fallback to original model_id if extraction fails
            return model_id

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        endpoint: str,
        request_query_params: Optional[dict],
        litellm_params: dict,
    ) -> Tuple["URL", str]:
        optional_params = litellm_params.copy()
        model_id = optional_params.get("model_id", None)

        aws_region_name = self._get_aws_region_name(
            optional_params=optional_params,
            model=model,
            model_id=model_id,
        )

        aws_bedrock_runtime_endpoint = optional_params.get("aws_bedrock_runtime_endpoint")
        endpoint_url, _ = self.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=aws_bedrock_runtime_endpoint,
            aws_region_name=aws_region_name,
            endpoint_type="runtime",
        )

        # If model_id is provided (e.g., Application Inference Profile ARN), use it in the endpoint
        # instead of the translated model name
        if model_id is not None:
            import re
            
            # Encode the model_id if it's an ARN to properly handle special characters
            encoded_model_id = self._encode_model_id_for_endpoint(model_id)
            
            # Replace the model name in the endpoint with the encoded model_id
            endpoint = re.sub(r'model/[^/]+/', f'model/{encoded_model_id}/', endpoint)
        return self.format_url(endpoint, endpoint_url, request_query_params or {}), endpoint_url

    def sign_request(
        self,
        headers: dict,
        litellm_params: dict,
        request_data: Optional[dict],
        api_base: str,
        model: Optional[str] = None,
    ) -> Tuple[dict, Optional[bytes]]:
        optional_params = litellm_params.copy()
        return self._sign_request(
            service_name="bedrock",
            headers=headers,
            optional_params=optional_params,
            request_data=request_data or {},
            api_base=api_base,
            model=model,
        )

    def logging_non_streaming_response(
        self,
        model: str,
        custom_llm_provider: str,
        httpx_response: Response,
        request_data: dict,
        logging_obj: Logging,
        endpoint: str,
    ) -> Optional["CostResponseTypes"]:
        from litellm import encoding
        from litellm.types.utils import LlmProviders, ModelResponse
        from litellm.utils import ProviderConfigManager

        if "invoke" in endpoint:
            chat_config_model = "invoke/" + model
        elif "converse" in endpoint:
            chat_config_model = "converse/" + model
        else:
            return None

        provider_chat_config = ProviderConfigManager.get_provider_chat_config(
            provider=LlmProviders(custom_llm_provider),
            model=chat_config_model,
        )

        if provider_chat_config is None:
            raise ValueError(f"No provider config found for model: {model}")

        litellm_model_response: ModelResponse = provider_chat_config.transform_response(
            model=model,
            messages=[{"role": "user", "content": "no-message-pass-through-endpoint"}],
            raw_response=httpx_response,
            model_response=ModelResponse(),
            logging_obj=logging_obj,
            optional_params={},
            litellm_params={},
            api_key="",
            request_data=request_data,
            encoding=encoding,
        )

        return litellm_model_response

    def _convert_raw_bytes_to_str_lines(self, raw_bytes: List[bytes]) -> List[str]:
        from botocore.eventstream import EventStreamBuffer

        all_chunks = []
        event_stream_buffer = EventStreamBuffer()
        for chunk in raw_bytes:
            event_stream_buffer.add_data(chunk)
            for event in event_stream_buffer:
                message = self._parse_message_from_event(event)
                if message is not None:
                    all_chunks.append(message)

        return all_chunks

    def handle_logging_collected_chunks(
        self,
        all_chunks: List[str],
        litellm_logging_obj: "LiteLLMLoggingObj",
        model: str,
        custom_llm_provider: str,
        endpoint: str,
    ) -> Optional["CostResponseTypes"]:
        """
        1. Convert all_chunks to a ModelResponseStream
        2. combine model_response_stream to model_response
        3. Return the model_response
        """

        from litellm.litellm_core_utils.streaming_handler import (
            convert_generic_chunk_to_model_response_stream,
            generic_chunk_has_all_required_fields,
        )
        from litellm.llms.bedrock.chat import get_bedrock_event_stream_decoder
        from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
            AmazonInvokeConfig,
        )
        from litellm.main import stream_chunk_builder
        from litellm.types.utils import GenericStreamingChunk, ModelResponseStream

        all_translated_chunks = []
        if "invoke" in endpoint:
            invoke_provider = AmazonInvokeConfig.get_bedrock_invoke_provider(model)
            if invoke_provider is None:
                raise ValueError(
                    f"Invalid invoke provider: {invoke_provider}, for model: {model}"
                )
            obj = get_bedrock_event_stream_decoder(
                invoke_provider=invoke_provider,
                model=model,
                sync_stream=True,
                json_mode=False,
            )
        elif "converse" in endpoint:
            obj = get_bedrock_event_stream_decoder(
                invoke_provider=None,
                model=model,
                sync_stream=True,
                json_mode=False,
            )
        else:
            return None

        for chunk in all_chunks:
            message = json.loads(chunk)
            translated_chunk = obj._chunk_parser(chunk_data=message)

            if isinstance(
                translated_chunk, dict
            ) and generic_chunk_has_all_required_fields(cast(dict, translated_chunk)):
                chunk_obj = convert_generic_chunk_to_model_response_stream(
                    cast(GenericStreamingChunk, translated_chunk)
                )
            elif isinstance(translated_chunk, ModelResponseStream):
                chunk_obj = translated_chunk
            else:
                continue

            all_translated_chunks.append(chunk_obj)

        if len(all_translated_chunks) > 0:
            model_response = stream_chunk_builder(
                chunks=all_translated_chunks,
                logging_obj=litellm_logging_obj,
            )
            return model_response
        return None
