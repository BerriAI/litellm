from datetime import datetime
from typing import TYPE_CHECKING, List, Optional, Tuple

from httpx import Response

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig

from ..base_aws_llm import BaseAWSLLM
from ..common_utils import BedrockEventStreamDecoderBase, BedrockModelInfo

if TYPE_CHECKING:
    from httpx import URL


class BedrockPassthroughConfig(
    BaseAWSLLM, BedrockModelInfo, BedrockEventStreamDecoderBase, BasePassthroughConfig
):
    def is_streaming_request(self, endpoint: str, request_data: dict) -> bool:
        return "stream" in endpoint

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

        aws_region_name = self._get_aws_region_name(
            optional_params=optional_params,
            model=model,
            model_id=None,
        )

        api_base = f"https://bedrock-runtime.{aws_region_name}.amazonaws.com"

        return self.format_url(endpoint, api_base, request_query_params or {}), api_base

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

    def passthrough_cost_calculator(
        self,
        model: str,
        custom_llm_provider: str,
        httpx_response: Response,
        request_data: dict,
        logging_obj: Logging,
        endpoint: str,
        start_time: datetime,
        end_time: datetime,
        cache_hit: bool,
    ) -> float:
        """
        1. Check if request is invoke or converse
        2. If invoke, get invoke chat config
        3. If converse, get converse chat config
        4. Convert response to ModelResponse
        5. Use completion_cost to calculate cost
        6. Return the cost
        """
        from litellm import completion_cost, encoding
        from litellm.types.utils import LlmProviders, ModelResponse
        from litellm.utils import ProviderConfigManager

        if "invoke" in endpoint:
            chat_config_model = "invoke/" + model
        elif "converse" in endpoint:
            chat_config_model = "converse/" + model
        else:
            return 0.0

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

        response_cost = completion_cost(
            completion_response=litellm_model_response,
            model=model,
            custom_llm_provider=custom_llm_provider,
        )

        return response_cost

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
