import json
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Final, Optional, cast

import httpx
from httpx import Response

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig, PassthroughStreamCollector
from litellm.types.utils import ModelResponseStream

from ..base_aws_llm import BaseAWSLLM
from ..common_utils import BedrockError, BedrockEventStreamDecoderBase, BedrockModelInfo

if TYPE_CHECKING:
    from botocore.eventstream import EventStreamMessage
    from httpx import URL

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder
    from litellm.types.utils import CostResponseTypes


_TEXT_ONLY_DELTA_FIELDS: Final = frozenset({"content", "role"})


def _plain_text_delta(chunk: ModelResponseStream) -> str | None:
    """Return the delta text when the chunk carries nothing else that stream_chunk_builder reads."""
    if chunk.get("usage") is not None or chunk.provider_specific_fields or len(chunk.choices) != 1:
        return None
    choice: Final = chunk.choices[0]
    if choice.finish_reason or choice.logprobs is not None:
        return None
    populated: Final = frozenset(key for key, value in choice.delta.model_dump().items() if value is not None)
    if not populated <= _TEXT_ONLY_DELTA_FIELDS:
        return None
    content: Final = choice.delta.get("content")
    return content if isinstance(content, str) else None


class _CoalescedChunks:
    """Retains translated chunks with consecutive text deltas folded into one, so memory tracks the response text,
    not the event count."""

    def __init__(self) -> None:
        self._chunks: list[ModelResponseStream] = []  # mutable-ok: instance accumulator for streaming chunks
        self._open_text_parts: list[str] = []  # mutable-ok: text deltas pending fold into self._chunks[-1]

    def add(self, chunk: ModelResponseStream) -> None:
        text: Final = _plain_text_delta(chunk)
        if text is not None and self._open_text_parts:
            self._open_text_parts.append(text)
            return
        self._seal_text_run()
        self._chunks.append(chunk)
        if text is not None:
            self._open_text_parts.append(text)

    def _seal_text_run(self) -> None:
        if len(self._open_text_parts) > 1:
            self._chunks[-1].choices[0].delta.content = "".join(self._open_text_parts)
        self._open_text_parts.clear()

    def chunks(self) -> Sequence[ModelResponseStream]:
        self._seal_text_run()
        return self._chunks


def _translate_message(decoder: "AWSEventStreamDecoder", message: str) -> ModelResponseStream | None:
    from litellm.litellm_core_utils.streaming_handler import (
        convert_generic_chunk_to_model_response_stream,
        generic_chunk_has_all_required_fields,
    )
    from litellm.types.utils import GenericStreamingChunk

    translated_chunk: Final = decoder._chunk_parser(chunk_data=json.loads(message))
    if isinstance(translated_chunk, ModelResponseStream):
        return translated_chunk
    if generic_chunk_has_all_required_fields(cast(dict, translated_chunk)):
        return convert_generic_chunk_to_model_response_stream(cast(GenericStreamingChunk, translated_chunk))
    return None


def _build_logged_response(
    chunks: Sequence[ModelResponseStream], litellm_logging_obj: "LiteLLMLoggingObj"
) -> Optional["CostResponseTypes"]:
    from litellm.main import stream_chunk_builder

    if len(chunks) == 0:
        return None
    return stream_chunk_builder(chunks=list(chunks), logging_obj=litellm_logging_obj)


class BedrockEventStreamCollector:
    """Decodes and translates Bedrock event-stream frames as they are relayed instead of buffering the stream."""

    def __init__(
        self,
        parse_event: Callable[["EventStreamMessage"], str | None],
        decoder: Optional["AWSEventStreamDecoder"],
    ) -> None:
        from botocore.eventstream import EventStreamBuffer

        self._parse_event = parse_event
        self._decoder = decoder
        self._event_stream_buffer: Final[EventStreamBuffer] = EventStreamBuffer()
        self._chunks: Final = _CoalescedChunks()

    def add(self, chunk: bytes) -> None:
        if self._decoder is None:
            return
        self._event_stream_buffer.add_data(chunk)
        for event in self._event_stream_buffer:
            self._add_event(self._decoder, event)

    def _add_event(self, decoder: "AWSEventStreamDecoder", event: "EventStreamMessage") -> None:
        message: Final = self._parse_event(event)
        translated: Final = _translate_message(decoder, message) if message is not None else None
        if translated is not None:
            self._chunks.add(translated)

    def build_logged_response(self, litellm_logging_obj: "LiteLLMLoggingObj") -> Optional["CostResponseTypes"]:
        return _build_logged_response(self._chunks.chunks(), litellm_logging_obj)


class BedrockPassthroughConfig(BaseAWSLLM, BedrockModelInfo, BedrockEventStreamDecoderBase, BasePassthroughConfig):
    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object] | httpx.Headers,  # mutable-ok: base passes response headers as a dict
    ) -> BedrockError:
        return BedrockError(status_code=status_code, message=error_message, headers=headers)

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
        import re

        from litellm.passthrough.utils import CommonUtils

        # Create a temporary endpoint with the model_id to check if encoding is needed
        temp_endpoint: Final = f"/model/{model_id}/converse"
        encoded_temp_endpoint: Final = CommonUtils.encode_bedrock_runtime_modelid_arn(temp_endpoint)

        # Extract the encoded model_id from the temporary endpoint
        encoded_model_id_match: Final = re.search(r"/model/([^/]+)/", encoded_temp_endpoint)
        if encoded_model_id_match:
            return encoded_model_id_match.group(1)
        else:
            # Fallback to original model_id if extraction fails
            return model_id

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        endpoint: str,
        request_query_params: dict | None,
        litellm_params: dict,
    ) -> tuple["URL", str]:
        optional_params: Final = litellm_params.copy()
        model_id: Final = optional_params.get("model_id", None)

        aws_region_name: Final = self._get_aws_region_name(
            optional_params=optional_params,
            model=model,
            model_id=model_id,
        )

        aws_bedrock_runtime_endpoint: Final = optional_params.get("aws_bedrock_runtime_endpoint")
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
            encoded_model_id: Final = self._encode_model_id_for_endpoint(model_id)

            # Replace the model name in the endpoint with the encoded model_id
            endpoint = re.sub(r"model/[^/]+/", f"model/{encoded_model_id}/", endpoint)
        return (
            self.format_url(endpoint, endpoint_url, request_query_params or {}),
            endpoint_url,
        )

    def get_bedrock_bearer_token(self, litellm_params: Mapping[str, object]) -> str | None:
        return None

    def sign_request(
        self,
        headers: dict,
        litellm_params: dict,
        request_data: dict | None,
        api_base: str,
        model: str | None = None,
    ) -> tuple[dict, bytes | None]:
        optional_params: Final = litellm_params.copy()
        return self._sign_request(
            service_name="bedrock",
            headers=headers,
            optional_params=optional_params,
            request_data=request_data or {},
            api_base=api_base,
            model=model,
            api_key=self.get_bedrock_bearer_token(optional_params),
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

        provider_chat_config: Final = ProviderConfigManager.get_provider_chat_config(
            provider=LlmProviders(custom_llm_provider),
            model=chat_config_model,
        )

        if provider_chat_config is None:
            raise ValueError(f"No provider config found for model: {model}")

        litellm_model_response: Final[ModelResponse] = provider_chat_config.transform_response(
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

    def create_stream_collector(
        self, model: str, custom_llm_provider: str, endpoint: str
    ) -> PassthroughStreamCollector:
        return BedrockEventStreamCollector(
            parse_event=self._parse_message_from_event,
            decoder=self._get_event_stream_decoder(model=model, endpoint=endpoint),
        )

    def _get_event_stream_decoder(self, model: str, endpoint: str) -> Optional["AWSEventStreamDecoder"]:
        from litellm.llms.bedrock.chat import get_bedrock_event_stream_decoder
        from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
            AmazonInvokeConfig,
        )

        if "invoke" in endpoint:
            invoke_provider: Final = AmazonInvokeConfig.get_bedrock_invoke_provider(model)
            if invoke_provider is None:
                verbose_logger.warning(
                    "Bedrock passthrough spend tracking skipped: no invoke provider for model %s", model
                )
                return None
            return get_bedrock_event_stream_decoder(
                invoke_provider=invoke_provider, model=model, sync_stream=True, json_mode=False
            )
        if "converse" in endpoint:
            return get_bedrock_event_stream_decoder(
                invoke_provider=None, model=model, sync_stream=True, json_mode=False
            )
        return None
