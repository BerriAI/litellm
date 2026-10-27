import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from functools import partial
from itertools import chain
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Optional, cast

import httpx
from httpx import Response
from pydantic import TypeAdapter

from litellm._logging import verbose_logger
from litellm.constants import AWS_ERROR_TYPE_HEADER
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.passthrough.transformation import BasePassthroughConfig, PassthroughStreamCollector
from litellm.llms.bedrock.chat.converse_to_openai import as_json_array, as_json_object
from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    bedrock_invoke_guardrail_headers,
)
from litellm.types.llms.openai import ChatCompletionSystemMessage, ChatCompletionUserMessage
from litellm.types.utils import ModelResponseStream

from ..base_aws_llm import BaseAWSLLM
from ..common_utils import BedrockError, BedrockEventStreamDecoderBase, BedrockModelInfo
from ..request_metadata import (
    BEDROCK_REQUEST_METADATA_HEADER,
    BedrockRequestMetadataViolation,
    bedrock_request_metadata_headers,
    bedrock_request_metadata_is_owned,
    bedrock_request_metadata_violation,
    caller_request_metadata_violations_are_dropped,
    merge_bedrock_invoke_headers,
    resolve_bedrock_request_metadata,
)

if TYPE_CHECKING:
    from botocore.eventstream import EventStreamMessage
    from httpx import URL

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder
    from litellm.types.utils import CostResponseTypes


BEDROCK_STREAMING_ACTIONS: Final = frozenset({"invoke-with-response-stream", "converse-stream"})
_BEDROCK_CONVERSE_ACTIONS: Final = frozenset({"converse", "converse-stream"})
_BEDROCK_INVOKE_ACTIONS: Final = frozenset({"invoke", "invoke-with-response-stream"})
_BEDROCK_INVOKE_GUARDRAIL_HEADERS: Final = frozenset(
    {"x-amzn-bedrock-guardrailidentifier", "x-amzn-bedrock-guardrailversion", "x-amzn-bedrock-trace"}
)


def _bedrock_endpoint_action(endpoint: str) -> str:
    return endpoint.partition("?")[0].rstrip("/").rsplit("/", 1)[-1]


def is_bedrock_streaming_endpoint(endpoint: str) -> bool:
    return _bedrock_endpoint_action(endpoint) in BEDROCK_STREAMING_ACTIONS


def _caller_request_metadata_violation(
    action: str, body: Mapping[str, object]
) -> BedrockRequestMetadataViolation | None:
    if action in _BEDROCK_INVOKE_ACTIONS and "requestMetadata" in body and body["requestMetadata"] is None:
        return BedrockRequestMetadataViolation("type", "requestMetadata must be a dictionary")
    return bedrock_request_metadata_violation(body.get("requestMetadata"))


def _aws_error_type(action: str, violation: BedrockRequestMetadataViolation) -> str:
    if action in _BEDROCK_CONVERSE_ACTIONS and violation.kind == "type":
        return "SerializationException"
    return "ValidationException"


def _caller_request_metadata(action: str, body: Mapping[str, object], litellm_params: Mapping[str, object]) -> object:
    violation: Final = (
        None
        if caller_request_metadata_violations_are_dropped(litellm_params.get("drop_params") is True)
        else _caller_request_metadata_violation(action, body)
    )
    if violation is not None:
        raise BedrockError(
            status_code=400,
            message=violation.message,
            headers={AWS_ERROR_TYPE_HEADER: _aws_error_type(action, violation)},
        )
    return body.get("requestMetadata")


def _with_body_request_metadata(
    action: str, body: Mapping[str, object], litellm_params: Mapping[str, object]
) -> Mapping[str, object]:
    resolved: Final = resolve_bedrock_request_metadata(
        litellm_params, caller_metadata=_caller_request_metadata(action, body, litellm_params)
    )
    without_caller: Final = {key: value for key, value in body.items() if key != "requestMetadata"}
    return without_caller if resolved is None else {**without_caller, "requestMetadata": resolved}


def _without_owned_request_metadata_header(headers: Mapping[str, str]) -> Mapping[str, str]:
    if not bedrock_request_metadata_is_owned():
        return headers
    return {name: value for name, value in headers.items() if name.lower() != BEDROCK_REQUEST_METADATA_HEADER.lower()}


def _with_request_metadata(
    action: str | None,
    headers: Mapping[str, str],
    body: Mapping[str, object],
    litellm_params: Mapping[str, object],
) -> tuple[Mapping[str, str], Mapping[str, object]]:
    if action in _BEDROCK_CONVERSE_ACTIONS:
        return headers, _with_body_request_metadata(action, body, litellm_params)
    if action in _BEDROCK_INVOKE_ACTIONS and "requestMetadata" in body:
        return _without_owned_request_metadata_header(headers), _with_body_request_metadata(
            action, body, litellm_params
        )
    if action in _BEDROCK_INVOKE_ACTIONS:
        return _with_invoke_request_metadata(headers, body, litellm_params)
    return headers, body


def _with_invoke_request_metadata(
    headers: Mapping[str, str],
    body: Mapping[str, object],
    litellm_params: Mapping[str, object],
) -> tuple[Mapping[str, str], Mapping[str, object]]:
    owned_names, metadata_headers = bedrock_request_metadata_headers(litellm_params)
    return merge_bedrock_invoke_headers(dict(headers), (), metadata_headers, owned_names), body


def _converse_guardrail_config(action: str, guardrail_config: object) -> object:
    fields: Final = as_json_object(guardrail_config)
    if action == "converse-stream" or fields is None:
        return guardrail_config
    return {key: value for key, value in fields.items() if key != "streamProcessingMode"}


def _with_deployment_guardrail(
    action: str,
    headers: Mapping[str, str],
    body: Mapping[str, object],
    guardrail_config: object,
) -> tuple[Mapping[str, str], Mapping[str, object]]:
    if guardrail_config is None:
        return headers, body
    if action in _BEDROCK_CONVERSE_ACTIONS:
        return headers, {**body, "guardrailConfig": _converse_guardrail_config(action, guardrail_config)}
    if action in _BEDROCK_INVOKE_ACTIONS:
        guardrail_headers: Final = tuple(bedrock_invoke_guardrail_headers(guardrail_config).items())
        return merge_bedrock_invoke_headers(
            dict(headers), (), guardrail_headers, _BEDROCK_INVOKE_GUARDRAIL_HEADERS
        ), body
    return headers, body


def _is_cache_point(entry: object) -> bool:
    fields: Final = as_json_object(entry)
    return fields is not None and "cachePoint" in fields


def _tool_result_part(item: object) -> str | None:
    fields: Final = as_json_object(item)
    if fields is None:
        return None
    text: Final = fields.get("text")
    if isinstance(text, str):
        return text
    return json.dumps(fields["json"]) if "json" in fields else None


def _tool_result_text(tool_result: Mapping[str, object]) -> str:
    parts: Final = as_json_array(tool_result.get("content")) or ()
    rendered: Final = "\n".join(part for item in parts if (part := _tool_result_part(item)) is not None).strip()
    return rendered or "<non-text tool result omitted>"


def _block_field(block: object, key: str) -> Mapping[str, object] | None:
    fields: Final = as_json_object(block)
    return None if fields is None else as_json_object(fields.get(key))


def _as_text_block(block: object) -> object:
    tool_use: Final = _block_field(block, "toolUse")
    if tool_use is not None:
        arguments: Final = json.dumps(tool_use.get("input"))
        return {"text": f"[tool call {tool_use.get('toolUseId')}: {tool_use.get('name')}({arguments})]"}
    tool_result: Final = _block_field(block, "toolResult")
    if tool_result is None:
        return block
    return {"text": f"[tool result for {tool_result.get('toolUseId')}: {_tool_result_text(tool_result)}]"}


def _has_tool_block(content: Sequence[object]) -> bool:
    return any(
        _block_field(block, "toolUse") is not None or _block_field(block, "toolResult") is not None for block in content
    )


def _as_text_message(message: object) -> object:
    fields: Final = as_json_object(message)
    content: Final = None if fields is None else as_json_array(fields.get("content"))
    if fields is None or content is None or not _has_tool_block(content):
        return message
    return {**fields, "content": [_as_text_block(block) for block in content]}


def _converse_body_to_send(body: Mapping[str, object]) -> Mapping[str, object]:
    """
    ``body`` without a Converse ``toolConfig`` that holds no tool, with every ``toolUse`` and ``toolResult`` turned into
    text, since Bedrock rejects tool blocks without a tool. Any other ``body`` comes back as it is.
    """
    tool_config: Final = as_json_object(body.get("toolConfig"))
    tools: Final = None if tool_config is None else as_json_array(tool_config.get("tools"))
    if tools is None or not all(map(_is_cache_point, tools)):
        return body
    messages: Final = as_json_array(body.get("messages"))
    rest: Final = {key: value for key, value in body.items() if key != "toolConfig"}
    return rest if messages is None else {**rest, "messages": [_as_text_message(message) for message in messages]}


_TEXT_ONLY_DELTA_FIELDS: Final = frozenset({"content", "role"})


def _plain_text_delta(chunk: ModelResponseStream) -> str | None:
    """Return the delta text when the chunk carries nothing else that stream_chunk_builder reads."""
    if chunk.get("usage") is not None or chunk.provider_specific_fields or len(chunk.choices) != 1:
        return None
    choice: Final = chunk.choices[0]
    if choice.finish_reason or choice.logprobs is not None:
        return None
    delta_fields: Final[Mapping[str, object]] = MappingProxyType(choice.delta.model_dump())
    populated: Final = frozenset(key for key, value in delta_fields.items() if value is not None)
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

    translated_chunk: Final = decoder.chunk_parser(chunk_data=json.loads(message))
    if isinstance(translated_chunk, ModelResponseStream):
        return translated_chunk
    if generic_chunk_has_all_required_fields(cast(dict, translated_chunk)):
        return convert_generic_chunk_to_model_response_stream(cast(GenericStreamingChunk, translated_chunk))
    return None


_JSON_OBJECT: Final = TypeAdapter(Mapping[str, object])
_JSON_OBJECTS: Final = TypeAdapter(tuple[Mapping[str, object], ...])


def _leaf_texts(block: Mapping[str, object]) -> Iterator[str]:
    """The prompt text a content block carries itself: its text or JSON, or a tool call's input."""
    text: Final = block.get("text")
    if isinstance(text, str):
        yield text
    if "json" in block:
        yield json.dumps(block["json"], default=str)
    if "toolUse" in block:
        yield json.dumps(_JSON_OBJECT.validate_python(block["toolUse"]).get("input"), default=str)


def _block_texts(block: Mapping[str, object]) -> Iterator[str]:
    """The prompt text a request content block carries, including the blocks of a tool result's content, which
    Bedrock does not nest further."""
    yield from _leaf_texts(block)
    if "toolResult" not in block:
        return
    tool_result: Final = _JSON_OBJECT.validate_python(block["toolResult"])
    yield from chain.from_iterable(map(_leaf_texts, _JSON_OBJECTS.validate_python(tool_result.get("content", ()))))


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(chain.from_iterable(map(_block_texts, _JSON_OBJECTS.validate_python(content))))


def _chat_message(turn: Mapping[str, object]) -> ChatCompletionUserMessage:
    return ChatCompletionUserMessage(role="user", content=_content_text(turn.get("content", "")))


def _count_prompt_tokens(model: str, request_body: object) -> int:
    """Estimate a Bedrock request's prompt tokens from its system prompt and turns, the way a chat completions
    stream that ended before the provider reported usage is estimated."""
    from litellm.litellm_core_utils.token_counter import token_counter

    body: Final = _JSON_OBJECT.validate_python(request_body)
    if "messages" not in body:
        return 0
    system: Final = body.get("system")
    messages: Final = (
        *((ChatCompletionSystemMessage(role="system", content=_content_text(system)),) if system else ()),
        *map(_chat_message, _JSON_OBJECTS.validate_python(body["messages"])),
    )
    return token_counter(model=model, messages=messages)


def _build_logged_response(
    chunks: Sequence[ModelResponseStream], litellm_logging_obj: "LiteLLMLoggingObj"
) -> Optional["CostResponseTypes"]:
    from litellm.litellm_core_utils.streaming_chunk_builder_utils import stream_chunk_builder

    if len(chunks) == 0:
        return None
    return stream_chunk_builder(
        chunks=list(chunks),
        logging_obj=litellm_logging_obj,
        count_prompt_tokens=partial(
            _count_prompt_tokens, litellm_logging_obj.model, litellm_logging_obj.model_call_details.get("input")
        ),
    )


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
        headers: dict[str, str],
        litellm_params: dict[str, object],
        request_data: dict[str, object] | None,
        api_base: str,
        model: str | None = None,
    ) -> tuple[dict, bytes | None]:
        action: Final = _bedrock_endpoint_action(api_base)
        body: Final[Mapping[str, object]] = request_data or {}
        optional_params: Final = litellm_params.copy()
        signed_headers, signed_body = _with_request_metadata(
            action=action if bedrock_request_metadata_is_owned() else None,
            headers=headers,
            body=_converse_body_to_send(body) if action in _BEDROCK_CONVERSE_ACTIONS else body,
            litellm_params=litellm_params,
        )
        guarded_headers, guarded_body = _with_deployment_guardrail(
            action, signed_headers, signed_body, litellm_params.get("guardrailConfig")
        )
        return self._sign_request(
            service_name="bedrock",
            headers=dict(guarded_headers),
            optional_params=optional_params,
            request_data=dict(guarded_body),
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
