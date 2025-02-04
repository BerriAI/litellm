import json
from typing import AsyncIterator, Iterator, List, Optional, Union

import httpx

from litellm import verbose_logger
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.utils import GenericStreamingChunk as GChunk
from litellm.types.utils import StreamingChatCompletionChunk

_response_stream_shape_cache = None


class SagemakerError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: Optional[Union[dict, httpx.Headers]] = None,
    ):
        super().__init__(status_code=status_code, message=message, headers=headers)


class AWSEventStreamDecoder:
    def __init__(self, model: str, is_messages_api: Optional[bool] = None) -> None:
        from botocore.parsers import EventStreamJSONParser

        self.model = model
        self.parser = EventStreamJSONParser()
        self.content_blocks: List = []
        self.is_messages_api = is_messages_api

    def _chunk_parser_messages_api(
        self, chunk_data: dict
    ) -> StreamingChatCompletionChunk:

        openai_chunk = StreamingChatCompletionChunk(**chunk_data)

        return openai_chunk

    def _chunk_parser(self, chunk_data: dict) -> GChunk:
        verbose_logger.debug("in sagemaker chunk parser, chunk_data %s", chunk_data)
        _token = chunk_data.get("token", {}) or {}
        _index = chunk_data.get("index", None) or 0
        is_finished = False
        finish_reason = ""

        _text = _token.get("text", "")
        if _text == "<|endoftext|>":
            return GChunk(
                text="",
                index=_index,
                is_finished=True,
                finish_reason="stop",
                usage=None,
            )

        return GChunk(
            text=_text,
            index=_index,
            is_finished=is_finished,
            finish_reason=finish_reason,
            usage=None,
        )

    def iter_bytes(
        self, iterator: Iterator[bytes]
    ) -> Iterator[Optional[Union[GChunk, StreamingChatCompletionChunk]]]:
        """Given an iterator that yields lines, iterate over it & yield every event encountered"""
        from botocore.eventstream import EventStreamBuffer

        event_stream_buffer = EventStreamBuffer()
        accumulated_json = ""

        for chunk in iterator:
            event_stream_buffer.add_data(chunk)
            for event in event_stream_buffer:
                message = self._parse_message_from_event(event)
                if message:
                    # remove data: prefix and "\n\n" at the end
                    message = message.replace("data:", "").replace("\n\n", "")

                    # Accumulate JSON data
                    accumulated_json += message

                    # Try to parse the accumulated JSON
                    try:
                        _data = json.loads(accumulated_json)
                        if self.is_messages_api:
                            yield self._chunk_parser_messages_api(chunk_data=_data)
                        else:
                            yield self._chunk_parser(chunk_data=_data)
                        # Reset accumulated_json after successful parsing
                        accumulated_json = ""
                    except json.JSONDecodeError:
                        # If it's not valid JSON yet, continue to the next event
                        continue

        # Handle any remaining data after the iterator is exhausted
        if accumulated_json:
            try:
                _data = json.loads(accumulated_json)
                if self.is_messages_api:
                    yield self._chunk_parser_messages_api(chunk_data=_data)
                else:
                    yield self._chunk_parser(chunk_data=_data)
            except json.JSONDecodeError:
                # Handle or log any unparseable data at the end
                verbose_logger.error(
                    f"Warning: Unparseable JSON data remained: {accumulated_json}"
                )
                yield None

    async def aiter_bytes(
        self, iterator: AsyncIterator[bytes]
    ) -> AsyncIterator[Optional[Union[GChunk, StreamingChatCompletionChunk]]]:
        """Given an async iterator that yields lines, iterate over it & yield every event encountered"""
        from botocore.eventstream import EventStreamBuffer

        event_stream_buffer = EventStreamBuffer()
        accumulated_json = ""

        async for chunk in iterator:
            event_stream_buffer.add_data(chunk)
            for event in event_stream_buffer:
                message = self._parse_message_from_event(event)
                if message:
                    verbose_logger.debug("sagemaker  parsed chunk bytes %s", message)
                    # remove data: prefix and "\n\n" at the end
                    message = message.replace("data:", "").replace("\n\n", "")

                    # Accumulate JSON data
                    accumulated_json += message

                    # Try to parse the accumulated JSON
                    try:
                        _data = json.loads(accumulated_json)
                        if self.is_messages_api:
                            yield self._chunk_parser_messages_api(chunk_data=_data)
                        else:
                            yield self._chunk_parser(chunk_data=_data)
                        # Reset accumulated_json after successful parsing
                        accumulated_json = ""
                    except json.JSONDecodeError:
                        # If it's not valid JSON yet, continue to the next event
                        continue

        # Handle any remaining data after the iterator is exhausted
        if accumulated_json:
            try:
                _data = json.loads(accumulated_json)
                if self.is_messages_api:
                    yield self._chunk_parser_messages_api(chunk_data=_data)
                else:
                    yield self._chunk_parser(chunk_data=_data)
            except json.JSONDecodeError:
                # Handle or log any unparseable data at the end
                verbose_logger.error(
                    f"Warning: Unparseable JSON data remained: {accumulated_json}"
                )
                yield None

    def _parse_message_from_event(self, event) -> Optional[str]:
        response_dict = event.to_response_dict()
        parsed_response = self.parser.parse(response_dict, get_response_stream_shape())

        if response_dict["status_code"] != 200:
            raise ValueError(f"Bad response code, expected 200: {response_dict}")

        if "chunk" in parsed_response:
            chunk = parsed_response.get("chunk")
            if not chunk:
                return None
            return chunk.get("bytes").decode()  # type: ignore[no-any-return]
        else:
            chunk = response_dict.get("body")
            if not chunk:
                return None

            return chunk.decode()  # type: ignore[no-any-return]


def get_response_stream_shape():
    global _response_stream_shape_cache
    if _response_stream_shape_cache is None:

        from botocore.loaders import Loader
        from botocore.model import ServiceModel

        loader = Loader()
        sagemaker_service_dict = loader.load_service_model(
            "sagemaker-runtime", "service-2"
        )
        sagemaker_service_model = ServiceModel(sagemaker_service_dict)
        _response_stream_shape_cache = sagemaker_service_model.shape_for(
            "InvokeEndpointWithResponseStreamOutput"
        )
    return _response_stream_shape_cache
