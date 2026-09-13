import functools
import json
from collections.abc import AsyncIterator, Iterator
from typing import Final

import httpx

import litellm
from litellm import verbose_logger
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.utils import GenericStreamingChunk as GChunk
from litellm.types.utils import StreamingChatCompletionChunk


def _load_sagemaker_response_stream_shape():
    try:
        from botocore.loaders import Loader
        from botocore.model import ServiceModel

        loader: Final = Loader()
        service_dict: Final = loader.load_service_model("sagemaker-runtime", "service-2")
        return ServiceModel(service_dict).shape_for("InvokeEndpointWithResponseStreamOutput")
    except Exception as e:
        verbose_logger.warning(
            "litellm: could not load sagemaker-runtime response stream shape "
            "— SageMaker event-stream decoding will be unavailable. Error: %s",
            e,
        )
        return None


@functools.lru_cache(maxsize=1)
def get_sagemaker_response_stream_shape():
    """
    Lazily load and cache the sagemaker-runtime stream shape for the process.

    Avoids importing botocore (and logging warnings) unless SageMaker event-stream
    decoding is actually needed.
    """
    return _load_sagemaker_response_stream_shape()


class SagemakerError(BaseLLMException):
    def __init__(
        self,
        status_code: int,
        message: str,
        headers: dict | httpx.Headers | None = None,
    ):
        super().__init__(status_code=status_code, message=message, headers=headers)


class AWSEventStreamDecoder:
    def __init__(self, model: str, is_messages_api: bool | None = None) -> None:
        from botocore.parsers import EventStreamJSONParser

        self.model = model
        self.parser = EventStreamJSONParser()
        self.content_blocks: list = []
        self.is_messages_api = is_messages_api

    def _chunk_parser_messages_api(self, chunk_data: dict) -> StreamingChatCompletionChunk:
        openai_chunk: Final = StreamingChatCompletionChunk(**{"model": self.model, **chunk_data})

        return openai_chunk

    def _chunk_parser(self, chunk_data: dict) -> GChunk:
        verbose_logger.debug("in sagemaker chunk parser, chunk_data %s", chunk_data)
        _token: Final = chunk_data.get("token", {}) or {}
        _index: Final = chunk_data.get("index", None) or 0
        is_finished: Final = False
        finish_reason: Final = ""

        _text: Final = _token.get("text", "")
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

    def iter_bytes(self, iterator: Iterator[bytes]) -> Iterator[GChunk | StreamingChatCompletionChunk | None]:
        """Given an iterator that yields lines, iterate over it & yield every event encountered"""
        from botocore.eventstream import EventStreamBuffer

        event_stream_buffer: Final = EventStreamBuffer()
        accumulated_json = ""

        for chunk in iterator:
            event_stream_buffer.add_data(chunk)
            for event in event_stream_buffer:
                message = self._parse_message_from_event(event)
                if message:
                    # remove data: prefix and "\n\n" at the end
                    message = litellm.CustomStreamWrapper._strip_sse_data_from_chunk(message) or ""
                    message = message.replace("\n\n", "")

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
                verbose_logger.error("Warning: Unparseable JSON data remained: %s", accumulated_json)
                yield None

    async def aiter_bytes(
        self, iterator: AsyncIterator[bytes]
    ) -> AsyncIterator[GChunk | StreamingChatCompletionChunk | None]:
        """Given an async iterator that yields lines, iterate over it & yield every event encountered"""
        from botocore.eventstream import EventStreamBuffer

        event_stream_buffer: Final = EventStreamBuffer()
        accumulated_json = ""

        async for chunk in iterator:
            event_stream_buffer.add_data(chunk)
            for event in event_stream_buffer:
                try:
                    message = self._parse_message_from_event(event)
                    if message:
                        verbose_logger.debug("sagemaker  parsed chunk bytes %s", message)
                        # remove data: prefix and "\n\n" at the end
                        message = litellm.CustomStreamWrapper._strip_sse_data_from_chunk(message) or ""
                        message = message.replace("\n\n", "")

                        # Accumulate JSON data
                        accumulated_json += message

                        # Try to parse the accumulated JSON
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
                except UnicodeDecodeError as e:
                    verbose_logger.warning("UnicodeDecodeError: %s. Attempting to combine with next event.", e)
                    continue
                except Exception as e:
                    verbose_logger.error("Error parsing message: %s. Attempting to combine with next event.", e)
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
                verbose_logger.error("Warning: Unparseable JSON data remained: %s", accumulated_json)
                yield None
            except Exception as e:
                verbose_logger.error("Final error parsing accumulated JSON: %s", e)

    def _parse_message_from_event(self, event) -> str | None:
        response_stream_shape: Final = get_sagemaker_response_stream_shape()
        if response_stream_shape is None:
            raise SagemakerError(
                status_code=500,
                message=(
                    "SageMaker event-stream shape could not be loaded from botocore. "
                    "Ensure botocore is correctly installed."
                ),
            )
        response_dict: Final = event.to_response_dict()
        parsed_response: Final = self.parser.parse(response_dict, response_stream_shape)

        if response_dict["status_code"] != 200:
            raise ValueError(f"Bad response code, expected 200: {response_dict}")

        if "chunk" in parsed_response:
            chunk = parsed_response.get("chunk")
            if not chunk:
                return None
            return chunk.get("bytes").decode()
        else:
            chunk = response_dict.get("body")
            if not chunk:
                return None

            return chunk.decode()
