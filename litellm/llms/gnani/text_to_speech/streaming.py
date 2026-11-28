import base64
from collections.abc import AsyncIterator, Iterator, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import httpx
from openai._streaming import SSEDecoder
from pydantic import TypeAdapter

from litellm.exceptions import BadGatewayError
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.types.llms.openai import HttpxBinaryResponseContent

from .transformation import GnaniTextToSpeechConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging

_PAYLOAD: Final = TypeAdapter(Mapping[str, object])
_MEDIA_TYPES: Final = MappingProxyType(
    {
        "wav": "audio/wav",
        "raw": "audio/pcm",
        "mp3": "audio/mpeg",
        "ogg": "audio/ogg",
        "mulaw": "audio/basic",
        "alaw": "audio/alaw",
    }
)
_ENCODINGS: Final = MappingProxyType({"pcm_mulaw": "mulaw", "pcm_alaw": "alaw", "oggopus": "ogg"})


class _AudioStream(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, response: httpx.Response, model: str) -> None:
        self._response: Final = response
        self._model: Final = model
        self._decoder: Final = SSEDecoder()
        self._received_audio = False

    def _failure(self, message: str) -> BadGatewayError:
        return BadGatewayError(message, llm_provider="gnani", model=self._model)

    def _audio(self, line: str) -> bytes | None:
        event: Final = self._decoder.decode(line)
        if event is None or event.event not in ("chunk", "complete", "error"):
            return b""
        if event.event == "complete":
            if not self._received_audio:
                raise self._failure("Gnani completed synthesis without sending audio.")
            return None
        try:
            payload: Final = _PAYLOAD.validate_json(event.data)
            if event.event == "error":
                raise self._failure(str(payload.get("message") or "Gnani speech synthesis failed."))
            encoded: Final = payload.get("audio")
            if not isinstance(encoded, str):
                raise self._failure("Gnani sent a speech chunk without an audio string.")
            audio: Final = base64.b64decode(encoded, validate=True)
        except ValueError as error:
            raise self._failure("Gnani sent malformed speech data.") from error
        if audio:
            self._received_audio = True
        return audio

    def __iter__(self) -> Iterator[bytes]:
        try:
            for line in self._response.iter_lines():
                if (audio := self._audio(line)) is None:
                    return
                if audio:
                    yield audio
            raise self._failure("Gnani disconnected before the complete event.")
        finally:
            self.close()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for line in self._response.aiter_lines():
                if (audio := self._audio(line)) is None:
                    return
                if audio:
                    yield audio
            raise self._failure("Gnani disconnected before the complete event.")
        finally:
            await self.aclose()

    def close(self) -> None:
        self._response.close()

    async def aclose(self) -> None:
        await self._response.aclose()


class GnaniStreamingTextToSpeechConfig(GnaniTextToSpeechConfig):
    supports_streaming = True
    endpoint = "sse"

    def __init__(self, params: Mapping[str, object]) -> None:
        config: Final = as_str_mapping(params.get("audio_config")) or {}
        container: Final = str(config.get("container", "raw"))
        output: Final = _ENCODINGS.get(str(config.get("encoding")), container) if container != "ogg" else container
        self._media_type: Final = _MEDIA_TYPES.get(output, "application/octet-stream")

    def _response(self, model: str, raw: httpx.Response) -> HttpxBinaryResponseContent:
        content_type: Final = next(iter(raw.headers.get_list("content-type")), "")
        if content_type.partition(";")[0].strip().lower() != "text/event-stream":
            raise BadGatewayError("Gnani returned a non-SSE speech response.", llm_provider="gnani", model=model)
        return HttpxBinaryResponseContent(
            httpx.Response(
                raw.status_code,
                request=raw.request,
                headers={"Content-Type": self._media_type},
                stream=_AudioStream(raw, model),
            )
        )

    def transform_text_to_speech_response(
        self, model: str, raw_response: httpx.Response, logging_obj: "Logging"
    ) -> HttpxBinaryResponseContent:
        try:
            return self._response(model, raw_response)
        except BadGatewayError:
            raw_response.close()
            raise

    async def async_transform_text_to_speech_response(
        self, model: str, raw_response: httpx.Response, logging_obj: "Logging"
    ) -> HttpxBinaryResponseContent:
        try:
            return self._response(model, raw_response)
        except BadGatewayError:
            await raw_response.aclose()
            raise
