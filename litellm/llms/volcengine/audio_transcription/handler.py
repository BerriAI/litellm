import asyncio
from typing import TYPE_CHECKING, Any, Coroutine

from litellm.constants import REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
from litellm.litellm_core_utils.asyncify import run_async_function
from litellm.litellm_core_utils.audio_utils.utils import (
    get_audio_file_name,
    process_audio_file,
)
from litellm.llms.custom_httpx.http_handler import get_shared_realtime_ssl_context
from litellm.llms.volcengine.audio_transcription.audio_utils import (
    chunk_volcengine_stt_pcm,
    resample_to_volcengine_stt_pcm,
)
from litellm.llms.volcengine.audio_transcription.sauc_protocol import (
    FLAG_NEG_SEQ,
    MSG_ERROR,
    MSG_FULL_SERVER,
    decode_sauc_frame,
    encode_sauc_audio_chunk,
    encode_sauc_json_config,
    parse_sauc_json_payload,
)
from litellm.llms.volcengine.audio_transcription.transformation import (
    VolcEngineAudioTranscriptionConfig,
    pick_stt_resource_id,
)
from litellm.llms.volcengine.common_utils import (
    VolcEngineError,
    get_volcengine_speech_api_key,
)
from litellm.types.utils import FileTypes, TranscriptionResponse
from litellm.utils import convert_to_model_response_object

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class VolcEngineAudioTranscription:
    def audio_transcriptions(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
        model_response: TranscriptionResponse,
        timeout: float,
        logging_obj: "LiteLLMLoggingObj",
        api_key: str | None,
        api_base: str | None,
        atranscription: bool = False,
        provider_config: VolcEngineAudioTranscriptionConfig | None = None,
    ) -> TranscriptionResponse | Coroutine[Any, Any, TranscriptionResponse]:
        config = provider_config or VolcEngineAudioTranscriptionConfig()
        if atranscription:
            return self.async_audio_transcriptions(
                model=model,
                audio_file=audio_file,
                optional_params=optional_params,
                litellm_params=litellm_params,
                model_response=model_response,
                timeout=timeout,
                logging_obj=logging_obj,
                api_key=api_key,
                api_base=api_base,
                provider_config=config,
            )
        return run_async_function(
            self.async_audio_transcriptions,
            model=model,
            audio_file=audio_file,
            optional_params=optional_params,
            litellm_params=litellm_params,
            model_response=model_response,
            timeout=timeout,
            logging_obj=logging_obj,
            api_key=api_key,
            api_base=api_base,
            provider_config=config,
        )

    async def async_audio_transcriptions(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
        model_response: TranscriptionResponse,
        timeout: float,
        logging_obj: "LiteLLMLoggingObj",
        api_key: str | None,
        api_base: str | None,
        provider_config: VolcEngineAudioTranscriptionConfig,
    ) -> TranscriptionResponse:
        response_format = optional_params.get("response_format") or "json"
        if response_format != "json":
            raise VolcEngineError(
                status_code=400,
                message="Volcengine STT adapter currently supports response_format=json.",
            )

        processed_audio = process_audio_file(audio_file)
        audio = resample_to_volcengine_stt_pcm(processed_audio.file_content)
        request_data = provider_config.transform_audio_transcription_request(
            model=model,
            audio_file=audio_file,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )
        if not isinstance(request_data.data, dict):
            raise VolcEngineError(
                status_code=500,
                message="Volcengine STT request payload must be an object.",
            )

        endpoint = provider_config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )
        speech_api_key = get_volcengine_speech_api_key(api_key)
        resource_id = litellm_params.get("resource_id") or pick_stt_resource_id(model)

        logging_obj.pre_call(
            input=get_audio_file_name(audio_file),
            api_key="",
            additional_args={
                "api_base": endpoint,
                "atranscription": True,
                "complete_input_dict": {
                    "model": model,
                    "response_format": response_format,
                    "resource_id": resource_id,
                    "duration_seconds": audio.duration_seconds,
                },
            },
        )

        transcript = await self._run_sauc_transcription(
            endpoint=endpoint,
            speech_api_key=speech_api_key,
            resource_id=resource_id,
            request_payload=request_data.data,
            pcm_bytes=audio.pcm_bytes,
            timeout=_timeout_seconds(timeout),
        )

        stringified_response = TranscriptionResponse(text=transcript).model_dump()
        logging_obj.post_call(
            input=get_audio_file_name(audio_file),
            api_key="",
            additional_args={"complete_input_dict": {"model": model}},
            original_response=stringified_response,
        )
        hidden_params = {
            "model": model,
            "custom_llm_provider": "volcengine",
            "audio_transcription_duration": audio.duration_seconds,
        }
        return convert_to_model_response_object(
            response_object=stringified_response,
            model_response_object=model_response,
            hidden_params=hidden_params,
            response_type="audio_transcription",
        )  # type: ignore

    async def _run_sauc_transcription(
        self,
        endpoint: str,
        speech_api_key: str,
        resource_id: str,
        request_payload: dict[str, Any],
        pcm_bytes: bytes,
        timeout: float,
    ) -> str:
        import websockets

        headers = {
            "X-Api-Key": speech_api_key,
            "X-Api-Resource-Id": resource_id,
            "X-Api-Request-Id": request_payload.get("request_id", "litellm-proxy"),
            "X-Api-Sequence": "-1",
        }
        final_texts: list[str] = []
        latest_text = ""
        sequence = 1
        ssl_context = get_shared_realtime_ssl_context()

        async with websockets.connect(  # type: ignore
            endpoint,
            additional_headers=headers,
            max_size=REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES,
            ssl=ssl_context,
        ) as ws:
            await ws.send(encode_sauc_json_config(request_payload))
            for chunk in chunk_volcengine_stt_pcm(pcm_bytes):
                sequence += 1
                await ws.send(
                    encode_sauc_audio_chunk(pcm=chunk, sequence=sequence, last=False)
                )
            sequence = -(abs(sequence) + 1)
            await ws.send(
                encode_sauc_audio_chunk(pcm=b"", sequence=sequence, last=True)
            )

            while True:
                message = await asyncio.wait_for(ws.recv(), timeout=timeout)
                if isinstance(message, str):
                    message = message.encode("utf-8")
                frame = decode_sauc_frame(bytes(message))
                if frame.message_type == MSG_ERROR:
                    detail = _decode_error_payload(frame.payload)
                    raise VolcEngineError(
                        status_code=502,
                        message=f"Volcengine STT server error code={frame.error_code}: {detail}",
                    )
                if frame.message_type != MSG_FULL_SERVER:
                    continue

                payload = parse_sauc_json_payload(frame)
                if payload is not None:
                    new_final_texts, new_latest = _extract_transcript_parts(
                        payload, frame.flags == FLAG_NEG_SEQ
                    )
                    final_texts.extend(new_final_texts)
                    if new_latest:
                        latest_text = new_latest
                if frame.flags == FLAG_NEG_SEQ:
                    break

        return "".join(final_texts).strip() or latest_text.strip()


def _extract_transcript_parts(
    payload: dict[str, Any], is_last_frame: bool
) -> tuple[list[str], str]:
    result = payload.get("result") if isinstance(payload, dict) else None
    utterances = result.get("utterances") if isinstance(result, dict) else None
    final_texts: list[str] = []
    latest_text = ""
    if isinstance(utterances, list) and utterances:
        for utterance in utterances:
            if not isinstance(utterance, dict):
                continue
            text = utterance.get("text")
            if not isinstance(text, str) or not text:
                continue
            if utterance.get("definite") or is_last_frame:
                final_texts.append(text)
            else:
                latest_text = text
    elif isinstance(result, dict) and isinstance(result.get("text"), str):
        if is_last_frame:
            final_texts.append(result["text"])
        else:
            latest_text = result["text"]
    return final_texts, latest_text


def _decode_error_payload(payload: bytes) -> str:
    if not payload:
        return "empty error payload"
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return "binary error payload"


def _timeout_seconds(timeout: Any) -> float:
    if isinstance(timeout, (int, float)):
        return float(timeout)
    if timeout is not None and hasattr(timeout, "read"):
        return float(timeout.read or timeout.connect or 600.0)
    return 600.0
