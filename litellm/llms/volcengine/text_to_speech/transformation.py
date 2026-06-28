import asyncio
import uuid
import wave
from io import BytesIO
from typing import TYPE_CHECKING, Any, Coroutine

import httpx

from litellm.constants import REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES
from litellm.litellm_core_utils.asyncify import run_async_function
from litellm.llms.base_llm.text_to_speech.transformation import (
    BaseTextToSpeechConfig,
    TextToSpeechRequestData,
)
from litellm.llms.custom_httpx.http_handler import get_shared_realtime_ssl_context
from litellm.llms.volcengine.common_utils import (
    VolcEngineError,
    get_volcengine_configured_ws_api_base,
    get_volcengine_speech_api_key,
)
from litellm.llms.volcengine.text_to_speech.protocol import (
    EV_CONNECTION_FAILED,
    EV_CONNECTION_FINISHED,
    EV_CONNECTION_STARTED,
    EV_FINISH_CONNECTION,
    EV_FINISH_SESSION,
    EV_SESSION_FAILED,
    EV_SESSION_FINISHED,
    EV_SESSION_STARTED,
    EV_START_CONNECTION,
    EV_START_SESSION,
    EV_TASK_REQUEST,
    EV_TTS_RESPONSE,
    MSG_AUDIO_SERVER,
    MSG_ERROR,
    decode_event_frame,
    encode_json_event,
    parse_json_payload,
)
from litellm.types.llms.openai import HttpxBinaryResponseContent

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

VOLCENGINE_TTS_DEFAULT_API_BASE = "wss://openspeech.bytedance.com/api/v3/tts/bidirection"
VOLCENGINE_TTS_DEFAULT_VOICE = "zh_female_vv_uranus_bigtts"
VOLCENGINE_TTS_SAMPLE_RATE_HZ = 24000
VOLCENGINE_TTS_CHANNELS = 1
VOLCENGINE_TTS_REQUEST_MODEL_NAMES = {
    "seed-tts-2.0-standard",
    "seed-tts-2.0-expressive",
}
VOLCENGINE_OPENAI_VOICE_NAMES = {
    "alloy",
    "ash",
    "ballad",
    "cedar",
    "coral",
    "echo",
    "fable",
    "marin",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
}


class VolcEngineTextToSpeechConfig(BaseTextToSpeechConfig):
    def get_supported_openai_params(self, model: str) -> list:
        return ["voice", "response_format"]

    def map_openai_params(
        self,
        model: str,
        optional_params: dict,
        voice: str | dict | None = None,
        drop_params: bool = False,
        kwargs: dict = {},
    ) -> tuple[str | None, dict]:
        response_format = optional_params.get("response_format") or "pcm"
        optional_params["response_format"] = response_format
        if isinstance(voice, str) and voice.strip():
            voice_name = voice.strip()
            if voice_name.lower() in VOLCENGINE_OPENAI_VOICE_NAMES:
                voice_name = VOLCENGINE_TTS_DEFAULT_VOICE
        else:
            voice_name = VOLCENGINE_TTS_DEFAULT_VOICE
        return voice_name, optional_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        return get_volcengine_configured_ws_api_base(
            litellm_params=litellm_params,
            default_api_base=VOLCENGINE_TTS_DEFAULT_API_BASE,
        )

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> TextToSpeechRequestData:
        return {
            "dict_body": {
                "model": model,
                "input": input,
                "voice": voice or VOLCENGINE_TTS_DEFAULT_VOICE,
                "response_format": optional_params.get("response_format") or "pcm",
            }
        }

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> HttpxBinaryResponseContent:
        return HttpxBinaryResponseContent(raw_response)

    def dispatch_text_to_speech(
        self,
        model: str,
        input: str,
        voice: str | dict | None,
        optional_params: dict,
        litellm_params_dict: dict,
        logging_obj: "LiteLLMLoggingObj",
        timeout: float | httpx.Timeout,
        extra_headers: dict[str, Any] | None,
        aspeech: bool,
        api_base: str | None,
        api_key: str | None,
        **kwargs: Any,
    ) -> HttpxBinaryResponseContent | Coroutine[Any, Any, HttpxBinaryResponseContent]:
        endpoint = self.get_complete_url(
            model=model,
            api_base=api_base,
            litellm_params=litellm_params_dict,
        )
        resolved_api_key = api_key or litellm_params_dict.get("api_key")
        voice_name = voice if isinstance(voice, str) else VOLCENGINE_TTS_DEFAULT_VOICE
        resolved_optional_params = {**optional_params}
        resource_id = litellm_params_dict.get("resource_id") or pick_tts_resource_id(model)
        if aspeech:
            return self._async_dispatch(
                model=model,
                input=input,
                voice=voice_name,
                optional_params=resolved_optional_params,
                logging_obj=logging_obj,
                timeout=timeout,
                api_base=endpoint,
                api_key=resolved_api_key,
                resource_id=resource_id,
            )
        return run_async_function(
            self._async_dispatch,
            model=model,
            input=input,
            voice=voice_name,
            optional_params=resolved_optional_params,
            logging_obj=logging_obj,
            timeout=timeout,
            api_base=endpoint,
            api_key=resolved_api_key,
            resource_id=resource_id,
        )

    async def _async_dispatch(
        self,
        model: str,
        input: str,
        voice: str,
        optional_params: dict,
        logging_obj: "LiteLLMLoggingObj",
        timeout: float | httpx.Timeout,
        api_base: str,
        api_key: str | None,
        resource_id: str,
    ) -> HttpxBinaryResponseContent:
        response_format = optional_params.get("response_format") or "pcm"
        if response_format not in {"pcm", "wav"}:
            raise VolcEngineError(
                status_code=400,
                message="Volcengine TTS adapter currently supports response_format=pcm or wav.",
            )
        speed = optional_params.get("speed")
        if speed not in (None, 1, 1.0):
            raise VolcEngineError(
                status_code=400,
                message="Volcengine TTS adapter currently supports speed=1 only.",
            )

        speech_api_key = get_volcengine_speech_api_key(api_key)
        logging_obj.pre_call(
            input=input,
            api_key="",
            additional_args={
                "api_base": api_base,
                "complete_input_dict": {
                    "model": model,
                    "voice": voice,
                    "resource_id": resource_id,
                    "response_format": response_format,
                },
            },
        )

        request_model = pick_tts_request_model(model, optional_params)
        pcm = await self._run_tts_session(
            endpoint=api_base,
            speech_api_key=speech_api_key,
            resource_id=resource_id,
            request_model=request_model,
            text=input,
            voice=voice,
            timeout=_timeout_seconds(timeout),
        )
        content = pcm if response_format == "pcm" else _wrap_pcm_as_wav(pcm)
        content_type = "audio/pcm" if response_format == "pcm" else "audio/wav"
        logging_obj.post_call(
            input=input,
            api_key="",
            additional_args={"complete_input_dict": {"model": model}},
            original_response={
                "bytes": len(content),
                "response_format": response_format,
            },
        )
        response = httpx.Response(
            status_code=200,
            content=content,
            headers={"content-type": content_type},
        )
        binary_response = HttpxBinaryResponseContent(response)
        binary_response._hidden_params = {
            "model": model,
            "custom_llm_provider": "volcengine",
            "content_type": content_type,
        }
        return binary_response

    async def _run_tts_session(
        self,
        endpoint: str,
        speech_api_key: str,
        resource_id: str,
        request_model: str | None,
        text: str,
        voice: str,
        timeout: float,
    ) -> bytes:
        import websockets

        session_id = str(uuid.uuid4())
        connect_id = str(uuid.uuid4())
        audio_chunks: list[bytes] = []
        headers = {
            "X-Api-Key": speech_api_key,
            "X-Api-Resource-Id": resource_id,
            "X-Api-Connect-Id": connect_id,
        }
        ssl_context = get_shared_realtime_ssl_context()
        async with websockets.connect(  # type: ignore
            endpoint,
            additional_headers=headers,
            max_size=REALTIME_WEBSOCKET_MAX_MESSAGE_SIZE_BYTES,
            ssl=ssl_context,
        ) as ws:
            await ws.send(encode_json_event(event=EV_START_CONNECTION, payload={}))
            await self._wait_for_event(ws, {EV_CONNECTION_STARTED}, timeout)
            await ws.send(
                encode_json_event(
                    event=EV_START_SESSION,
                    session_id=session_id,
                    payload=_start_session_payload(voice=voice, request_model=request_model),
                )
            )
            await self._wait_for_event(ws, {EV_SESSION_STARTED}, timeout)
            await ws.send(
                encode_json_event(
                    event=EV_TASK_REQUEST,
                    session_id=session_id,
                    payload={
                        "user": {"uid": "litellm-proxy"},
                        "namespace": "BidirectionalTTS",
                        "req_params": {"text": text},
                    },
                )
            )
            await ws.send(encode_json_event(event=EV_FINISH_SESSION, session_id=session_id, payload={}))
            while True:
                frame = await self._recv_frame(ws, timeout)
                if frame.message_type == MSG_AUDIO_SERVER and frame.event == EV_TTS_RESPONSE:
                    audio_chunks.append(bytes(frame.payload))
                    continue
                if frame.event == EV_SESSION_FINISHED:
                    break
                _raise_for_tts_failure(frame)

            try:
                await ws.send(encode_json_event(event=EV_FINISH_CONNECTION, payload={}))
                await asyncio.wait_for(self._wait_for_event(ws, {EV_CONNECTION_FINISHED}, 0.5), 0.5)
            except (asyncio.TimeoutError, ConnectionError, OSError, VolcEngineError):
                pass
        return b"".join(audio_chunks)

    async def _wait_for_event(self, ws: Any, expected_events: set[int], timeout: float) -> None:
        while True:
            frame = await self._recv_frame(ws, timeout)
            if frame.event in expected_events:
                return
            _raise_for_tts_failure(frame)

    async def _recv_frame(self, ws: Any, timeout: float):
        message = await asyncio.wait_for(ws.recv(), timeout=timeout)
        if isinstance(message, str):
            message = message.encode("utf-8")
        return decode_event_frame(bytes(message))


def pick_tts_resource_id(model_name: str | None) -> str:
    model = _normalize_volcengine_model_name(model_name)
    if not model:
        return "seed-tts-2.0"
    if model in VOLCENGINE_TTS_REQUEST_MODEL_NAMES:
        return "seed-tts-2.0"
    if "icl" in model and "2.0" in model:
        return "seed-icl-2.0"
    if "icl" in model and "1.0-concurr" in model:
        return "seed-icl-1.0-concurr"
    if "icl" in model:
        return "seed-icl-1.0"
    if "1.0-concurr" in model:
        return "seed-tts-1.0-concurr"
    if model.startswith("seed-tts-1") or "1.0" in model:
        return "seed-tts-1.0"
    return "seed-tts-2.0"


def pick_tts_request_model(model_name: str | None, optional_params: dict[str, Any] | None = None) -> str | None:
    params = optional_params or {}
    explicit_model = params.get("request_model") or params.get("tts_model")
    if isinstance(explicit_model, str):
        model = _normalize_volcengine_model_name(explicit_model)
        if model in VOLCENGINE_TTS_REQUEST_MODEL_NAMES:
            return model
    model = _normalize_volcengine_model_name(model_name)
    if model in VOLCENGINE_TTS_REQUEST_MODEL_NAMES:
        return model
    return None


def _start_session_payload(voice: str, request_model: str | None = None) -> dict[str, Any]:
    payload = {
        "user": {"uid": "litellm-proxy"},
        "namespace": "BidirectionalTTS",
        "req_params": {
            "speaker": voice or VOLCENGINE_TTS_DEFAULT_VOICE,
            "audio_params": {
                "format": "pcm",
                "sample_rate": VOLCENGINE_TTS_SAMPLE_RATE_HZ,
            },
        },
    }
    if request_model is not None:
        payload["req_params"]["model"] = request_model
    return payload


def _normalize_volcengine_model_name(model_name: str | None) -> str:
    model = (model_name or "").lower().strip()
    if model.startswith("volcengine/"):
        model = model.split("/", 1)[1]
    return model


def _raise_for_tts_failure(frame: Any) -> None:
    if frame.message_type == MSG_ERROR:
        detail = _safe_tts_payload(frame)
        raise VolcEngineError(
            status_code=502,
            message=f"Volcengine TTS server error code={frame.error_code}: {detail}",
        )
    if frame.event in {EV_CONNECTION_FAILED, EV_SESSION_FAILED}:
        detail = _safe_tts_payload(frame)
        raise VolcEngineError(
            status_code=502,
            message=f"Volcengine TTS session failed event={frame.event}: {detail}",
        )


def _safe_tts_payload(frame: Any) -> str:
    try:
        payload = parse_json_payload(frame)
        return str(payload.get("message") or payload.get("error") or payload)
    except (EOFError, OSError, UnicodeDecodeError, ValueError, VolcEngineError):
        return "unparseable error payload"


def _timeout_seconds(timeout: float | httpx.Timeout | None) -> float:
    if isinstance(timeout, (int, float)):
        return float(timeout)
    if timeout is not None:
        return float(timeout.read or timeout.connect or 600.0)
    return 600.0


def _wrap_pcm_as_wav(pcm: bytes) -> bytes:
    buf = BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(VOLCENGINE_TTS_CHANNELS)
        wav.setsampwidth(2)
        wav.setframerate(VOLCENGINE_TTS_SAMPLE_RATE_HZ)
        wav.writeframes(pcm)
    return buf.getvalue()
