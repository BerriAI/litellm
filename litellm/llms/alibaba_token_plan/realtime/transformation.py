import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, cast  # noqa: TID251  # OpenAI's union omits validated provider extension events
from urllib.parse import urlencode

from pydantic import JsonValue

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.alibaba_token_plan.common_utils import REALTIME_PATH, get_api_url, validate_headers
from litellm.llms.base_llm.realtime.transcription_protocol import (
    RealtimeTranscriptionProtocolError,
    json_mapping,
    json_object,
)
from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig
from litellm.types.llms.openai import OpenAIRealtimeEvents
from litellm.types.realtime import RealtimeResponseTransformInput, RealtimeResponseTypedDict

_EVENT_TYPES: Final = MappingProxyType(
    {
        "conversation.item.created": "conversation.item.added",
        "response.text.delta": "response.output_text.delta",
        "response.text.done": "response.output_text.done",
        "response.audio.delta": "response.output_audio.delta",
        "response.audio.done": "response.output_audio.done",
        "response.audio_transcript.delta": "response.output_audio_transcript.delta",
        "response.audio_transcript.done": "response.output_audio_transcript.done",
    }
)
_SESSION_KEYS: Final = frozenset(
    (
        "modalities",
        "voice",
        "instructions",
        "enable_speech_emotion",
        "input_audio_transcription",
        "output_audio",
        "max_history_turns",
        "enable_search",
        "search_options",
        "tools",
        "turn_detection",
    )
)
_ITEM_CONTENT_TYPES: Final = MappingProxyType({"audio": "output_audio", "text": "output_text"})
_USAGE_KEYS: Final = MappingProxyType(
    {"input_tokens_details": "input_token_details", "output_tokens_details": "output_token_details"}
)


def _items(value: JsonValue | None, name: str) -> Sequence[JsonValue]:
    if not isinstance(value, list):
        raise RealtimeTranscriptionProtocolError(f"{name} must be an array")
    return value


def _tools(value: JsonValue | None) -> Sequence[Mapping[str, JsonValue]]:
    return tuple(_tool(json_mapping(item, "tool")) for item in _items(value, "tools"))


def _tool(tool: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    function: Final = json_mapping(tool.get("function"), "function") if "function" in tool else tool
    return {"type": "function", "function": {key: value for key, value in function.items() if key != "type"}}


def _disables_turn_control(value: JsonValue | None) -> bool:
    return isinstance(value, dict) and (
        value.get("create_response") is False or value.get("interrupt_response") is False
    )


def _turn_detection(value: JsonValue) -> Mapping[str, JsonValue] | None:
    if value is None:
        return None
    config: Final = json_mapping(value, "turn_detection")
    if config.get("type", "server_vad") not in ("server_vad", "smart_turn"):
        raise RealtimeTranscriptionProtocolError("Qwen Audio turn_detection supports server_vad, smart_turn, or null")
    if _disables_turn_control(value):
        raise RealtimeTranscriptionProtocolError(
            "Qwen Audio cannot disable VAD responses or interruptions; transcription guardrails are not supported"
        )
    return {key: value for key, value in config.items() if key not in ("create_response", "interrupt_response")}


def _audio_rate(value: JsonValue | None, *, output: bool = False) -> int:
    if value in (None, "pcm16"):
        return 24000
    if value == "pcm":
        return 24000 if output else 16000
    config: Final = json_mapping(value, "audio format")
    rate: Final = config.get("rate", 24000)
    if config.get("type") != "audio/pcm" or config.get("channels", 1) != 1:
        raise RealtimeTranscriptionProtocolError("Qwen Audio realtime requires mono PCM16 audio")
    if not isinstance(rate, int) or rate != (24000 if output else 16000):
        raise RealtimeTranscriptionProtocolError("Input PCM must be 16000 Hz; output PCM must be 24000 Hz")
    return rate


def _modalities(value: JsonValue | None) -> Sequence[str]:
    modalities: Final = _items(value, "modalities")
    if not modalities or any(item not in ("audio", "text") for item in modalities):
        raise RealtimeTranscriptionProtocolError("Qwen Audio realtime supports text and audio output only")
    return ("text", "audio") if "audio" in modalities else ("text",)


def _part_to_ga(part: JsonValue, *, item_content: bool) -> JsonValue:
    if not isinstance(part, dict):
        return part
    part_type: Final = part.get("type")
    renamed_type: Final = _ITEM_CONTENT_TYPES.get(str(part_type), part_type) if item_content else part_type
    return {
        **part,
        **({"type": renamed_type} if "type" in part else {}),
        **({"transcript": part["text"]} if part_type == "audio" and "text" in part else {}),
    }


def _item_to_ga(item: JsonValue) -> JsonValue:
    if not isinstance(item, dict) or not isinstance(item.get("content"), list):
        return item
    content: Final = _items(item["content"], "content")
    return {**item, "content": [_part_to_ga(part, item_content=True) for part in content]}


def _usage_to_ga(usage: JsonValue) -> JsonValue:
    if not isinstance(usage, dict):
        return usage
    return {_USAGE_KEYS.get(key, key): value for key, value in usage.items()}


def _response_to_ga(response: JsonValue) -> JsonValue:
    if not isinstance(response, dict):
        return response
    output: Final = response.get("output")
    return {
        **response,
        **({"output": [_item_to_ga(item) for item in output]} if isinstance(output, list) else {}),
        **({"usage": _usage_to_ga(response["usage"])} if "usage" in response else {}),
    }


def _event_to_ga(event: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return {
        **event,
        **({"item": _item_to_ga(event["item"])} if "item" in event else {}),
        **({"part": _part_to_ga(event["part"], item_content=False)} if "part" in event else {}),
        **({"usage": _usage_to_ga(event["usage"])} if "usage" in event else {}),
        **({"response": _response_to_ga(event["response"])} if "response" in event else {}),
    }


class AlibabaTokenPlanRealtimeConfig(BaseRealtimeConfig):
    def __init__(self) -> None:
        self._transcription_prefixes: Mapping[tuple[str, int], str] = MappingProxyType({})
        self._input_rate_declared = False
        self._audio_blocked = False

    def validate_environment(
        self, headers: Mapping[str, str], model: str, api_key: str | None = None
    ) -> dict[str, str]:  # mutable-ok: BaseRealtimeConfig requires a dictionary
        return validate_headers(headers, api_key)

    def get_complete_url(self, api_base: str | None, model: str, api_key: str | None = None) -> str:
        url: Final = get_api_url(api_base, REALTIME_PATH)
        return f"{url.replace('https://', 'wss://', 1).replace('http://', 'ws://', 1)}?{urlencode({'model': model})}"

    def _session_request(self, session: Mapping[str, JsonValue]) -> Mapping[str, object]:
        raw_audio: Final = session.get("audio")
        raw_audio_input: Final = raw_audio.get("input") if isinstance(raw_audio, dict) else None
        turn_controls: Final = (
            session.get("turn_detection"),
            raw_audio_input.get("turn_detection") if isinstance(raw_audio_input, dict) else None,
        )
        if any(_disables_turn_control(value) for value in turn_controls):
            self._audio_blocked = True
        for turn_control in turn_controls:
            _turn_detection(turn_control)
        if session.get("type", "realtime") != "realtime":
            raise RealtimeTranscriptionProtocolError("Qwen Audio supports realtime conversation sessions only")
        if session.get("tool_choice", "auto") != "auto":
            raise RealtimeTranscriptionProtocolError("Qwen Audio supports automatic function tool choice only")
        audio: Final = json_mapping(session.get("audio"), "audio")
        audio_input: Final = json_mapping(audio.get("input"), "audio.input")
        audio_output: Final = json_mapping(audio.get("output"), "audio.output")
        input_format: Final = audio_input.get("format", session.get("input_audio_format"))
        if input_format is not None and _audio_rate(input_format) != 16000:
            raise RealtimeTranscriptionProtocolError("Qwen Audio realtime input must be 16000 Hz PCM16")
        if input_format is not None:
            self._input_rate_declared = True
        _audio_rate(audio_output.get("format", session.get("output_audio_format")), output=True)
        normalized: Final[Mapping[str, JsonValue]] = {
            **{key: value for key, value in session.items() if key in _SESSION_KEYS},
            **({"modalities": session["output_modalities"]} if "output_modalities" in session else {}),
            **({"voice": audio_output["voice"]} if "voice" in audio_output else {}),
            **({"turn_detection": audio_input["turn_detection"]} if "turn_detection" in audio_input else {}),
            **({"input_audio_transcription": audio_input["transcription"]} if "transcription" in audio_input else {}),
        }
        result: Final = {
            **normalized,
            "input_audio_format": "pcm",
            "output_audio_format": "pcm",
            **({"tools": _tools(normalized["tools"])} if "tools" in normalized else {}),
            **({"modalities": _modalities(normalized["modalities"])} if "modalities" in normalized else {}),
            **(
                {"turn_detection": _turn_detection(normalized["turn_detection"])}
                if "turn_detection" in normalized
                else {}
            ),
        }
        return result

    def _item_request(self, item: Mapping[str, JsonValue]) -> Mapping[str, object]:
        if "content" not in item:
            return item
        return {
            **item,
            "content": tuple(
                self._input_content(json_mapping(part, "content")) for part in _items(item["content"], "content")
            ),
        }

    def _input_content(self, content: Mapping[str, JsonValue]) -> Mapping[str, object]:
        if content.get("type") == "text":
            return {**content, "type": "output_text"}
        if content.get("type") == "input_audio":
            self._validate_audio_input()
        return content

    def _validate_audio_input(self) -> None:
        if self._audio_blocked:
            raise RealtimeTranscriptionProtocolError(
                "Qwen Audio realtime audio is blocked because the session requested unsupported turn controls"
            )
        if not self._input_rate_declared:
            raise RealtimeTranscriptionProtocolError(
                "Qwen Audio realtime input is 16000 Hz PCM16; declare that input format in session.update before "
                "sending audio"
            )

    def transform_realtime_request(
        self, message: str, model: str, session_configuration_request: str | None = None
    ) -> Sequence[str | bytes]:
        event: Final = json_object(message)
        event_type: Final = event.get("type")
        if event_type == "session.update":
            return (
                json.dumps({**event, "session": self._session_request(json_mapping(event.get("session"), "session"))}),
            )
        if event_type == "input_audio_buffer.append":
            self._validate_audio_input()
        if event_type == "conversation.item.create":
            return (json.dumps({**event, "item": self._item_request(json_mapping(event.get("item"), "item"))}),)
        if event_type == "response.create" and "response" in event:
            response: Final = json_mapping(event["response"], "response")
            normalized: Final = {key: value for key, value in response.items() if key != "output_modalities"}
            modalities: Final = response.get("output_modalities", response.get("modalities"))
            return (
                json.dumps(
                    {
                        **event,
                        "response": {
                            **normalized,
                            **({"modalities": _modalities(modalities)} if modalities is not None else {}),
                        },
                    }
                ),
            )
        return (message,)

    def _session_response(self, session: Mapping[str, JsonValue]) -> Mapping[str, object]:
        return {
            **{
                key: value
                for key, value in session.items()
                if key
                not in (
                    "modalities",
                    "voice",
                    "input_audio_format",
                    "output_audio_format",
                    "turn_detection",
                    "input_audio_transcription",
                    "tools",
                )
            },
            "type": "realtime",
            "modalities": session.get("modalities", ["text", "audio"]),
            "voice": session.get("voice"),
            "input_audio_format": "pcm",
            "output_audio_format": "pcm16",
            "turn_detection": session.get("turn_detection"),
            "input_audio_transcription": session.get("input_audio_transcription"),
            "output_modalities": ["audio"]
            if "audio" in _items(session.get("modalities", ["text", "audio"]), "modalities")
            else ["text"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 16000},
                    "turn_detection": session.get("turn_detection"),
                    "transcription": session.get("input_audio_transcription"),
                },
                "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": session.get("voice")},
            },
            **(
                {
                    "tools": tuple(
                        self._openai_tool(json_mapping(tool, "tool")) for tool in _items(session["tools"], "tools")
                    )
                }
                if "tools" in session
                else {}
            ),
        }

    @staticmethod
    def _openai_tool(tool: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
        if "function" not in tool:
            return tool
        return {"type": "function", **json_mapping(tool["function"], "function")}

    @staticmethod
    def _response_body(response: Mapping[str, JsonValue]) -> Mapping[str, object]:
        return {
            **response,
            **(
                {
                    "output_modalities": ["audio"]
                    if "audio" in _items(response["modalities"], "modalities")
                    else ["text"]
                }
                if "modalities" in response
                else {}
            ),
            **(
                {"audio": {"output": {"voice": response["voice"], "format": {"type": "audio/pcm", "rate": 24000}}}}
                if "voice" in response
                else {}
            ),
        }

    def transform_realtime_response(
        self,
        message: str | bytes,
        model: str,
        logging_obj: Logging,
        realtime_response_transform_input: RealtimeResponseTransformInput,
    ) -> RealtimeResponseTypedDict:
        event: Final = json_object(message.decode("utf-8") if isinstance(message, bytes) else message)
        event_type: Final = event.get("type")
        normalized: Final = _event_to_ga(event)
        result: Final = {
            **normalized,
            "type": _EVENT_TYPES.get(str(event_type), event_type),
            **(
                {"session": self._session_response(json_mapping(event.get("session"), "session"))}
                if event_type in ("session.created", "session.updated")
                else {}
            ),
            **self._transcription_delta(event),
            **(
                {"response": self._response_body(json_mapping(normalized.get("response"), "response"))}
                if "response" in normalized
                else {}
            ),
        }
        return {
            **realtime_response_transform_input,
            "response": cast(  # cast-ok: validated JSON retains provider extension events beyond the OpenAI union
                OpenAIRealtimeEvents, result
            ),
        }

    def _transcription_delta(self, event: Mapping[str, JsonValue]) -> Mapping[str, str]:
        event_type: Final = event.get("type")
        if event_type not in (
            "conversation.item.input_audio_transcription.delta",
            "conversation.item.input_audio_transcription.completed",
        ):
            return {}
        item_id: Final = event.get("item_id")
        content_index: Final = event.get("content_index", 0)
        if not isinstance(item_id, str) or not isinstance(content_index, int):
            raise RealtimeTranscriptionProtocolError("Transcription events require an item ID and content index")
        key: Final = (item_id, content_index)
        if event_type == "conversation.item.input_audio_transcription.completed":
            self._transcription_prefixes = MappingProxyType(
                {item: text for item, text in self._transcription_prefixes.items() if item != key}
            )
            return {}
        text: Final = event.get("text", "")
        if not isinstance(text, str):
            raise RealtimeTranscriptionProtocolError("Transcription text must be a string")
        previous: Final = self._transcription_prefixes.get(key, "")
        delta: Final = text[len(previous) :] if text.startswith(previous) else ""
        self._transcription_prefixes = MappingProxyType({**self._transcription_prefixes, key: text})
        return {"delta": delta}
