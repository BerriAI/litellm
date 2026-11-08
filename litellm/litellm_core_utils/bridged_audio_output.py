from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, NoReturn, TypeAlias

from pydantic import TypeAdapter, ValidationError
from typing_extensions import assert_never

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.core_helpers import drops_unsupported_params

BridgedEndpoint: TypeAlias = Literal["/v1/responses", "/v1/messages", "/v1/completions"]


_STRINGS: Final = TypeAdapter(tuple[str, ...])
_AUDIO_PARAMS: Final = ("audio", "modalities")


@dataclass(frozen=True, slots=True)
class AudioOnlyModel:
    model: str
    custom_llm_provider: str | None


@dataclass(frozen=True, slots=True)
class AudioRequested:
    model: str
    custom_llm_provider: str | None
    params: tuple[str, ...]


AudioOutputRefusal: TypeAlias = AudioOnlyModel | AudioRequested


def _strings(value: object) -> tuple[str, ...]:
    try:
        return _STRINGS.validate_python(value)
    except ValidationError:
        return ()


def _requested_audio_params(request: Mapping[str, object]) -> tuple[str, ...]:
    requested: Final = (
        ("audio", request.get("audio") is not None),
        ("modalities", "audio" in _strings(request.get("modalities"))),
    )
    explicitly_dropped: Final = frozenset(_strings(request.get("additional_drop_params")))
    return tuple(name for name, asks_for_audio in requested if asks_for_audio and name not in explicitly_dropped)


def _without_audio_output(request: Mapping[str, object]) -> Mapping[str, object]:
    modalities: Final = _strings(request.get("modalities"))
    if "audio" not in modalities:
        return MappingProxyType({key: value for key, value in request.items() if key != "audio"})
    kept: Final = {key: value for key, value in request.items() if key not in _AUDIO_PARAMS}
    other_modalities: Final = [modality for modality in modalities if modality != "audio"]
    return MappingProxyType({**kept, "modalities": other_modalities} if other_modalities else kept)


def bridged_audio_output_request(
    request: Mapping[str, object], model: str, custom_llm_provider: str | None
) -> Mapping[str, object] | AudioOutputRefusal:
    display_name: Final = "/".join(part for part in (custom_llm_provider, model) if part)
    if litellm.utils.is_gemini_tts_model(model):
        return AudioOnlyModel(model=display_name, custom_llm_provider=custom_llm_provider)
    params: Final = _requested_audio_params(request)
    if not params:
        return request
    if not drops_unsupported_params(request.get("drop_params")):
        return AudioRequested(model=display_name, custom_llm_provider=custom_llm_provider, params=params)
    return _without_audio_output(request)


def _refusal_message(refusal: AudioOutputRefusal, endpoint: BridgedEndpoint) -> str:
    match refusal:
        case AudioOnlyModel():
            return (
                f"{refusal.model} returns audio, which {endpoint} cannot carry. "
                "Call /v1/chat/completions to get the audio."
            )
        case AudioRequested():
            return (
                f"The request asks {refusal.model} for audio through {' and '.join(refusal.params)}, "
                f"which {endpoint} cannot carry. Leave out {' and '.join(refusal.params)}, "
                "or set drop_params: true to send the request without it. "
                f"If {refusal.model} can return audio, call /v1/chat/completions to get it."
            )
        case _:
            assert_never(refusal)


def raise_audio_output_refusal(refusal: AudioOutputRefusal, endpoint: BridgedEndpoint) -> NoReturn:
    raise UnsupportedParamsError(
        message=_refusal_message(refusal, endpoint),
        llm_provider=refusal.custom_llm_provider,
        model=refusal.model,
    )
