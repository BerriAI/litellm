from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, NoReturn, TypeAlias

from pydantic import ConfigDict, TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.tool_fields import drop_policy
from litellm.utils import is_gemini_tts_model


@dataclass(frozen=True, slots=True)
class AudioOnlyModel:
    model: str


@dataclass(frozen=True, slots=True)
class AudioOutputRequested:
    fields: tuple[str, ...]


AudioOutputRefusal: TypeAlias = AudioOnlyModel | AudioOutputRequested

_MODALITY_LIST: Final = TypeAdapter(list[object], config=ConfigDict(strict=True))


def _listed_modalities(value: object) -> tuple[object, ...] | None:
    try:
        return tuple(_MODALITY_LIST.validate_python(value))
    except ValidationError:
        return None


def _asks_for_audio_modality(value: object) -> bool:
    return "audio" in (_listed_modalities(value) or ())


_AUDIO_OUTPUT_REQUESTS: Final[Mapping[str, Callable[[object], bool]]] = MappingProxyType(
    {"audio": lambda value: value is not None, "modalities": _asks_for_audio_modality}
)


def _forwarded_modalities(value: object) -> object:
    listed: Final = _listed_modalities(value)
    if listed is None:
        return value
    return [modality for modality in listed if modality != "audio"] or None


def _without_audio_output(params: Mapping[str, object]) -> Mapping[str, object]:
    modalities: Final = _forwarded_modalities(params.get("modalities"))
    without_audio: Final = {key: value for key, value in params.items() if key not in _AUDIO_OUTPUT_REQUESTS}
    return MappingProxyType(without_audio if modalities is None else {**without_audio, "modalities": modalities})


def params_without_audio_output(params: Mapping[str, object], model: str) -> Mapping[str, object] | AudioOutputRefusal:
    if is_gemini_tts_model(model):
        return AudioOnlyModel(model)
    policy: Final = drop_policy(params)
    requested: Final = tuple(
        field
        for field, requests_audio in _AUDIO_OUTPUT_REQUESTS.items()
        if field not in policy.additional_drop_params and requests_audio(params.get(field))
    )
    if not requested:
        return params
    if not policy.drop_params:
        return AudioOutputRequested(requested)
    return _without_audio_output(params)


def _refusal_message(refusal: AudioOutputRefusal, *, endpoint: str, model: str) -> str:
    match refusal:
        case AudioOnlyModel():
            return (
                f"{refusal.model} only answers with audio, which {endpoint} cannot return. "
                "Call /v1/audio/speech or /v1/chat/completions instead."
            )
        case AudioOutputRequested(fields=fields):
            return (
                f"{endpoint} cannot return audio output, so it does not support {list(fields)} for model={model}. "
                "Call /v1/chat/completions for audio, or set drop_params: true to drop these params."
            )
        case _:
            assert_never(refusal)


def raise_audio_output_refusal(
    refusal: AudioOutputRefusal, *, endpoint: str, model: str, custom_llm_provider: str | None
) -> NoReturn:
    raise UnsupportedParamsError(
        status_code=400,
        model=model,
        llm_provider=custom_llm_provider,
        message=_refusal_message(refusal, endpoint=endpoint, model=model),
    )


def forward_without_audio_output(
    params: Mapping[str, object], *, endpoint: str, model: str, custom_llm_provider: str | None
) -> Mapping[str, object]:
    forwarded: Final = params_without_audio_output(params, model)
    if isinstance(forwarded, AudioOnlyModel | AudioOutputRequested):
        raise_audio_output_refusal(forwarded, endpoint=endpoint, model=model, custom_llm_provider=custom_llm_provider)
    return forwarded
