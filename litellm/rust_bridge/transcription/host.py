from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Final, Protocol, cast  # noqa: TID251  # legacy mapper is dynamically typed

from pydantic import TypeAdapter

from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.rust_bridge.lifecycle import Complete
from litellm.types.utils import TranscriptionResponse
from litellm.utils import (
    get_optional_params_transcription,  # pyright: ignore[reportUnknownVariableType]  # legacy mapper is untyped
)

_OBJECT_MAPPING: Final = TypeAdapter(dict[str, object])
_OPTIONAL_FIELDS: Final = (
    "language",
    "prompt",
    "response_format",
    "temperature",
    "timestamp_granularities",
)
_HOST_ONLY_FIELDS: Final = frozenset(
    {
        "api_base",
        "api_key",
        "atranscription",
        "client",
        "custom_llm_provider",
        "extra_headers",
        "file",
        "litellm_call_id",
        "litellm_logging_obj",
        "model",
        "timeout",
        "user",
    }
)


class OptionalParamsMapper(Protocol):
    def __call__(self, *, model: str, custom_llm_provider: str, **kwargs: object) -> object: ...


class ModelDumper(Protocol):
    def model_dump(self) -> object: ...


def _mapping(value: object) -> dict[str, object]:
    return _OBJECT_MAPPING.validate_python(value)


def _provider(model: object, explicit: object) -> tuple[str, str]:
    if type(model) is not str:
        raise TypeError("model must be a string")
    if type(explicit) is str and explicit:
        return model.removeprefix(f"{explicit}/"), explicit
    prefix, separator, suffix = model.partition("/")
    return (suffix, prefix) if separator else (model, "")


def _audio(file: object) -> dict[str, object]:
    processed: Final = process_audio_file(file)  # pyright: ignore[reportArgumentType]  # public binding validates FileTypes
    formats: Final = {
        "audio/flac": "flac",
        "audio/mpeg": "mp3",
        "audio/mp3": "mp3",
        "audio/ogg": "ogg",
        "audio/wav": "wav",
        "audio/x-wav": "wav",
    }
    suffix: Final = processed.filename.rsplit(".", 1)[-1].lower() if "." in processed.filename else ""
    return {
        "data": base64.b64encode(processed.file_content).decode("ascii"),
        "format": formats.get(processed.content_type, suffix),
        "filename": processed.filename,
    }


def _project(request: Mapping[str, object], kwargs: Mapping[str, object]) -> dict[str, object]:
    merged: Final = {**request, **kwargs}
    model, provider = _provider(merged.get("model"), merged.get("custom_llm_provider"))
    mapper: Final = cast(OptionalParamsMapper, get_optional_params_transcription)
    optional_inputs: Final = {
        **{key: value for key, value in merged.items() if key not in _HOST_ONLY_FIELDS},
        **{name: merged.get(name) for name in _OPTIONAL_FIELDS},
    }
    optional_params: Final = _OBJECT_MAPPING.validate_python(
        mapper(model=model, custom_llm_provider=provider, **optional_inputs)
    )
    return {
        "model": model,
        "audio": _audio(merged.get("file")),
        "optional_params": optional_params,
        "api_key": merged.get("api_key"),
        "api_base": merged.get("api_base"),
        "custom_llm_provider": provider,
        "extra_headers": merged.get("extra_headers"),
        "timeout": merged.get("timeout"),
        "litellm_call_id": merged.get("litellm_call_id"),
    }


class TranscriptionLifecycleHost:
    def invoke(
        self,
        operation: str,
        payload: object,
        request: object,
        kwargs: dict[str, object],
        logger: object,
    ) -> Complete:
        if operation == "project":
            return Complete(_project(_mapping(request), kwargs))
        if operation in ("response", "cached_response"):
            return Complete(TranscriptionResponse(**_mapping(payload)))
        if operation == "cache_response":
            dumper: Final = cast(ModelDumper, payload)
            return Complete(dumper.model_dump() if isinstance(payload, TranscriptionResponse) else payload)
        if operation in ("before_request", "after_response", "map_failure"):
            return Complete(payload)
        if operation == "post_process":
            return Complete(None)
        raise ValueError(f"unknown transcription lifecycle operation: {operation}")


HOST: Final = TranscriptionLifecycleHost()
