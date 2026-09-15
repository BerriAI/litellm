from __future__ import annotations

from collections.abc import Mapping
from typing import Final, Protocol, cast  # noqa: TID251  # legacy callables are validated at the boundary

from pydantic import TypeAdapter

from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,  # pyright: ignore[reportUnknownVariableType]  # legacy converter lacks complete annotations
)
from litellm.rust_bridge.lifecycle import Complete
from litellm.types.utils import ModelResponse

RUST_RESPONSE_HEADER: Final = "x-litellm-rust"
_OBJECT_MAPPING: Final = TypeAdapter(dict[str, object])
_HOST_ONLY_FIELDS: Final = frozenset(
    {
        "acompletion",
        "api_base",
        "api_key",
        "base_url",
        "client",
        "custom_llm_provider",
        "host_facts",
        "litellm_call_id",
        "litellm_logging_obj",
        "logger_fn",
        "model",
        "model_list",
        "optional_params",
        "shared_session",
        "timeout",
    }
)


class OptionalParamsMapper(Protocol):
    def __call__(self, *, model: str, custom_llm_provider: str, **kwargs: object) -> object: ...


class ModelDumper(Protocol):
    def model_dump(self) -> object: ...


def _mapping(value: object) -> dict[str, object]:
    return _OBJECT_MAPPING.validate_python(value)


def _response(value: object) -> ModelResponse:
    built: Final = convert_to_model_response_object(
        response_object=dict(_mapping(value)),  # mutable-ok: the converter takes a real dict and rewrites it
        model_response_object=ModelResponse(),
        hidden_params={"additional_headers": {RUST_RESPONSE_HEADER: "true"}},  # mutable-ok: converter rewrites it
    )
    if not isinstance(built, ModelResponse):
        raise TypeError(f"expected a ModelResponse from the rust path, got {type(built).__name__}")
    return built


def _provider(model: object, explicit: object) -> tuple[str, str | None]:
    if not isinstance(model, str):
        raise TypeError("model must be a string")
    if isinstance(explicit, str) and explicit:
        return model.removeprefix(f"{explicit}/"), explicit
    prefix, separator, suffix = model.partition("/")
    if separator and prefix in ("anthropic", "bedrock"):
        return suffix, prefix
    return model, None


def _project(request: Mapping[str, object], kwargs: Mapping[str, object]) -> dict[str, object]:
    from litellm.utils import (
        get_optional_params,  # pyright: ignore[reportUnknownVariableType]  # legacy mapper is untyped
    )

    merged: Final = {**request, **kwargs}
    model, provider = _provider(merged.get("model"), merged.get("custom_llm_provider"))
    mapping_args: Final = {key: value for key, value in merged.items() if key not in _HOST_ONLY_FIELDS}
    mapper: Final = cast(OptionalParamsMapper, get_optional_params)
    optional_params: Final = _OBJECT_MAPPING.validate_python(
        mapper(model=model, custom_llm_provider=provider or "", **mapping_args)
    )
    return {
        "model": model,
        "messages": merged.get("messages", []),
        "optional_params": optional_params,
        "api_key": merged.get("api_key"),
        "api_base": merged.get("api_base") or merged.get("base_url"),
        "custom_llm_provider": provider,
        "extra_headers": merged.get("extra_headers"),
        "timeout": merged.get("timeout"),
        "litellm_call_id": merged.get("litellm_call_id"),
    }


class ChatLifecycleHost:
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
            return Complete(_response(payload))
        if operation == "cache_response":
            dumper: Final = cast(ModelDumper, payload)
            value: Final = dumper.model_dump() if isinstance(payload, ModelResponse) else payload
            return Complete(value)
        if operation in ("before_request", "after_response"):
            return Complete(payload)
        if operation == "map_failure":
            return Complete(payload)
        if operation == "post_process":
            return Complete(None)
        raise ValueError(f"unknown chat lifecycle operation: {operation}")


HOST: Final = ChatLifecycleHost()
