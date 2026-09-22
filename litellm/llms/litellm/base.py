from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from types import MappingProxyType
from typing import Final, Protocol, TypeVar, cast  # noqa: TID251  # response stamping narrows runtime containers

from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper

LITELLM_MODEL_PREFIX: Final = "litellm/"
ResponseT = TypeVar("ResponseT")


class BaseLiteLLMModel(Protocol):
    async def acompletion(
        self,
        *,
        messages: list[AllMessageValues],  # mutable-ok: public SDK boundary
        stream: bool,
        request_kwargs: Mapping[str, object],
    ) -> ModelResponse | CustomStreamWrapper: ...


class LiteLLMModelFactory(Protocol):
    def __call__(self) -> BaseLiteLLMModel: ...


@lru_cache(maxsize=1)
def _model_factories() -> Mapping[str, LiteLLMModelFactory]:
    from litellm.llms.litellm.fusion import FusionLiteLLMModel

    return MappingProxyType({"litellm/fusion": FusionLiteLLMModel})


def is_litellm_model(model: str) -> bool:
    return model in _model_factories()


def get_litellm_model(model: str) -> BaseLiteLLMModel:
    factory: Final = _model_factories().get(model)
    if factory is None:
        raise ValueError(f"Unknown LiteLLM model {model!r}")
    return factory()


def stamp_litellm_model_response(
    response: ResponseT,
    model: str,
) -> ResponseT:
    if isinstance(response, dict):
        response_mapping: Final = cast(  # cast-ok: guarded by the dict check above
            dict[str, object], response
        )
        raw_hidden_params: Final = response_mapping.setdefault(
            "_hidden_params",
            {},  # mutable-ok: hidden SDK metadata is attached in place
        )
        if isinstance(raw_hidden_params, dict):
            raw_hidden_params["router"] = model
        return cast(ResponseT, response)  # cast-ok: the same generic response instance is returned
    hidden_params: Final = getattr(response, "_hidden_params", None)
    if isinstance(hidden_params, dict):
        hidden_params["router"] = model
    return response
