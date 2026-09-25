from collections.abc import Mapping
from typing import Final

import litellm
import pytest
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.litellm_params import CONTROL_PARAMS_KEY, LiteLLMControlParams


class LitellmParamsRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.seen: tuple[Mapping[str, object], ...] = ()

    def log_pre_api_call(self, model: str, messages: object, kwargs: Mapping[str, object]) -> None:
        params: Final = kwargs["litellm_params"]
        assert isinstance(params, Mapping)
        self.seen = (*self.seen, params)


def record_litellm_params(monkeypatch: pytest.MonkeyPatch) -> LitellmParamsRecorder:
    recorder: Final = LitellmParamsRecorder()
    monkeypatch.setattr(litellm, "input_callback", [recorder])
    return recorder


def keys_at_every_depth(value: object) -> frozenset[str]:
    if isinstance(value, Mapping):
        return frozenset(value) | frozenset().union(*(keys_at_every_depth(item) for item in value.values()))
    if isinstance(value, (list, tuple)):
        return frozenset().union(*(keys_at_every_depth(item) for item in value))
    return frozenset()


def recorded_stream_chunk_size(params: Mapping[str, object]) -> int | None:
    control: Final = params[CONTROL_PARAMS_KEY]
    assert isinstance(control, LiteLLMControlParams)
    return control.stream_chunk_size
