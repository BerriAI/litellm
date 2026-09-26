from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import pytest

from litellm.constants import CONTROL_OPTIONS_KEY
from litellm.types.litellm_params import ControlOptions

DEFAULT_CHUNKING_REQUESTS: Final = (
    pytest.param(MappingProxyType({}), id="unset"),
    pytest.param(MappingProxyType({"stream_chunk_size": "sixty-four", "drop_params": True}), id="dropped"),
    pytest.param(MappingProxyType({CONTROL_OPTIONS_KEY: ControlOptions(stream_chunk_size=1)}), id="forged_options"),
    pytest.param(MappingProxyType({CONTROL_OPTIONS_KEY: {"stream_chunk_size": 1}}), id="forged_mapping"),
)


def keys_at_every_depth(value: object) -> frozenset[str]:
    if isinstance(value, Mapping):
        return frozenset(value) | frozenset().union(*(keys_at_every_depth(item) for item in value.values()))
    if isinstance(value, (list, tuple)):
        return frozenset().union(*(keys_at_every_depth(item) for item in value))
    return frozenset()
