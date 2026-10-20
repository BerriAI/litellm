from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import anthropic
import openai
from pydantic import BaseModel

_SDK_PACKAGES: Final = frozenset({openai.__name__, anthropic.__name__})


def _subclasses(model: type[BaseModel]) -> Iterator[type[BaseModel]]:
    for subclass in model.__subclasses__():
        yield subclass
        yield from _subclasses(subclass)


def build_sdk_models() -> None:
    """Build every pydantic model the OpenAI and Anthropic SDKs define now rather than on its first use."""
    for model in frozenset(_subclasses(BaseModel)):
        if model.__module__.partition(".")[0] in _SDK_PACKAGES:
            _ = model.model_rebuild(raise_errors=False)
