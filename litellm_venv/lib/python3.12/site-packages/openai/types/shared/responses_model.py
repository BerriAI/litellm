# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Union
from typing_extensions import Literal, TypeAlias

from .chat_model import ChatModel

__all__ = ["ResponsesModel"]

ResponsesModel: TypeAlias = Union[
    str,
    ChatModel,
    Literal[
        "o1-pro",
        "o1-pro-2025-03-19",
        "o3-pro",
        "o3-pro-2025-06-10",
        "o3-deep-research",
        "o3-deep-research-2025-06-26",
        "o4-mini-deep-research",
        "o4-mini-deep-research-2025-06-26",
        "computer-use-preview",
        "computer-use-preview-2025-03-11",
    ],
]
