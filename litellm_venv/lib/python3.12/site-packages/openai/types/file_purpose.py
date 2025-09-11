# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing_extensions import Literal, TypeAlias

__all__ = ["FilePurpose"]

FilePurpose: TypeAlias = Literal[
    "assistants", "batch", "fine-tune", "vision", "user_data", "evals"
]
