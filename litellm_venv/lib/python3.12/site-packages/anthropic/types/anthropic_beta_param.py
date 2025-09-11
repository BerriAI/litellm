# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union
from typing_extensions import Literal, TypeAlias

__all__ = ["AnthropicBetaParam"]

AnthropicBetaParam: TypeAlias = Union[
    str,
    Literal[
        "message-batches-2024-09-24",
        "prompt-caching-2024-07-31",
        "computer-use-2024-10-22",
        "computer-use-2025-01-24",
        "pdfs-2024-09-25",
        "token-counting-2024-11-01",
        "token-efficient-tools-2025-02-19",
        "output-128k-2025-02-19",
        "files-api-2025-04-14",
        "mcp-client-2025-04-04",
        "dev-full-thinking-2025-05-14",
        "interleaved-thinking-2025-05-14",
        "code-execution-2025-05-22",
        "extended-cache-ttl-2025-04-11",
    ],
]
