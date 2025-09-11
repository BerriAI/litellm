# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import List, Optional
from typing_extensions import TypedDict

__all__ = ["BetaRequestMCPServerToolConfigurationParam"]


class BetaRequestMCPServerToolConfigurationParam(TypedDict, total=False):
    allowed_tools: Optional[List[str]]

    enabled: Optional[bool]
