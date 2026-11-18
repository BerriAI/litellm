from collections.abc import Mapping
from typing import Final

from fastapi import HTTPException

LEGACY_IDENTITY_MESSAGE: Final = (
    "litellm_params.identity is not supported: bind an Entra application through the top-level identity field"
)


def has_legacy_identity(params: Mapping[str, object] | None) -> bool:
    return params is not None and "identity" in params


def reject_legacy_identity(params: Mapping[str, object] | None) -> None:
    if has_legacy_identity(params):
        raise HTTPException(400, LEGACY_IDENTITY_MESSAGE)
