"""Encoding and decoding helpers for proxy interaction routing IDs."""

from __future__ import annotations

import base64
import binascii
import json
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing_extensions import ReadOnly, TypedDict

INTERACTION_ID_PREFIX: Final = "litellm_int_"
_INTERACTION_ID_VERSION: Final = 1


class _InteractionIdPayload(BaseModel):
    model_config = ConfigDict(frozen=True)

    v: Literal[1]
    d: str = Field(min_length=1)
    i: str = Field(min_length=1)


class DecodedInteractionId(TypedDict):
    """Decoded interaction identifier paired with an optional creating deployment ID."""

    interaction_id: ReadOnly[str]
    deployment_id: ReadOnly[str | None]


def encode_interaction_id(interaction_id: str, deployment_id: str | None) -> str:
    """Wrap ``interaction_id`` with ``deployment_id`` for ID-only follow-up routing."""
    if not interaction_id or not deployment_id:
        return interaction_id
    raw_id: Final = decode_interaction_id(interaction_id)["interaction_id"]
    if not raw_id:
        return interaction_id
    payload: Final = json.dumps(
        {"v": _INTERACTION_ID_VERSION, "d": deployment_id, "i": raw_id},
        separators=(",", ":"),
    ).encode("utf-8")
    encoded: Final = base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")
    return f"{INTERACTION_ID_PREFIX}{encoded}"


def decode_interaction_id(interaction_id: str) -> DecodedInteractionId:
    """Extract the upstream interaction ID and optional deployment ID from ``interaction_id``."""
    if not interaction_id.startswith(INTERACTION_ID_PREFIX):
        return {"interaction_id": interaction_id, "deployment_id": None}
    encoded: Final = interaction_id.removeprefix(INTERACTION_ID_PREFIX)
    if not encoded:
        return {"interaction_id": interaction_id, "deployment_id": None}
    padded: Final = encoded + "=" * (-len(encoded) % 4)
    try:
        raw_bytes: Final = base64.urlsafe_b64decode(padded.encode("ascii"))
        parsed: Final = _InteractionIdPayload.model_validate_json(raw_bytes)
    except (ValueError, UnicodeDecodeError, binascii.Error, ValidationError):
        return {"interaction_id": interaction_id, "deployment_id": None}
    return {"interaction_id": parsed.i, "deployment_id": parsed.d}
