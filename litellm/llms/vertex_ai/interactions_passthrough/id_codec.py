from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Optional

_PREFIX = "litellm_proxy"
_DISCRIMINATOR = "vertex_interaction"
_HEAD = f"{_PREFIX}:{_DISCRIMINATOR};"


@dataclass(frozen=True, slots=True)
class VertexInteractionId:
    project: str
    location: str
    raw_id: str


def encode(project: str, location: str, raw_id: str) -> str:
    plaintext = f"{_HEAD}project,{project};location,{location};raw_id,{raw_id}"
    return base64.urlsafe_b64encode(plaintext.encode()).decode().rstrip("=")


def decode(value: str) -> Optional[VertexInteractionId]:
    if not isinstance(value, str) or not value:
        return None
    padded = value + "=" * (-len(value) % 4)
    try:
        plaintext = base64.urlsafe_b64decode(padded).decode()
    except (ValueError, UnicodeDecodeError):
        return None
    if not plaintext.startswith(_HEAD):
        return None
    rest = plaintext[len(_HEAD) :]
    try:
        project_part, rest2 = rest.split(";", 1)
        location_part, raw_id_part = rest2.split(";", 1)
    except ValueError:
        return None
    if not (
        project_part.startswith("project,")
        and location_part.startswith("location,")
        and raw_id_part.startswith("raw_id,")
    ):
        return None
    return VertexInteractionId(
        project=project_part[len("project,") :],
        location=location_part[len("location,") :],
        raw_id=raw_id_part[len("raw_id,") :],
    )


def is_encoded(value: str) -> bool:
    return decode(value) is not None
