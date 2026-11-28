from __future__ import annotations

import base64
import json
from typing import Final

import pytest

from litellm.interactions.id_utils import (
    INTERACTION_ID_PREFIX,
    decode_interaction_id,
    encode_interaction_id,
)


def _b64(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


@pytest.mark.parametrize(
    ("raw_id", "deployment_id"),
    [
        pytest.param("abc123", "split-west", id="simple"),
        pytest.param("v1_abc;def:ghi=jkl/mno+pqr", "dep;with:special/chars", id="special-chars"),
        pytest.param("int-987654321", "a" * 64, id="sha-style-deployment-id"),
    ],
)
def test_encode_decode_interaction_id_roundtrip(raw_id: str, deployment_id: str) -> None:
    encoded: Final = encode_interaction_id(raw_id, deployment_id)

    assert encoded.startswith(INTERACTION_ID_PREFIX)
    suffix: Final = encoded.removeprefix(INTERACTION_ID_PREFIX)
    assert "/" not in suffix
    assert "+" not in suffix
    assert "=" not in suffix

    decoded: Final = decode_interaction_id(encoded)
    assert decoded["interaction_id"] == raw_id
    assert decoded["deployment_id"] == deployment_id


def test_encode_interaction_id_is_idempotent_and_ignores_missing_deployment() -> None:
    encoded_once: Final = encode_interaction_id("raw-1", "dep-a")
    assert encode_interaction_id(encoded_once, "dep-a") == encoded_once
    assert encode_interaction_id(encoded_once, "dep-b") == encode_interaction_id("raw-1", "dep-b")
    assert encode_interaction_id("raw-1", None) == "raw-1"
    assert encode_interaction_id("raw-1", "") == "raw-1"
    assert encode_interaction_id("", "dep-a") == ""


@pytest.mark.parametrize(
    "bad_id",
    [
        "plain-interaction-id",
        f"{INTERACTION_ID_PREFIX}not!valid*base64",
        f"{INTERACTION_ID_PREFIX}{_b64(b'not-json')}",
        f"{INTERACTION_ID_PREFIX}{_b64(json.dumps({'v': 2, 'd': 'dep', 'i': 'id'}).encode())}",
        f"{INTERACTION_ID_PREFIX}{_b64(json.dumps({'v': 1, 'd': '', 'i': 'id'}).encode())}",
        f"{INTERACTION_ID_PREFIX}{_b64(json.dumps({'v': 1, 'd': 'dep', 'i': ''}).encode())}",
    ],
)
def test_decode_interaction_id_handles_invalid_or_unencoded_values(bad_id: str) -> None:
    decoded: Final = decode_interaction_id(bad_id)
    assert decoded["interaction_id"] == bad_id
    assert decoded["deployment_id"] is None
