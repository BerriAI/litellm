# pyright: reportAny=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Final

from litellm.proxy._types import UserAPIKeyAuth
from litellm.python_extension.generated.v1 import extension_host_pb2 as pb
from litellm.types.utils import ModelResponse, ModelResponseStream


def auth_from_proto(auth: pb.AuthContext) -> UserAPIKeyAuth:
    metadata: Final = dict(auth.request_metadata)  # mutable-ok: LiteLLM compatibility payload
    return UserAPIKeyAuth(
        token=auth.key_hash or None,
        user_id=auth.user_id or None,
        team_id=auth.team_id or None,
        metadata=metadata,
    )


def decode_json(raw: bytes, name: str) -> object:
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid JSON: {error}") from error


def decode_json_object(raw: bytes, name: str) -> dict[str, object]:  # mutable-ok: LiteLLM compatibility payload
    value = decode_json(raw, name)  # rebind-ok: invocation-scoped RPC state
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain a JSON object")
    return value


def response_from_json(raw: bytes) -> object:
    value = decode_json(raw, "response_json")  # rebind-ok: invocation-scoped RPC state
    if not isinstance(value, dict):
        return value
    object_type: Final = value.get("object")
    try:
        if object_type == "chat.completion.chunk":
            return ModelResponseStream(**value)
        if object_type == "chat.completion" or "choices" in value:
            return ModelResponse(**value)
    except (TypeError, ValueError):
        pass
    return value


def encode_json(value: object) -> bytes:
    model_dump: Final = getattr(value, "model_dump", None)
    if callable(model_dump):
        value = model_dump(mode="json")  # rebind-ok: invocation-scoped RPC state
    elif isinstance(value, Mapping):
        value = dict(value)  # mutable-ok: LiteLLM compatibility payload # rebind-ok: invocation-scoped RPC state
    return json.dumps(value, separators=(",", ":"), sort_keys=True, default=str).encode()
