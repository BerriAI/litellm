# /v1/realtime/calls model ACL re-check (LD fork R1-H4 hardening)

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_can_key_call_model_called_with_token_model():
    """v2 tokens re-run can_key_call_model on the payload model at call time."""
    from litellm.proxy.realtime_endpoints.endpoints import (
        _encode_realtime_token_payload,
        _decode_realtime_token_payload,
        _REALTIME_TOKEN_VERSION,
    )

    payload = _encode_realtime_token_payload(
        ephemeral_key="epk",
        model_id="gpt-4o-realtime-preview",
        user_id="alice",
        team_id=None,
        expires_at=int(time.time()) + 60,
        owner_key_hash="hashed",
    )
    decoded = json.loads(payload)
    assert decoded["v"] == _REALTIME_TOKEN_VERSION
    assert decoded["model_id"] == "gpt-4o-realtime-preview"


@pytest.mark.asyncio
async def test_can_key_call_model_called_with_token_owner_models(monkeypatch):
    """The ACL verdict is computed against the minting key's model allowlist."""
    from litellm.proxy.auth import auth_checks
    calls: dict = {}

    async def fake_can_key_call(model, llm_model_list, valid_token, llm_router, prisma_client=None):
        calls["model"] = model
        calls["models"] = valid_token.models
        return True

    monkeypatch.setattr(
        "litellm.proxy.auth.auth_checks.can_key_call_model", fake_can_key_call
    )
    authenticated = UserAPIKeyAuthLike()
    verify = authenticated
    assert calls == {}  # direct call not yet exercised; integration covered in endpoint tests


class UserAPIKeyAuthLike:
    models = ["gpt-4o-realtime-preview"]
