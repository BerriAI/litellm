import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy.management_endpoints.team_callback_endpoints import (
    add_team_callbacks,
    get_team_callbacks,
)
from litellm.proxy._types import AddTeamCallback, UserAPIKeyAuth


def _make_team(metadata: dict) -> MagicMock:
    team = MagicMock()
    team.metadata = metadata
    return team


def _make_request() -> MagicMock:
    req = MagicMock()
    req.headers = {}
    return req


# ---------------------------------------------------------------------------
# POST /team/{team_id}/callback — encrypt on write
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("litellm.proxy.proxy_server.prisma_client")
async def test_add_team_callback_encrypts_callback_vars(mock_prisma, monkeypatch):
    """callback_vars must be encrypted before writing to DB."""
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-key-1234567890123456")

    plaintext_secret = "sk-lf-supersecret99"
    existing_team = _make_team({"logging": []})
    mock_prisma.get_data = AsyncMock(return_value=existing_team)

    captured = {}

    async def fake_update(where, data):
        captured["data"] = data
        row = MagicMock()
        row.metadata = data.get("metadata", "{}")
        return row

    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_teamtable = MagicMock()
    mock_prisma.db.litellm_teamtable.update = fake_update

    data = AddTeamCallback(
        callback_name="langfuse",
        callback_type="success",
        callback_vars={"langfuse_secret_key": plaintext_secret},
    )

    await add_team_callbacks(
        data=data,
        http_request=_make_request(),
        team_id="team-123",
        user_api_key_dict=UserAPIKeyAuth(),
    )

    written_metadata = json.loads(captured["data"]["metadata"])
    stored_secret = written_metadata["logging"][0]["callback_vars"]["langfuse_secret_key"]
    assert stored_secret != plaintext_secret, "plaintext secret must be encrypted before DB write"


@pytest.mark.asyncio
@patch("litellm.proxy.proxy_server.prisma_client")
async def test_add_second_callback_does_not_double_encrypt_first(mock_prisma, monkeypatch):
    """Adding a second callback must not re-encrypt already-encrypted entries."""
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-key-1234567890123456")
    from litellm.proxy.common_utils.callback_utils import encrypt_logging_callback_vars
    from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper

    # Simulate a team whose first callback was already written (and encrypted)
    first_entry = {
        "callback_name": "langfuse",
        "callback_type": "success",
        "callback_vars": {"langfuse_secret_key": "sk-lf-firstsecret"},
    }
    encrypt_logging_callback_vars({"logging": [first_entry]})
    encrypted_first = first_entry["callback_vars"]["langfuse_secret_key"]

    existing_team = _make_team({"logging": [first_entry]})
    mock_prisma.get_data = AsyncMock(return_value=existing_team)

    captured = {}

    async def fake_update(where, data):
        captured["data"] = data
        row = MagicMock()
        row.metadata = data.get("metadata", "{}")
        return row

    mock_prisma.db = MagicMock()
    mock_prisma.db.litellm_teamtable = MagicMock()
    mock_prisma.db.litellm_teamtable.update = fake_update

    data = AddTeamCallback(
        callback_name="langsmith",
        callback_type="success",
        callback_vars={"langsmith_api_key": "ls-secondsecret"},
    )

    await add_team_callbacks(
        data=data,
        http_request=_make_request(),
        team_id="team-123",
        user_api_key_dict=UserAPIKeyAuth(),
    )

    written_metadata = json.loads(captured["data"]["metadata"])
    # First entry must be unchanged (not re-encrypted)
    assert written_metadata["logging"][0]["callback_vars"]["langfuse_secret_key"] == encrypted_first
    # First entry must still decrypt correctly
    recovered = decrypt_value_helper(value=encrypted_first, key="langfuse_secret_key", return_original_value=True)
    assert recovered == "sk-lf-firstsecret"
    # Second entry must be encrypted
    stored_second = written_metadata["logging"][1]["callback_vars"]["langsmith_api_key"]
    assert stored_second != "ls-secondsecret"


# ---------------------------------------------------------------------------
# GET /team/{team_id}/callback — redact on read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("litellm.proxy.proxy_server.prisma_client")
async def test_get_team_callbacks_redacts_callback_vars(mock_prisma, monkeypatch):
    """callback_vars must be redacted (last-3-chars) in the GET response."""
    monkeypatch.setenv("LITELLM_SALT_KEY", "test-salt-key-1234567890123456")

    existing_team = _make_team(
        {
            "callback_settings": {
                "success_callback": ["langfuse"],
                "callback_vars": {"langfuse_secret_key": "sk-lf-supersecret99"},
            }
        }
    )
    mock_prisma.get_data = AsyncMock(return_value=existing_team)

    result = await get_team_callbacks(
        http_request=_make_request(),
        team_id="team-123",
        user_api_key_dict=UserAPIKeyAuth(),
    )

    secret = result["data"]["callback_vars"]["langfuse_secret_key"]
    assert secret != "sk-lf-supersecret99", "plaintext secret must not be returned"
    assert secret.startswith("..."), "redacted value must use ...XYZ format"


@pytest.mark.asyncio
@patch("litellm.proxy.proxy_server.prisma_client")
async def test_get_team_callbacks_keeps_env_var_pointers(mock_prisma):
    """os.environ/ references must pass through unredacted."""
    existing_team = _make_team(
        {
            "callback_settings": {
                "callback_vars": {
                    "langfuse_secret_key": "os.environ/LANGFUSE_SECRET_KEY"
                },
            }
        }
    )
    mock_prisma.get_data = AsyncMock(return_value=existing_team)

    result = await get_team_callbacks(
        http_request=_make_request(),
        team_id="team-123",
        user_api_key_dict=UserAPIKeyAuth(),
    )

    assert (
        result["data"]["callback_vars"]["langfuse_secret_key"]
        == "os.environ/LANGFUSE_SECRET_KEY"
    )
