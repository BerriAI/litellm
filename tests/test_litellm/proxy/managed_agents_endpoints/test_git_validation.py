"""Unit tests for git_validation module.

Mocks subprocess + prisma + encrypt/decrypt helpers. Verifies:
  - authed_repo_url: HTTPS w/ token, HTTPS no token, ssh, malformed
  - validate_repo_branch: success, empty stdout, nonzero exit, FileNotFoundError, TimeoutExpired
  - decrypt_git_token: None id, found w/ token, not found, missing key
  - encrypt_and_store_git_token: happy path, credential_name shape, created_by/updated_by
"""

import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from litellm.proxy.managed_agents_endpoints.git_validation import (
    authed_repo_url,
    decrypt_git_token,
    encrypt_and_store_git_token,
    validate_repo_branch,
)


# ---------------------------------------------------------------------------
# authed_repo_url
# ---------------------------------------------------------------------------


def test_authed_repo_url_returns_input_unchanged():
    """authed_repo_url no longer rewrites the URL — auth is now passed via
    GIT_CONFIG_VALUE_0 env var to keep the token out of argv."""
    original = "https://github.com/org/repo.git"
    assert authed_repo_url(original, "secret-token") == original
    assert authed_repo_url(original, None) == original
    assert authed_repo_url(original, "") == original
    assert authed_repo_url("git@github.com:org/repo.git", "secret-token") == (
        "git@github.com:org/repo.git"
    )


# ---------------------------------------------------------------------------
# validate_repo_branch (sync, raises HTTPException on failure)
# ---------------------------------------------------------------------------


def _completed(returncode: int, stdout: str = "", stderr: str = ""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


def test_validate_repo_branch_success_no_exception():
    with patch(
        "litellm.proxy.managed_agents_endpoints.git_validation.subprocess.run",
        return_value=_completed(0, stdout="abc123\trefs/heads/main\n", stderr=""),
    ) as run_mock:
        # Should not raise.
        validate_repo_branch("https://github.com/org/repo.git", "main", git_token=None)
    run_mock.assert_called_once()


def test_validate_repo_branch_empty_stdout_raises_400_branch_not_found():
    with patch(
        "litellm.proxy.managed_agents_endpoints.git_validation.subprocess.run",
        return_value=_completed(0, stdout="", stderr=""),
    ):
        with pytest.raises(HTTPException) as exc_info:
            validate_repo_branch(
                "https://github.com/org/repo.git", "missing-branch", git_token=None
            )
    assert exc_info.value.status_code == 400
    assert "missing-branch" in exc_info.value.detail
    assert "not found" in exc_info.value.detail


def test_validate_repo_branch_token_not_in_argv():
    """The git token must not appear as a subprocess argument — it should
    flow through the GIT_CONFIG_* env vars instead."""
    repo_url = "https://github.com/org/repo.git"
    token = "super-secret-token"
    with patch(
        "litellm.proxy.managed_agents_endpoints.git_validation.subprocess.run",
        return_value=_completed(0, stdout="abc\trefs/heads/main\n"),
    ) as run_mock:
        validate_repo_branch(repo_url, "main", git_token=token)

    args, kwargs = run_mock.call_args
    cmd = args[0]
    assert token not in " ".join(cmd)
    env = kwargs["env"]
    assert env["GIT_CONFIG_COUNT"] == "1"
    assert env["GIT_CONFIG_KEY_0"] == "http.extraheader"
    # value is `Authorization: Basic <b64(x-access-token:<token>)>`
    assert env["GIT_CONFIG_VALUE_0"].startswith("Authorization: Basic ")
    assert token not in env["GIT_CONFIG_VALUE_0"]


def test_validate_repo_branch_file_not_found_raises_500():
    with patch(
        "litellm.proxy.managed_agents_endpoints.git_validation.subprocess.run",
        side_effect=FileNotFoundError("git not on PATH"),
    ):
        with pytest.raises(HTTPException) as exc_info:
            validate_repo_branch(
                "https://github.com/org/repo.git", "main", git_token=None
            )
    assert exc_info.value.status_code == 500
    assert "git not installed" in exc_info.value.detail


def test_validate_repo_branch_timeout_raises_400():
    with patch(
        "litellm.proxy.managed_agents_endpoints.git_validation.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="git", timeout=15),
    ):
        with pytest.raises(HTTPException) as exc_info:
            validate_repo_branch(
                "https://github.com/org/repo.git", "main", git_token=None
            )
    assert exc_info.value.status_code == 400
    assert "timed out" in exc_info.value.detail


# ---------------------------------------------------------------------------
# decrypt_git_token (async)
# ---------------------------------------------------------------------------


def _fake_prisma_credentials(row):
    """Build a fake prisma client whose litellm_credentialstable.find_unique returns row."""
    table = MagicMock()
    table.find_unique = AsyncMock(return_value=row)
    db = MagicMock()
    db.litellm_credentialstable = table
    client = MagicMock()
    client.db = db
    return client, table


@pytest.mark.asyncio
async def test_decrypt_git_token_none_id_returns_none_no_db_call():
    client, table = _fake_prisma_credentials(row=None)
    result = await decrypt_git_token(client, credential_id=None)
    assert result is None
    table.find_unique.assert_not_called()


@pytest.mark.asyncio
async def test_decrypt_git_token_found_returns_plaintext():
    row = SimpleNamespace(
        credential_id="cred-1",
        credential_values={"git_token": "ENCRYPTED-BLOB"},
    )
    client, table = _fake_prisma_credentials(row=row)
    with patch(
        "litellm.proxy.managed_agents_endpoints.git_validation.decrypt_value_helper",
        return_value="plaintext-token",
    ) as decrypt_mock:
        result = await decrypt_git_token(client, credential_id="cred-1")

    assert result == "plaintext-token"
    decrypt_mock.assert_called_once_with("ENCRYPTED-BLOB", key="git_token")
    table.find_unique.assert_awaited_once_with(where={"credential_id": "cred-1"})


@pytest.mark.asyncio
async def test_decrypt_git_token_not_found_returns_none():
    client, table = _fake_prisma_credentials(row=None)
    with patch(
        "litellm.proxy.managed_agents_endpoints.git_validation.decrypt_value_helper"
    ) as decrypt_mock:
        result = await decrypt_git_token(client, credential_id="missing")

    assert result is None
    decrypt_mock.assert_not_called()


@pytest.mark.asyncio
async def test_decrypt_git_token_missing_git_token_key_returns_none():
    row = SimpleNamespace(
        credential_id="cred-1",
        credential_values={"some_other_key": "blob"},
    )
    client, _ = _fake_prisma_credentials(row=row)
    with patch(
        "litellm.proxy.managed_agents_endpoints.git_validation.decrypt_value_helper"
    ) as decrypt_mock:
        result = await decrypt_git_token(client, credential_id="cred-1")

    assert result is None
    decrypt_mock.assert_not_called()


# ---------------------------------------------------------------------------
# encrypt_and_store_git_token (async)
# ---------------------------------------------------------------------------


def _fake_prisma_credentials_create(returned):
    table = MagicMock()
    table.create = AsyncMock(return_value=returned)
    db = MagicMock()
    db.litellm_credentialstable = table
    client = MagicMock()
    client.db = db
    return client, table


@pytest.mark.asyncio
async def test_encrypt_and_store_git_token_happy_path():
    created = SimpleNamespace(credential_id="cred-new-123")
    client, table = _fake_prisma_credentials_create(returned=created)

    with patch(
        "litellm.proxy.managed_agents_endpoints.git_validation.encrypt_value_helper",
        return_value="ENCRYPTED-OUTPUT",
    ) as encrypt_mock:
        new_id = await encrypt_and_store_git_token(
            client, raw_token="raw-secret", created_by="user-alice"
        )

    assert new_id == "cred-new-123"
    encrypt_mock.assert_called_once_with("raw-secret")

    table.create.assert_awaited_once()
    create_kwargs = table.create.call_args.kwargs
    data = create_kwargs["data"]

    # Encrypted value is what the create was called with.
    assert data["credential_values"] == {"git_token": "ENCRYPTED-OUTPUT"}

    # credential_name format: managed-agent-git-token-<uuid4>
    assert re.match(r"^managed-agent-git-token-[0-9a-f-]+$", data["credential_name"])

    # created_by + updated_by both set to caller-supplied value.
    assert data["created_by"] == "user-alice"
    assert data["updated_by"] == "user-alice"
