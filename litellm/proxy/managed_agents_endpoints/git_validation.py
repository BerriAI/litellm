"""Git repo + branch validation utilities for managed-agent template create."""

import base64
import os
import subprocess
import uuid
from typing import Any, Optional

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)


def authed_repo_url(repo_url: str, git_token: Optional[str]) -> str:
    """Return the URL unchanged. The git token is now injected via
    `http.extraheader` (passed through GIT_CONFIG_* env vars) rather than
    embedded in the URL netloc — embedding in argv leaks the token to
    `/proc/<PID>/cmdline` and `ps aux`."""
    return repo_url


def _git_auth_env(git_token: Optional[str]) -> dict:
    """Build the environment for a `git ls-remote` call.

    When a token is provided we set `http.extraheader = AUTHORIZATION: Basic
    <b64>` via GIT_CONFIG_COUNT/KEY_*/VALUE_*, which keeps the credential
    out of argv. Falls back to no auth when no token is provided.
    """
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    if not git_token:
        return env
    auth_value = base64.b64encode(f"x-access-token:{git_token}".encode("utf-8")).decode(
        "ascii"
    )
    env.update(
        {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "http.extraheader",
            "GIT_CONFIG_VALUE_0": f"Authorization: Basic {auth_value}",
        }
    )
    return env


def validate_repo_branch(
    repo_url: str, branch: str, git_token: Optional[str] = None
) -> None:
    env = _git_auth_env(git_token)
    try:
        result = subprocess.run(
            ["git", "ls-remote", "--heads", "--tags", repo_url, branch],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
            env=env,
        )
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=500, detail=f"git not installed on proxy host: {e}"
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=400, detail=f"timed out reaching {repo_url}")

    if result.returncode != 0:
        msg_lines = (result.stderr or result.stdout).strip().splitlines()
        tail = msg_lines[-1] if msg_lines else "unknown error"
        raise HTTPException(
            status_code=400,
            detail=f"git ls-remote failed for {repo_url}: {tail}",
        )
    if not result.stdout.strip():
        raise HTTPException(
            status_code=400,
            detail=f"branch or tag '{branch}' not found in {repo_url}",
        )


async def decrypt_git_token(
    prisma_client: Any, credential_id: Optional[str]
) -> Optional[str]:
    if credential_id is None:
        return None
    if prisma_client is None:
        return None
    row = await prisma_client.db.litellm_credentialstable.find_unique(
        where={"credential_id": credential_id}
    )
    if row is None:
        return None
    credential_values = getattr(row, "credential_values", None)
    if credential_values is None and isinstance(row, dict):
        credential_values = row.get("credential_values")
    if not isinstance(credential_values, dict):
        return None
    encrypted = credential_values.get("git_token")
    if not encrypted:
        return None
    return decrypt_value_helper(encrypted, key="git_token")


async def encrypt_and_store_git_token(
    prisma_client: Any, *, raw_token: str, created_by: str
) -> str:
    if prisma_client is None:
        raise HTTPException(status_code=500, detail="prisma client not available")

    credential_name = f"managed-agent-git-token-{uuid.uuid4()}"
    encrypted_token = encrypt_value_helper(raw_token)

    created = await prisma_client.db.litellm_credentialstable.create(
        data={
            "credential_name": credential_name,
            "credential_values": {"git_token": encrypted_token},
            "credential_info": {"source": "managed_agent_template"},
            "created_by": created_by,
            "updated_by": created_by,
        }
    )

    credential_id = getattr(created, "credential_id", None)
    if credential_id is None and isinstance(created, dict):
        credential_id = created.get("credential_id")
    if not credential_id:
        raise HTTPException(
            status_code=500, detail="failed to persist git token credential"
        )

    verbose_proxy_logger.debug(
        "Stored managed-agent git token credential credential_id=%s credential_name=%s",
        credential_id,
        credential_name,
    )
    return credential_id
