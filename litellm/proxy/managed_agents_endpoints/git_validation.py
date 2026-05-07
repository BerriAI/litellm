"""Git repo + branch validation utilities for managed-agent template create."""

import os
import subprocess
import uuid
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)


def authed_repo_url(repo_url: str, git_token: Optional[str]) -> str:
    if not git_token:
        return repo_url
    parsed = urlparse(repo_url)
    if parsed.scheme != "https":
        return repo_url
    if not parsed.hostname:
        return repo_url
    netloc = f"x-access-token:{git_token}@{parsed.hostname}"
    if parsed.port:
        netloc += f":{parsed.port}"
    return urlunparse(
        (
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


def validate_repo_branch(
    repo_url: str, branch: str, git_token: Optional[str] = None
) -> None:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    url = authed_repo_url(repo_url, git_token)
    try:
        result = subprocess.run(
            ["git", "ls-remote", "--heads", "--tags", url, branch],
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
        # Scrub authed URL so token is not echoed back to clients/logs.
        tail = tail.replace(url, repo_url)
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
