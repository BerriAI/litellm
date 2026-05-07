from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException


async def harness_create_session(
    sandbox_url: str,
    client: httpx.AsyncClient,
    *,
    title: str = "default",
    timeout: int = 30,
) -> str:
    """POST {sandbox_url}/session w/ {"title": ...}. Returns session id.
    Handle response shape: bare object OR single-element array (proto comment)."""
    r = await client.post(
        f"{sandbox_url}/session",
        json={"title": title},
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    if isinstance(data, list):
        if not data:
            raise RuntimeError(f"unexpected harness session response: {data}")
        data = data[0]
    if not isinstance(data, dict) or "id" not in data:
        raise RuntimeError(f"unexpected harness session response: {data}")
    return data["id"]


async def harness_send_message(
    sandbox_url: str,
    harness_session_id: str,
    client: httpx.AsyncClient,
    *,
    model: str,
    parts: List[Dict[str, Any]],
    timeout: int = 240,
) -> Dict[str, Any]:
    """POST {sandbox_url}/session/{id}/message w/ {model:{providerID:'litellm',modelID:model}, parts:...}.
    Returns full response JSON."""
    body = {
        "model": {"providerID": "litellm", "modelID": model},
        "parts": parts,
    }
    r = await client.post(
        f"{sandbox_url}/session/{harness_session_id}/message",
        json=body,
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def expand_message(
    text: Optional[str],
    parts: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Coerce {text} or {parts} into harness parts list. Raises HTTPException(400) if both missing."""
    if parts is not None:
        return parts
    if text is not None:
        return [{"type": "text", "text": text}]
    raise HTTPException(400, "message body must include 'text' or 'parts'")
