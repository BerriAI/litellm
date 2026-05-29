"""
LiteLLM customer support drafting agent endpoint (v1).

Exposes a single HTTP endpoint that accepts a customer question (+ optional
context and segment) and returns a drafted reply produced by a Cursor Cloud
Agent running with this repo + BerriAI/litellm-docs as a multi-repo
environment. The Cursor Agent applies the support rule and skill defined in
this repository.

Run:
    uv run python support/customer_support_agent.py

Required environment variables:
    CURSOR_API_KEY              Cursor Cloud Agents API key (Basic auth).
                                Or set LITELLM_PROXY_URL + LITELLM_PROXY_API_KEY
                                to route through a litellm proxy (which has a
                                built-in /cursor pass-through).
    SUPPORT_AGENT_REPO          GitHub URL for the litellm repo the cloud agent
                                should clone (e.g. https://github.com/BerriAI/litellm).
                                Defaults to https://github.com/BerriAI/litellm.
    SUPPORT_AGENT_REF           Git ref/branch to run on. Defaults to "main".

Optional:
    SUPPORT_AGENT_MODEL         Cursor model id (omit for API default).
    SUPPORT_AGENT_POLL_INTERVAL Seconds between status polls (default 5).
    SUPPORT_AGENT_TIMEOUT       Seconds before giving up polling (default 600).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
from typing import Any, Dict, Literal, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("litellm.support_agent")
logging.basicConfig(level=logging.INFO)


CustomerSegment = Literal["paying", "prospect", "oss"]


class DraftReplyRequest(BaseModel):
    question: str = Field(..., description="The customer question or issue.")
    context: Optional[str] = Field(
        None,
        description="Optional context: logs, config snippets, version, provider, etc.",
    )
    customer_segment: CustomerSegment = Field(
        "paying",
        description="Customer segment. Defaults to 'paying' (LiteLLM Enterprise).",
    )
    tone_override: Optional[str] = Field(
        None,
        description="Optional tone nudge for the reviewer (e.g. 'customer is frustrated').",
    )


class DraftReplyResponse(BaseModel):
    agent_id: str
    status: str
    customer_reply: Optional[str]
    internal_notes: Optional[str]
    raw_text: str


SUPPORT_AGENT_PROMPT_TEMPLATE = """\
You are drafting a LiteLLM customer support reply.

Apply the rule at `.cursor/rules/customer-support.mdc` and follow the workflow
in `.cursor/skills/draft-support-reply/SKILL.md`. Ground the answer in this
repo and the `litellm-docs` repo (already cloned in this environment).

Customer segment: {segment}
{tone_line}

=== CUSTOMER QUESTION ===
{question}

=== ADDITIONAL CONTEXT (optional) ===
{context}

=== OUTPUT REQUIREMENTS ===
Your entire final message must be ONLY the two sections below (no preamble,
todos narrative, or postamble like "draft is ready above"). The customer reply
must paste cleanly into Gmail AND Slack — Gmail re-flows nested bullets and
"###" headers into a table on send, so use plain text with light formatting
only and wrap it in a "text" fenced block so the Cursor "Copy" button copies
plain text.

=== CUSTOMER REPLY ===
```text
Short, copy-paste reply: under 350 words. Plain text with:
- Numbered sections as "1. Title" on their own line (NO ### headers).
- Single-level "- " bullets only (no nested bullets).
- No bold **...** or italic for prose; inline backticks OK for short
  identifiers like x-litellm-api-key.
- At most ONE small code fence (<=15 lines) inside the reply.
- No repo paths, no Python symbols, no confidence scores.
- Doc URLs on their own lines (https://docs.litellm.ai/...).
Assume LiteLLM Enterprise unless segment is "oss" or the customer asks
about OSS vs Enterprise.
```

=== INTERNAL NOTES ===
- Classification: <one line>
- Sources: <paths/URLs>
- Confidence: high | medium | low — <one line why>
- Open questions: <bullets>
- Follow-ups: <bullets>
- Reviewer tip: paste into Gmail with Cmd+Shift+V to keep plain text.

Do not send the reply anywhere. Do not open PRs. Do not modify files.
"""


def _build_prompt(req: DraftReplyRequest) -> str:
    tone_line = (
        f"Tone override from reviewer: {req.tone_override}" if req.tone_override else ""
    )
    return SUPPORT_AGENT_PROMPT_TEMPLATE.format(
        segment=req.customer_segment,
        tone_line=tone_line,
        question=req.question.strip(),
        context=(req.context or "(none provided)").strip(),
    )


def _resolve_cursor_target() -> Tuple[str, Dict[str, str]]:
    """Return (base_url, headers) for the Cursor Cloud Agents API.

    Prefer routing through a litellm proxy's `/cursor` pass-through when
    LITELLM_PROXY_URL is set; otherwise call api.cursor.com directly using
    CURSOR_API_KEY.
    """
    proxy_url = os.getenv("LITELLM_PROXY_URL")
    proxy_key = os.getenv("LITELLM_PROXY_API_KEY")
    if proxy_url:
        if not proxy_key:
            raise RuntimeError(
                "LITELLM_PROXY_URL is set but LITELLM_PROXY_API_KEY is missing."
            )
        base = proxy_url.rstrip("/") + "/cursor"
        headers = {"Authorization": f"Bearer {proxy_key}"}
        return base, headers

    cursor_key = os.getenv("CURSOR_API_KEY")
    if not cursor_key:
        raise RuntimeError(
            "Set CURSOR_API_KEY (or LITELLM_PROXY_URL + LITELLM_PROXY_API_KEY)."
        )
    encoded = base64.b64encode(f"{cursor_key}:".encode("utf-8")).decode("ascii")
    base = os.getenv("CURSOR_API_BASE", "https://api.cursor.com").rstrip("/")
    headers = {"Authorization": f"Basic {encoded}"}
    return base, headers


_FENCE_PATTERN = re.compile(
    r"^```(?:text|plain|plaintext)?\s*\n(?P<inner>.*?)\n```\s*$",
    re.DOTALL,
)


def _strip_outer_text_fence(s: str) -> str:
    """Drop a single outer ```text ... ``` fence so consumers (Slack, Gmail) get clean text.

    The Cursor UI shows a "Copy" button on the fence which copies as plain text;
    here we strip the fence markers for HTTP / Slack responses so the literal
    backticks don't show up in the customer reply.
    """
    match = _FENCE_PATTERN.match(s.strip())
    if match:
        return match.group("inner").strip()
    return s


def _split_sections(raw_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract CUSTOMER REPLY and INTERNAL NOTES sections from agent output."""
    pattern = re.compile(
        r"===\s*CUSTOMER REPLY\s*===\s*(?P<reply>.*?)\s*===\s*INTERNAL NOTES\s*===\s*(?P<notes>.*)",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(raw_text)
    if not match:
        return None, None
    reply = _strip_outer_text_fence(match.group("reply").strip())
    return reply, match.group("notes").strip()


async def _launch_agent(
    client: httpx.AsyncClient,
    base_url: str,
    headers: Dict[str, str],
    prompt: str,
) -> str:
    repo = os.getenv("SUPPORT_AGENT_REPO", "https://github.com/BerriAI/litellm")
    ref = os.getenv("SUPPORT_AGENT_REF", "main")
    model = os.getenv("SUPPORT_AGENT_MODEL")

    body: Dict[str, Any] = {
        "prompt": {"text": prompt},
        "source": {"repository": repo, "ref": ref},
        "target": {"autoCreatePr": False},
    }
    if model:
        body["model"] = model

    resp = await client.post(f"{base_url}/v0/agents", json=body, headers=headers)
    if resp.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Cursor Cloud Agents launch failed ({resp.status_code}): {resp.text}",
        )
    data = resp.json()
    agent_id = data.get("id") or data.get("agentId") or data.get("agent_id")
    if not agent_id:
        raise HTTPException(
            status_code=502, detail=f"Launch response missing agent id: {data}"
        )
    return agent_id


async def _poll_agent(
    client: httpx.AsyncClient,
    base_url: str,
    headers: Dict[str, str],
    agent_id: str,
) -> Dict[str, Any]:
    poll_interval = float(os.getenv("SUPPORT_AGENT_POLL_INTERVAL", "5"))
    timeout = float(os.getenv("SUPPORT_AGENT_TIMEOUT", "600"))
    deadline = asyncio.get_event_loop().time() + timeout

    while True:
        resp = await client.get(f"{base_url}/v0/agents/{agent_id}", headers=headers)
        if resp.status_code >= 400:
            raise HTTPException(
                status_code=502,
                detail=f"Cursor agent status failed ({resp.status_code}): {resp.text}",
            )
        data = resp.json()
        status = (data.get("status") or "").upper()
        if status in {"FINISHED", "COMPLETED", "SUCCEEDED", "DONE"}:
            return data
        if status in {"FAILED", "ERROR", "CANCELLED", "CANCELED", "STOPPED"}:
            raise HTTPException(
                status_code=502,
                detail=f"Cursor agent ended with status={status}: {data}",
            )
        if asyncio.get_event_loop().time() > deadline:
            raise HTTPException(
                status_code=504,
                detail=f"Cursor agent did not finish within {timeout}s (last status={status}).",
            )
        await asyncio.sleep(poll_interval)


async def _fetch_conversation(
    client: httpx.AsyncClient,
    base_url: str,
    headers: Dict[str, str],
    agent_id: str,
) -> str:
    resp = await client.get(
        f"{base_url}/v0/agents/{agent_id}/conversation", headers=headers
    )
    if resp.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"Cursor conversation fetch failed ({resp.status_code}): {resp.text}",
        )
    data = resp.json()
    messages = data.get("messages") or data.get("conversation") or []
    for msg in reversed(messages):
        role = (msg.get("role") or msg.get("author") or "").lower()
        if role in {"assistant", "agent", "ai"}:
            text = msg.get("text") or msg.get("content") or ""
            if isinstance(text, list):
                text = "".join(
                    chunk.get("text", "") for chunk in text if isinstance(chunk, dict)
                )
            if text:
                return text
    return ""


async def produce_draft(req: DraftReplyRequest) -> DraftReplyResponse:
    """Core drafting logic — reusable from the HTTP route and from Slack handlers."""
    try:
        base_url, headers = _resolve_cursor_target()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    prompt = _build_prompt(req)
    logger.info(
        "Launching Cursor Cloud Agent (segment=%s, q_len=%d, ctx_len=%d)",
        req.customer_segment,
        len(req.question),
        len(req.context or ""),
    )

    async with httpx.AsyncClient(timeout=60) as client:
        agent_id = await _launch_agent(client, base_url, headers, prompt)
        logger.info("Cursor agent launched: %s", agent_id)
        status_data = await _poll_agent(client, base_url, headers, agent_id)
        raw_text = await _fetch_conversation(client, base_url, headers, agent_id)

    customer_reply, internal_notes = _split_sections(raw_text)
    return DraftReplyResponse(
        agent_id=agent_id,
        status=str(status_data.get("status", "UNKNOWN")),
        customer_reply=customer_reply,
        internal_notes=internal_notes,
        raw_text=raw_text,
    )


app = FastAPI(title="LiteLLM Customer Support Agent", version="0.1.0")


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/draft-reply", response_model=DraftReplyResponse)
async def draft_reply(req: DraftReplyRequest) -> DraftReplyResponse:
    return await produce_draft(req)


def _maybe_mount_slack(api: FastAPI) -> None:
    """Mount Slack endpoints if slack-bolt is installed and credentials are present."""
    try:
        from support.slack_app import get_slack_handler
    except ImportError as exc:
        logger.info("Slack handler not mounted (import failed: %s)", exc)
        return

    handler = get_slack_handler()
    if handler is None:
        logger.info(
            "Slack handler not configured "
            "(set SLACK_BOT_TOKEN and SLACK_SIGNING_SECRET to enable)"
        )
        return

    async def _dispatch(request):
        return await handler.handle(request)

    api.post("/slack/events")(_dispatch)
    api.post("/slack/commands")(_dispatch)
    api.post("/slack/interactions")(_dispatch)
    logger.info("Slack handler mounted at /slack/{events,commands,interactions}")


_maybe_mount_slack(app)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("SUPPORT_AGENT_PORT", "8088"))
    uvicorn.run(app, host="0.0.0.0", port=port)
