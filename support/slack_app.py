"""Slack Bolt app for the LiteLLM customer support drafting agent.

Exposes:
- Slash command: /support-draft <question>           (or open a modal if no args)
- Global shortcut + message shortcut: "Draft support reply"
- Modal: collects question, context, customer segment, tone override
- Background drafting: posts the draft (or error) back to the channel/thread

Drafting itself runs via support.customer_support_agent.produce_draft, which
launches a Cursor Cloud Agent with the litellm + litellm-docs multi-repo
environment and applies the support rule and skill.

Required env vars (Slack handler only mounts if both are set):
    SLACK_BOT_TOKEN         xoxb-... (scopes below)
    SLACK_SIGNING_SECRET    from the Slack app's Basic Information page

Required bot scopes (set in the Slack app config):
    commands, chat:write, chat:write.public, im:write
    (Add app_mentions:read if you want @-mention support later.)

The slash command, shortcut callback_ids, and modal callback_id below must
match what is configured in the Slack app:
    Slash command:      /support-draft
    Message shortcut:   callback_id = draft_support_reply_msg
    Global shortcut:    callback_id = draft_support_reply_global
    Modal:              callback_id = draft_support_reply_modal
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("litellm.support_agent.slack")

_SLACK_HANDLER: Optional[Any] = None


def get_slack_handler() -> Optional[Any]:
    """Return a cached AsyncSlackRequestHandler, or None if Slack is not configured.

    Importing slack_bolt is deferred so the support service runs without the
    optional dependency installed.
    """
    global _SLACK_HANDLER
    if _SLACK_HANDLER is not None:
        return _SLACK_HANDLER

    bot_token = os.getenv("SLACK_BOT_TOKEN")
    signing_secret = os.getenv("SLACK_SIGNING_SECRET")
    if not bot_token or not signing_secret:
        return None

    try:
        from slack_bolt.adapter.fastapi.async_handler import (
            AsyncSlackRequestHandler,
        )
        from slack_bolt.async_app import AsyncApp
    except ImportError as exc:
        logger.warning(
            "slack-bolt not installed; cannot enable Slack handler (%s). "
            "Install with: pip install slack-bolt",
            exc,
        )
        return None

    bolt_app = AsyncApp(token=bot_token, signing_secret=signing_secret)
    _register_handlers(bolt_app)
    _SLACK_HANDLER = AsyncSlackRequestHandler(bolt_app)
    return _SLACK_HANDLER


def _parse_csv_env(name: str) -> Set[str]:
    raw = os.getenv(name, "")
    return {item.strip() for item in raw.split(",") if item.strip()}


def _slack_access_check(user_id: str, channel_id: str) -> Tuple[bool, str]:
    """Return (allowed, reason).

    Two env vars control access; both are comma-separated. If both are empty,
    access is open to any workspace member (current default).

    SUPPORT_AGENT_SLACK_ALLOWED_USERS      e.g. "U01ABC,U02DEF"
    SUPPORT_AGENT_SLACK_ALLOWED_CHANNELS   e.g. "C01XYZ,C02UVW"

    When non-empty, the requester must match every populated allowlist. The
    channel allowlist is skipped only when channel_id is empty (global
    shortcut DM flow) — set SUPPORT_AGENT_SLACK_BLOCK_GLOBAL_SHORTCUT=1 to
    deny that path as well.
    """
    allowed_users = _parse_csv_env("SUPPORT_AGENT_SLACK_ALLOWED_USERS")
    allowed_channels = _parse_csv_env("SUPPORT_AGENT_SLACK_ALLOWED_CHANNELS")
    block_global = os.getenv("SUPPORT_AGENT_SLACK_BLOCK_GLOBAL_SHORTCUT") == "1"

    if allowed_users and user_id not in allowed_users:
        return False, "user not in allowlist"
    if allowed_channels:
        if not channel_id:
            if block_global:
                return False, "global-shortcut / DM path is disabled"
        elif channel_id not in allowed_channels:
            return False, "channel not in allowlist"
    return True, ""


_DENY_TEXT = (
    ":lock: You don't have access to draft support replies. "
    "Ping the support team if you think this is a mistake."
)


def _register_handlers(bolt_app: Any) -> None:
    from support.customer_support_agent import DraftReplyRequest, produce_draft

    @bolt_app.command("/support-draft")
    async def handle_command(ack, body, client):
        text = (body.get("text") or "").strip()
        channel_id = body.get("channel_id") or ""
        user_id = body.get("user_id") or ""
        trigger_id = body.get("trigger_id") or ""

        allowed, reason = _slack_access_check(user_id, channel_id)
        if not allowed:
            logger.info(
                "Slack access denied (command): user=%s channel=%s reason=%s",
                user_id,
                channel_id,
                reason,
            )
            await ack(response_type="ephemeral", text=_DENY_TEXT)
            return

        if not text:
            await ack()
            await client.views_open(
                trigger_id=trigger_id,
                view=_modal_view(
                    initial_question="",
                    initial_context="",
                    metadata={"channel_id": channel_id, "user_id": user_id},
                ),
            )
            return

        await ack(
            response_type="ephemeral",
            text=":writing_hand: Drafting a support reply (this can take a minute)...",
        )
        await _post_draft_async(
            client=client,
            req=DraftReplyRequest(question=text),
            channel_id=channel_id,
            user_id=user_id,
            thread_ts=None,
            produce_draft_fn=produce_draft,
        )

    @bolt_app.shortcut("draft_support_reply_global")
    async def handle_global_shortcut(ack, body, client):
        await ack()
        user_id = (body.get("user") or {}).get("id") or ""

        allowed, reason = _slack_access_check(user_id, channel_id="")
        if not allowed:
            logger.info(
                "Slack access denied (global shortcut): user=%s reason=%s",
                user_id,
                reason,
            )
            if user_id:
                await client.chat_postMessage(channel=user_id, text=_DENY_TEXT)
            return

        await client.views_open(
            trigger_id=body.get("trigger_id"),
            view=_modal_view(
                initial_question="",
                initial_context="",
                metadata={"channel_id": "", "user_id": user_id},
            ),
        )

    @bolt_app.shortcut("draft_support_reply_msg")
    async def handle_message_shortcut(ack, body, client):
        await ack()
        message = body.get("message") or {}
        channel = body.get("channel") or {}
        user = body.get("user") or {}
        user_id = user.get("id") or ""
        channel_id = channel.get("id") or ""

        allowed, reason = _slack_access_check(user_id, channel_id)
        if not allowed:
            logger.info(
                "Slack access denied (message shortcut): user=%s channel=%s reason=%s",
                user_id,
                channel_id,
                reason,
            )
            if user_id:
                await client.chat_postMessage(channel=user_id, text=_DENY_TEXT)
            return

        await client.views_open(
            trigger_id=body.get("trigger_id"),
            view=_modal_view(
                initial_question="",
                initial_context=message.get("text") or "",
                metadata={
                    "channel_id": channel.get("id") or "",
                    "user_id": user.get("id") or "",
                    "thread_ts": message.get("thread_ts") or message.get("ts") or "",
                },
            ),
        )

    @bolt_app.view("draft_support_reply_modal")
    async def handle_view_submission(ack, body, view, client):
        try:
            submitter_metadata = json.loads(view.get("private_metadata") or "{}")
        except json.JSONDecodeError:
            submitter_metadata = {}
        submitter_id = submitter_metadata.get("user_id") or (
            body.get("user") or {}
        ).get("id", "")
        submitter_channel = submitter_metadata.get("channel_id") or ""

        allowed, reason = _slack_access_check(submitter_id, submitter_channel)
        if not allowed:
            logger.info(
                "Slack access denied (modal submission): user=%s channel=%s reason=%s",
                submitter_id,
                submitter_channel,
                reason,
            )
            await ack(
                response_action="errors",
                errors={"question_block": _DENY_TEXT},
            )
            return

        await ack()
        values = view["state"]["values"]
        question = values["question_block"]["question"]["value"] or ""
        context = (values.get("context_block", {}).get("context") or {}).get("value")
        segment = (
            values.get("segment_block", {})
            .get("segment", {})
            .get("selected_option", {})
            .get("value", "paying")
        )
        tone = (values.get("tone_block", {}).get("tone") or {}).get("value")

        try:
            metadata = json.loads(view.get("private_metadata") or "{}")
        except json.JSONDecodeError:
            metadata = {}

        user_id = metadata.get("user_id") or (body.get("user") or {}).get("id", "")
        channel_id = metadata.get("channel_id") or user_id  # DM if no channel
        thread_ts = metadata.get("thread_ts") or None

        req = DraftReplyRequest(
            question=question.strip(),
            context=(context or None),
            customer_segment=(
                segment if segment in {"paying", "prospect", "oss"} else "paying"
            ),
            tone_override=tone or None,
        )

        await _post_draft_async(
            client=client,
            req=req,
            channel_id=channel_id,
            user_id=user_id,
            thread_ts=thread_ts,
            produce_draft_fn=produce_draft,
        )


def _modal_view(
    initial_question: str,
    initial_context: str,
    metadata: Dict[str, str],
) -> Dict[str, Any]:
    return {
        "type": "modal",
        "callback_id": "draft_support_reply_modal",
        "private_metadata": json.dumps(metadata),
        "title": {"type": "plain_text", "text": "Draft Support Reply"},
        "submit": {"type": "plain_text", "text": "Draft"},
        "close": {"type": "plain_text", "text": "Cancel"},
        "blocks": [
            {
                "type": "input",
                "block_id": "question_block",
                "label": {"type": "plain_text", "text": "Customer question"},
                "element": {
                    "type": "plain_text_input",
                    "action_id": "question",
                    "multiline": True,
                    "initial_value": initial_question,
                },
            },
            {
                "type": "input",
                "block_id": "context_block",
                "optional": True,
                "label": {
                    "type": "plain_text",
                    "text": "Context (logs, config, version, redacted)",
                },
                "element": {
                    "type": "plain_text_input",
                    "action_id": "context",
                    "multiline": True,
                    "initial_value": initial_context,
                },
            },
            {
                "type": "input",
                "block_id": "segment_block",
                "label": {"type": "plain_text", "text": "Customer segment"},
                "element": {
                    "type": "static_select",
                    "action_id": "segment",
                    "initial_option": {
                        "text": {"type": "plain_text", "text": "Paying (Enterprise)"},
                        "value": "paying",
                    },
                    "options": [
                        {
                            "text": {
                                "type": "plain_text",
                                "text": "Paying (Enterprise)",
                            },
                            "value": "paying",
                        },
                        {
                            "text": {"type": "plain_text", "text": "Prospect"},
                            "value": "prospect",
                        },
                        {
                            "text": {"type": "plain_text", "text": "OSS"},
                            "value": "oss",
                        },
                    ],
                },
            },
            {
                "type": "input",
                "block_id": "tone_block",
                "optional": True,
                "label": {
                    "type": "plain_text",
                    "text": "Tone override (optional, for reviewer)",
                },
                "element": {
                    "type": "plain_text_input",
                    "action_id": "tone",
                    "placeholder": {
                        "type": "plain_text",
                        "text": "e.g. customer is frustrated, lead with empathy",
                    },
                },
            },
        ],
    }


async def _post_draft_async(
    client: Any,
    req: Any,
    channel_id: str,
    user_id: str,
    thread_ts: Optional[str],
    produce_draft_fn: Any,
) -> None:
    """Post a 'drafting...' message, run the draft, then post the result."""
    target_channel = channel_id or user_id
    try:
        await client.chat_postMessage(
            channel=target_channel,
            thread_ts=thread_ts,
            text=f":writing_hand: Drafting a support reply for <@{user_id}>...",
        )
    except Exception:
        logger.exception("Slack 'drafting' message failed (continuing)")

    try:
        result = await produce_draft_fn(req)
    except Exception as exc:
        logger.exception("Draft generation failed")
        await client.chat_postMessage(
            channel=target_channel,
            thread_ts=thread_ts,
            text=f":x: Draft failed: `{exc}`",
        )
        return

    blocks = _format_blocks(result)
    await client.chat_postMessage(
        channel=target_channel,
        thread_ts=thread_ts,
        text=f"Draft support reply ready (agent `{result.agent_id}`)",
        blocks=blocks,
    )


def _format_blocks(result: Any) -> List[Dict[str, Any]]:
    """Format the draft into Slack blocks; chunk long text to stay under limits."""
    blocks: List[Dict[str, Any]] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "Customer reply (draft)"},
        }
    ]
    blocks.extend(
        _mrkdwn_sections(
            result.customer_reply or "_(no reply produced — see raw text)_"
        )
    )
    blocks.append({"type": "divider"})
    blocks.append(
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "Internal notes"},
        }
    )
    blocks.extend(_mrkdwn_sections(result.internal_notes or "_(no notes)_"))
    blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": (
                        f"agent: `{result.agent_id}`  ·  status: `{result.status}`  ·  "
                        "human review required before sending"
                    ),
                }
            ],
        }
    )
    return blocks


def _mrkdwn_sections(text: str, max_len: int = 2900) -> List[Dict[str, Any]]:
    """Split text into multiple section blocks if it exceeds Slack's 3000-char limit."""
    if not text:
        return [{"type": "section", "text": {"type": "mrkdwn", "text": "_(empty)_"}}]
    chunks: List[str] = []
    remaining = text
    while len(remaining) > max_len:
        split = remaining.rfind("\n", 0, max_len)
        if split <= 0:
            split = max_len
        chunks.append(remaining[:split])
        remaining = remaining[split:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return [
        {"type": "section", "text": {"type": "mrkdwn", "text": chunk}}
        for chunk in chunks
    ]
