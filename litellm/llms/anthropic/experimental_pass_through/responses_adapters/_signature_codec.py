"""Signature codec for round-tripping OpenAI Responses reasoning items
through Anthropic's `thinking.signature` field.

Claude Code's ``clear_thinking_20251015`` context-management edit preserves
only ``{type, thinking, signature}`` on each turn — the plaintext reasoning
is cleared. To replay an OpenAI Responses ``reasoning`` item we need both
``id`` and ``encrypted_content``; we pack both into the single ``signature``
channel that Claude Code preserves.

``_unpack_signature`` returns ``(None, None)`` for any signature without
the ``lllm-rsenc-v1:`` prefix, so native Anthropic thinking signatures
(produced by a model switch mid-conversation) safely degrade — we skip
the reasoning replay rather than crashing.
"""

import base64
import json
from typing import Optional, Tuple

_SIG_PREFIX = "lllm-rsenc-v1:"


def _pack_signature(
    rs_id: Optional[str], encrypted_content: Optional[str]
) -> Optional[str]:
    if not encrypted_content:
        return None
    return (
        _SIG_PREFIX
        + base64.b64encode(
            json.dumps({"id": rs_id, "ec": encrypted_content}).encode()
        ).decode()
    )


def _unpack_signature(sig: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not sig or not sig.startswith(_SIG_PREFIX):
        return None, None
    try:
        p = json.loads(base64.b64decode(sig[len(_SIG_PREFIX) :]).decode())
        return p.get("id"), p.get("ec")
    except Exception:
        return None, None
