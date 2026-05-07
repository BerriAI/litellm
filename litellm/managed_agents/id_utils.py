"""Prefixed UUID helpers for managed agents.

Format: `<prefix><uuid4 hex (no dashes)>` → 36-char strings like
`agt_2c1f7a3b9d4e4f8aa7c2d1e3f4b5a6c7`.
"""

import uuid

AGENT_ID_PREFIX = "agt_"
SESSION_ID_PREFIX = "ses_"
MESSAGE_ID_PREFIX = "msg_"


def _new_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex}"


def new_agent_id() -> str:
    return _new_id(AGENT_ID_PREFIX)


def new_session_id() -> str:
    return _new_id(SESSION_ID_PREFIX)


def new_message_id() -> str:
    return _new_id(MESSAGE_ID_PREFIX)


def is_agent_id(s: str) -> bool:
    return isinstance(s, str) and s.startswith(AGENT_ID_PREFIX)


def is_session_id(s: str) -> bool:
    return isinstance(s, str) and s.startswith(SESSION_ID_PREFIX)


def is_message_id(s: str) -> bool:
    return isinstance(s, str) and s.startswith(MESSAGE_ID_PREFIX)
