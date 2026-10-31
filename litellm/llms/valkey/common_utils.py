"""Shared helpers for Valkey integrations (semantic cache, vector stores)."""

import struct
from collections.abc import Sequence
from typing import Final
from urllib.parse import quote

# Sent via CLIENT SETINFO LIB-NAME so server-side client listings can attribute
# connections to LiteLLM. redis-py's lib_name replaces its default "redis-py",
# so the base name is kept and the tag appended in the conventional base(tag) form.
VALKEY_LIB_NAME: Final = "redis-py(litellm)"


def build_valkey_url(host: str, port: str, password: str | None = None, ssl: bool = False) -> str:
    """Deliberately reads no environment: callers of the vector store control the
    host, so an env-sourced password would be sent to a caller-chosen server."""
    credentials: Final = f":{quote(password, safe='')}@" if password else ""
    scheme: Final = "rediss" if ssl else "redis"
    return f"{scheme}://{credentials}{host}:{port}"


def pack_vector(embedding: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(embedding)}f", *embedding)
