"""
litellm.interactions.agents

Full CRUD SDK for provider-side managed agents (e.g. Gemini v1beta/agents).

    litellm.interactions.agents.create(id=..., ...)
    litellm.interactions.agents.list(api_key=...)
    litellm.interactions.agents.get(id=..., ...)
    litellm.interactions.agents.delete(id=..., ...)
    litellm.interactions.agents.list_versions(id=..., ...)

Async counterparts: acreate, alist, aget, adelete, alist_versions
"""

from litellm.interactions.agents.main import (
    acreate,
    adelete,
    aget,
    alist,
    alist_versions,
    create,
    delete,
    get,
    list,
    list_versions,
)

__all__ = [
    "acreate",
    "adelete",
    "aget",
    "alist",
    "alist_versions",
    "create",
    "delete",
    "get",
    "list",
    "list_versions",
]
