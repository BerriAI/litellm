"""
litellm.interactions.agents

Full CRUD SDK for provider-side managed agents (e.g. Gemini v1beta/agents).

    litellm.interactions.agents.create(name=..., ...)
    litellm.interactions.agents.list(api_key=...)
    litellm.interactions.agents.get(name=..., ...)
    litellm.interactions.agents.delete(name=..., ...)
    litellm.interactions.agents.list_versions(name=..., ...)

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
    "create",
    "acreate",
    "list",
    "alist",
    "get",
    "aget",
    "delete",
    "adelete",
    "list_versions",
    "alist_versions",
]
