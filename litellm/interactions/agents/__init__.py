"""
litellm.interactions.agents

Exposes:
    create(...)   — sync SDK method for provider-side agent creation
    acreate(...)  — async SDK method for provider-side agent creation
"""

from litellm.interactions.agents.main import acreate, create

__all__ = ["create", "acreate"]
