"""
LiteLLM Proxy Skills - Database-backed skills storage

This module provides database-backed skills storage as an alternative to
Anthropic's cloud-based skills API.

Main components:
- handler.py: LiteLLMSkillsTransformationHandler - sync/async methods for CRUD
- transformation.py: LiteLLMSkillsConfig - BaseSkillsAPIConfig implementation (optional)
"""

from litellm.llms.litellm_proxy.skills.handler import (
    LiteLLMSkillsTransformationHandler,
)

