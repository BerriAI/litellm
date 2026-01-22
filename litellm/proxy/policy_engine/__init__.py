"""
LiteLLM Policy Engine

The Policy Engine allows administrators to define policies that combine guardrails
with scoping rules. Policies can target specific teams, API keys, and models using
wildcard patterns, and support inheritance from base policies.
"""

from litellm.proxy.policy_engine.policy_matcher import PolicyMatcher
from litellm.proxy.policy_engine.policy_registry import (
    PolicyRegistry,
    get_policy_registry,
)
from litellm.proxy.policy_engine.policy_resolver import PolicyResolver
from litellm.proxy.policy_engine.policy_validator import PolicyValidator

__all__ = [
    "PolicyRegistry",
    "get_policy_registry",
    "PolicyMatcher",
    "PolicyResolver",
    "PolicyValidator",
]
