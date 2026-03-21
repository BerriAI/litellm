"""
LiteLLM Policy Engine

The Policy Engine allows administrators to define policies that combine guardrails
with scoping rules. Policies can target specific teams, API keys, and models using
wildcard patterns, and support inheritance from base policies.

Configuration structure:
- `policies`: Define WHAT guardrails to apply (with inheritance and conditions)
- `policy_attachments`: Define WHERE policies apply (teams, keys, models)

Example:
```yaml
policies:
  global-baseline:
    description: "Base guardrails for all requests"
    guardrails:
      add: [pii_blocker]

  gpt4-safety:
    inherit: global-baseline
    description: "Extra safety for GPT-4"
    guardrails:
      add: [toxicity_filter]
    condition:
      model: "gpt-4.*"  # regex pattern

policy_attachments:
  - policy: global-baseline
    scope: "*"
  - policy: gpt4-safety
    scope: "*"
```
"""

from litellm.proxy.policy_engine.attachment_registry import (
    AttachmentRegistry,
    get_attachment_registry,
)
from litellm.proxy.policy_engine.condition_evaluator import ConditionEvaluator
from litellm.proxy.policy_engine.policy_matcher import PolicyMatcher
from litellm.proxy.policy_engine.policy_registry import (
    PolicyRegistry,
    get_policy_registry,
)
from litellm.proxy.policy_engine.policy_resolver import PolicyResolver
from litellm.proxy.policy_engine.policy_validator import PolicyValidator

__all__ = [
    # Registries
    "PolicyRegistry",
    "get_policy_registry",
    "AttachmentRegistry",
    "get_attachment_registry",
    # Core components
    "PolicyMatcher",
    "PolicyResolver",
    "PolicyValidator",
    "ConditionEvaluator",
]
