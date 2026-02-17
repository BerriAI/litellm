"""
Core policy type definitions.

Policy Engine Configuration:
```yaml
policies:
  global-baseline:
    description: "Base guardrails for all requests"
    guardrails:
      add: [pii_blocker]

  healthcare-compliance:
    inherit: global-baseline
    guardrails:
      add: [hipaa_audit]
    condition:
      model: "gpt-4"  # exact match or regex pattern

policy_attachments:
  - policy: global-baseline
    scope: "*"
  - policy: healthcare-compliance
    teams: [healthcare-team]
```

Key concepts:
- `policies`: Define WHAT guardrails to apply (with inheritance via `inherit` and `guardrails.add`/`remove`)
- `policy_attachments`: Define WHERE policies apply (teams, keys, models)
- `condition`: Optional model condition for when guardrails apply
"""

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from litellm.types.proxy.policy_engine.pipeline_types import GuardrailPipeline

# ─────────────────────────────────────────────────────────────────────────────
# Policy Condition
# ─────────────────────────────────────────────────────────────────────────────


class PolicyCondition(BaseModel):
    """
    Condition for when a policy's guardrails apply.

    Currently supports model-based conditions with exact match or regex.

    Example YAML:
    ```yaml
    condition:
      model: "gpt-4"           # exact match
      model: "gpt-4.*"         # regex pattern
      model: ["gpt-4", "gpt-4-turbo"]  # list of exact matches
    ```
    """

    model: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Model name(s) to match. Can be exact string, regex pattern, or list.",
    )

    model_config = ConfigDict(extra="forbid")


# ─────────────────────────────────────────────────────────────────────────────
# Policy Scope (used internally by attachments)
# ─────────────────────────────────────────────────────────────────────────────


class PolicyScope(BaseModel):
    """
    Defines the scope for matching requests.

    Used internally by PolicyAttachment to define WHERE a policy applies.

    Scope Fields:
    | Field  | What it matches | Wildcard support      | Default behavior    |
    |--------|-----------------|----------------------|---------------------|
    | teams  | Team aliases    | *, healthcare-*      | None → matches all  |
    | keys   | Key aliases     | *, dev-key-*         | None → matches all  |
    | models | Model names     | *, bedrock/*, gpt-*  | None → matches all  |
    | tags   | Key/team tags   | *, health-*, prod-*  | None → not checked  |

    If teams/keys/models is None or empty, it defaults to matching everything (["*"]).
    If tags is None or empty, the tag dimension is NOT checked (matches all).
    A request must match ALL specified scope fields for the attachment to apply.
    """

    teams: Optional[List[str]] = Field(
        default=None,
        description="Team aliases or wildcard patterns. Use '*' for all teams.",
    )
    keys: Optional[List[str]] = Field(
        default=None,
        description="Key aliases or wildcard patterns. Use '*' for all keys.",
    )
    models: Optional[List[str]] = Field(
        default=None,
        description="Model names or wildcard patterns. Use '*' for all models.",
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Tag patterns to match against key/team tags. Supports wildcards (e.g., health-*).",
    )

    model_config = ConfigDict(extra="forbid")

    def get_teams(self) -> List[str]:
        """Returns teams list, defaulting to ['*'] if not specified."""
        return self.teams if self.teams else ["*"]

    def get_keys(self) -> List[str]:
        """Returns keys list, defaulting to ['*'] if not specified."""
        return self.keys if self.keys else ["*"]

    def get_models(self) -> List[str]:
        """Returns models list, defaulting to ['*'] if not specified."""
        return self.models if self.models else ["*"]

    def get_tags(self) -> List[str]:
        """Returns tags list, defaulting to empty list if not specified.

        Unlike teams/keys/models, empty tags means 'do not check tags'
        rather than 'match all'. This is because tags are opt-in scoping.
        """
        return self.tags if self.tags else []


# ─────────────────────────────────────────────────────────────────────────────
# Policy Guardrails
# ─────────────────────────────────────────────────────────────────────────────


class PolicyGuardrails(BaseModel):
    """
    Defines guardrails to add or remove in a policy.

    - `add`: List of guardrail names to add (on top of inherited guardrails)
    - `remove`: List of guardrail names to remove (from inherited guardrails)

    This supports the inheritance pattern where child policies can:
    - Add new guardrails on top of parent's guardrails
    - Remove specific guardrails inherited from parent
    """

    add: Optional[List[str]] = Field(
        default=None,
        description="Guardrail names to add to this policy.",
    )
    remove: Optional[List[str]] = Field(
        default=None,
        description="Guardrail names to remove (typically from inherited policy).",
    )

    model_config = ConfigDict(extra="forbid")

    def get_add(self) -> List[str]:
        """Returns add list, defaulting to empty list if not specified."""
        return self.add if self.add else []

    def get_remove(self) -> List[str]:
        """Returns remove list, defaulting to empty list if not specified."""
        return self.remove if self.remove else []


# ─────────────────────────────────────────────────────────────────────────────
# Policy
# ─────────────────────────────────────────────────────────────────────────────


class Policy(BaseModel):
    """
    A policy that defines WHAT guardrails to apply.

    Policies define guardrails but NOT where they apply - that's done via policy_attachments.

    Policies can inherit from other policies using the `inherit` field.
    When inheriting:
    - Guardrails from `guardrails.add` are added to the inherited guardrails
    - Guardrails from `guardrails.remove` are removed from the inherited guardrails

    Policies can have a `condition` for model-based guardrail application.

    Example configuration:
    ```yaml
    policies:
      global-baseline:
        description: "Base guardrails for all requests"
        guardrails:
          add:
            - pii_blocker
            - phi_blocker

      healthcare-compliance:
        inherit: global-baseline
        description: "HIPAA compliance for healthcare"
        guardrails:
          add:
            - hipaa_audit

      gpt4-safety:
        description: "Extra safety for GPT-4 models"
        guardrails:
          add:
            - toxicity_filter
        condition:
          model: "gpt-4.*"  # regex pattern

    policy_attachments:
      - policy: global-baseline
        scope: "*"
      - policy: healthcare-compliance
        teams: [healthcare-team]
      - policy: gpt4-safety
        scope: "*"
    ```
    """

    inherit: Optional[str] = Field(
        default=None,
        description="Name of the parent policy to inherit from.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of the policy.",
    )
    guardrails: PolicyGuardrails = Field(
        default_factory=PolicyGuardrails,
        description="Guardrails configuration with add/remove lists.",
    )
    condition: Optional[PolicyCondition] = Field(
        default=None,
        description="Optional condition for when this policy's guardrails apply.",
    )
    pipeline: Optional[GuardrailPipeline] = Field(
        default=None,
        description="Optional pipeline for ordered, conditional guardrail execution.",
    )

    model_config = ConfigDict(extra="forbid")


# ─────────────────────────────────────────────────────────────────────────────
# Policy Attachments
# ─────────────────────────────────────────────────────────────────────────────


class PolicyAttachment(BaseModel):
    """
    Attaches a policy to a scope - defines WHERE a policy applies.

    Attachments are REQUIRED to make policies active. A policy without
    an attachment will not be applied to any requests.

    Example YAML:
    ```yaml
    policy_attachments:
      - policy: global-baseline
        scope: "*"  # applies to all requests
      - policy: healthcare-compliance
        teams: [healthcare-team, medical-research]
      - policy: dev-safety
        keys: ["dev-key-*", "test-key-*"]
      - policy: gpt4-specific
        models: ["gpt-4", "gpt-4-turbo"]
    ```
    """

    policy: str = Field(
        description="Name of the policy to attach.",
    )
    scope: Optional[str] = Field(
        default=None,
        description="Use '*' for global scope (applies to all requests).",
    )
    teams: Optional[List[str]] = Field(
        default=None,
        description="Team aliases or patterns this attachment applies to.",
    )
    keys: Optional[List[str]] = Field(
        default=None,
        description="Key aliases or patterns this attachment applies to.",
    )
    models: Optional[List[str]] = Field(
        default=None,
        description="Model names or patterns this attachment applies to.",
    )
    tags: Optional[List[str]] = Field(
        default=None,
        description="Tag patterns this attachment applies to. Supports wildcards (e.g., health-*).",
    )

    model_config = ConfigDict(extra="forbid")

    def is_global(self) -> bool:
        """Check if this is a global attachment (scope='*')."""
        return self.scope == "*"

    def to_policy_scope(self) -> PolicyScope:
        """Convert attachment to a PolicyScope for matching."""
        if self.is_global():
            return PolicyScope(teams=["*"], keys=["*"], models=["*"])
        return PolicyScope(
            teams=self.teams,
            keys=self.keys,
            models=self.models,
            tags=self.tags,
        )


class PolicyConfig(BaseModel):
    """
    Root configuration for all policies.

    Maps policy names to their Policy definitions.
    """

    policies: Dict[str, Policy] = Field(
        default_factory=dict,
        description="Map of policy names to Policy objects.",
    )

    model_config = ConfigDict(extra="forbid")
