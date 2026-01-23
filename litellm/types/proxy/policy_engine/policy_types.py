"""
Core policy type definitions.

These types define the structure of policies in the configuration.

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
    statements:
      - sid: "GPT4Only"
        guardrails: [toxicity_filter]
        condition:
          model:
            in: ["gpt-4", "gpt-4-turbo"]

policy_attachments:
  - policy: global-baseline
    scope: "*"
  - policy: healthcare-compliance
    teams: [healthcare-team]
```

Key concepts:
- `policies`: Define WHAT guardrails to apply (with inheritance via `inherit` and `guardrails.add`/`remove`)
- `policy_attachments`: Define WHERE policies apply (teams, keys, models)
- `statements`: Fine-grained conditional guardrails within a policy
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

# ─────────────────────────────────────────────────────────────────────────────
# Condition Operators (AWS IAM-style)
# ─────────────────────────────────────────────────────────────────────────────


class ConditionOperator(BaseModel):
    """
    AWS IAM-style condition operators for matching values.

    Supports:
    - equals: Exact string match
    - in_: Value must be in the list (alias: "in" in YAML)
    - prefix: Value must start with the given prefix
    - not_equals: Value must NOT equal
    - not_in: Value must NOT be in the list

    Example YAML:
    ```yaml
    condition:
      model:
        in: ["gpt-4", "gpt-4-turbo"]
      team:
        prefix: "healthcare-"
    ```
    """

    equals: Optional[str] = Field(
        default=None,
        description="Exact string match.",
    )
    in_: Optional[List[str]] = Field(
        default=None,
        alias="in",
        description="Value must be in this list.",
    )
    prefix: Optional[str] = Field(
        default=None,
        description="Value must start with this prefix.",
    )
    not_equals: Optional[str] = Field(
        default=None,
        description="Value must NOT equal this.",
    )
    not_in: Optional[List[str]] = Field(
        default=None,
        alias="notIn",
        description="Value must NOT be in this list.",
    )

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class PolicyCondition(BaseModel):
    """
    Condition for when a policy statement applies.

    All specified conditions must match (AND logic).
    If a field is None, it matches any value for that field.

    Example YAML:
    ```yaml
    condition:
      model:
        in: ["gpt-4", "gpt-4-turbo"]
      team:
        prefix: "healthcare-"
      metadata:
        environment:
          equals: "production"
    ```
    """

    model: Optional[ConditionOperator] = Field(
        default=None,
        description="Condition on the model name.",
    )
    team: Optional[ConditionOperator] = Field(
        default=None,
        description="Condition on the team alias.",
    )
    key: Optional[ConditionOperator] = Field(
        default=None,
        description="Condition on the API key alias.",
    )
    metadata: Optional[Dict[str, ConditionOperator]] = Field(
        default=None,
        description="Conditions on request metadata fields.",
    )

    model_config = ConfigDict(extra="forbid")


# ─────────────────────────────────────────────────────────────────────────────
# Policy Statements
# ─────────────────────────────────────────────────────────────────────────────


class PolicyStatement(BaseModel):
    """
    A single statement within a policy.

    Statements allow fine-grained control over when guardrails apply
    using AWS IAM-style conditions.

    Example YAML:
    ```yaml
    statements:
      - sid: "RequirePIIOnGPT4"
        guardrails: [pii_blocker]
        condition:
          model:
            in: ["gpt-4", "gpt-4-turbo"]
    ```
    """

    sid: Optional[str] = Field(
        default=None,
        description="Statement ID for identification and debugging.",
    )
    guardrails: List[str] = Field(
        default_factory=list,
        description="Guardrail names to apply when condition matches.",
    )
    condition: Optional[PolicyCondition] = Field(
        default=None,
        description="Condition for when this statement applies. If None, always applies.",
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
    | Field  | What it matches | Wildcard support      |
    |--------|-----------------|----------------------|
    | teams  | Team aliases    | *, healthcare-*      |
    | keys   | Key aliases     | *, dev-key-*         |
    | models | Model names     | *, bedrock/*, gpt-*  |

    If a field is None or empty, it defaults to matching everything (["*"]).
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


class Policy(BaseModel):
    """
    A policy that defines WHAT guardrails to apply.

    Policies define guardrails but NOT where they apply - that's done via policy_attachments.

    Policies can inherit from other policies using the `inherit` field.
    When inheriting:
    - Guardrails from `guardrails.add` are added to the inherited guardrails
    - Guardrails from `guardrails.remove` are removed from the inherited guardrails

    Policies can also have `statements` for fine-grained conditional guardrails.
    Statements are evaluated in addition to the base guardrails.

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

      internal-dev:
        inherit: global-baseline
        description: "Relaxed policy for dev"
        guardrails:
          add:
            - toxicity_filter
          remove:
            - phi_blocker

      conditional-policy:
        description: "Model-specific guardrails"
        guardrails:
          add:
            - base_guardrail
        statements:
          - sid: "GPT4Safety"
            guardrails: [toxicity_filter]
            condition:
              model:
                in: ["gpt-4", "gpt-4-turbo"]

    policy_attachments:
      - policy: global-baseline
        scope: "*"
      - policy: healthcare-compliance
        teams: [healthcare-team]
      - policy: internal-dev
        keys: ["dev-key-*"]
    ```
    """

    inherit: Optional[str] = Field(
        default=None,
        description="Name of the parent policy to inherit from.",
    )
    guardrails: PolicyGuardrails = Field(
        default_factory=PolicyGuardrails,
        description="Guardrails configuration with add/remove lists.",
    )
    statements: Optional[List[PolicyStatement]] = Field(
        default=None,
        description="Optional list of conditional statements for fine-grained guardrail control.",
    )
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of the policy.",
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
