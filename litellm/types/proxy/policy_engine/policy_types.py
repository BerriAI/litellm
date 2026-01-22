"""
Core policy type definitions.

These types define the structure of policies in the configuration.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PolicyScope(BaseModel):
    """
    Defines the scope for a policy - which requests it applies to.

    Scope Fields:
    | Field  | What it matches | Wildcard support      |
    |--------|-----------------|----------------------|
    | teams  | Team aliases    | *, healthcare-*      |
    | keys   | Key aliases     | *, dev-key-*         |
    | models | Model names     | *, bedrock/*, gpt-*  |

    If a field is None or empty, it defaults to matching everything (["*"]).
    A request must match ALL specified scope fields for the policy to apply.
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


class PolicyGuardrails(BaseModel):
    """
    Defines guardrails to add or remove in a policy.

    - `add`: List of guardrail names to add (on top of inherited guardrails)
    - `remove`: List of guardrail names to remove (from inherited guardrails)
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
    A policy that defines which guardrails apply to requests matching its scope.

    Policies can inherit from other policies using the `inherit` field.
    When inheriting:
    - Guardrails from `guardrails.add` are added to the inherited guardrails
    - Guardrails from `guardrails.remove` are removed from the inherited guardrails

    Example configuration:
    ```yaml
    policies:
      global-baseline:
        guardrails:
          add:
            - pii_blocker
            - phi_blocker
        scope:
          teams: ["*"]
          keys: ["*"]
          models: ["*"]

      healthcare-compliance:
        inherit: global-baseline
        guardrails:
          add:
            - hipaa_audit
        scope:
          teams: [healthcare-team, medical-research]
          models: [gpt-4, bedrock/claude-*]

      internal-dev:
        inherit: global-baseline
        guardrails:
          add:
            - toxicity_filter
          remove:
            - phi_blocker
        scope:
          keys: [dev-key-*, test-key-*]
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
    scope: PolicyScope = Field(
        default_factory=PolicyScope,
        description="Scope defining which requests this policy applies to.",
    )

    model_config = ConfigDict(extra="forbid")


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
