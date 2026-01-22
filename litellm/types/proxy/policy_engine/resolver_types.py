"""
Policy resolver type definitions.

These types are used for matching requests to policies and resolving
the final guardrails list.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class PolicyMatchContext(BaseModel):
    """
    Context used to match a request against policies.

    Contains the team alias, key alias, and model from the incoming request.
    """

    team_alias: Optional[str] = Field(
        default=None,
        description="Team alias from the request.",
    )
    key_alias: Optional[str] = Field(
        default=None,
        description="API key alias from the request.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Model name from the request.",
    )

    model_config = ConfigDict(extra="forbid")


class ResolvedPolicy(BaseModel):
    """
    Result of resolving a policy with its inheritance chain.

    Contains the final list of guardrails after applying all add/remove operations.
    """

    policy_name: str = Field(description="Name of the resolved policy.")
    guardrails: List[str] = Field(
        default_factory=list,
        description="Final list of guardrail names to apply.",
    )
    inheritance_chain: List[str] = Field(
        default_factory=list,
        description="List of policy names in the inheritance chain (from root to this policy).",
    )

    model_config = ConfigDict(extra="forbid")
