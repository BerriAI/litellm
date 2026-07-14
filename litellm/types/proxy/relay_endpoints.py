from typing import Dict, Optional

from pydantic import BaseModel, Field, JsonValue


class RelayClaudeCodeConfig(BaseModel):
    channel: str = Field(default="pinned")
    version: Optional[str] = Field(default=None)
    registry: str = Field(default="npm")
    package: str = Field(default="@anthropic-ai/claude-code")
    model: Optional[str] = Field(default=None)
    managed_settings: Dict[str, JsonValue] = Field(default_factory=dict)


class RelayManagedConfigResponse(BaseModel):
    claude_code: RelayClaudeCodeConfig = Field(default_factory=RelayClaudeCodeConfig)
    policy_version: Optional[int] = Field(default=None)
    updated_by: Optional[str] = Field(default=None)
    updated_at: Optional[str] = Field(default=None)
