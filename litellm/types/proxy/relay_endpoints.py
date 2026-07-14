from typing import Dict, Optional

from pydantic import BaseModel, Field, JsonValue


class RelayClaudeCodeConfig(BaseModel):
    channel: str = Field(default="pinned")
    version: Optional[str] = Field(default=None)
    registry: str = Field(default="npm")
    package: str = Field(default="@anthropic-ai/claude-code")
    model: Optional[str] = Field(default=None)
    managed_settings: Dict[str, JsonValue] = Field(default_factory=dict)


class RelayCodexConfig(BaseModel):
    channel: str = Field(default="pinned")
    version: Optional[str] = Field(default=None)
    registry: str = Field(default="npm")
    package: str = Field(default="@openai/codex")
    model: Optional[str] = Field(default=None)


class RelayManagedConfigResponse(BaseModel):
    claude_code: RelayClaudeCodeConfig = Field(default_factory=RelayClaudeCodeConfig)
    codex: RelayCodexConfig = Field(default_factory=RelayCodexConfig)
    policy_version: Optional[int] = Field(default=None)
    updated_by: Optional[str] = Field(default=None)
    updated_at: Optional[str] = Field(default=None)
