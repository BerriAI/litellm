from pydantic import BaseModel, Field, JsonValue


class RelayClaudeCodeConfig(BaseModel):
    channel: str = Field(default="pinned")
    version: str | None = Field(default=None)
    registry: str = Field(default="npm")
    package: str = Field(default="@anthropic-ai/claude-code")
    model: str | None = Field(default=None)
    managed_settings: dict[str, JsonValue] = Field(default_factory=dict)


class RelayCodexConfig(BaseModel):
    channel: str = Field(default="pinned")
    version: str | None = Field(default=None)
    registry: str = Field(default="npm")
    package: str = Field(default="@openai/codex")
    model: str | None = Field(default=None)


class RelayManagedConfigResponse(BaseModel):
    claude_code: RelayClaudeCodeConfig = Field(default_factory=RelayClaudeCodeConfig)
    codex: RelayCodexConfig = Field(default_factory=RelayCodexConfig)
    policy_version: int | None = Field(default=None)
    updated_by: str | None = Field(default=None)
    updated_at: str | None = Field(default=None)
