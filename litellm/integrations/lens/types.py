from datetime import datetime

from pydantic import ConfigDict, Field

from litellm.types.llms.base import LiteLLMBaseModel


class TraceAgent(LiteLLMBaseModel):
    """One agent seen in the caller's traces, for picking which agent's runs to look at."""

    model_config = ConfigDict(frozen=True)

    name: str
    runs: int = Field(ge=0)
    failed_runs: int = Field(ge=0)
    last_seen: datetime
    frameworks: tuple[str, ...] = ()


class TraceAgentList(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    agents: tuple[TraceAgent, ...]
