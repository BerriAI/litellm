"""Agent Skills discovery index, version 0.2.0.

Schema: https://schemas.agentskills.io/discovery/0.2.0/schema.json
"""

from typing import Final, Literal

from pydantic import Field

from litellm.types.llms.base import LiteLLMBaseModel

AGENT_SKILLS_DISCOVERY_SCHEMA_URL: Final = "https://schemas.agentskills.io/discovery/0.2.0/schema.json"
MAX_SKILL_NAME_LENGTH: Final = 64
MAX_SKILL_DESCRIPTION_LENGTH: Final = 1024


class AgentSkillsIndexEntry(LiteLLMBaseModel):
    name: str
    type: Literal["archive"]
    description: str
    url: str
    digest: str


class AgentSkillsIndex(LiteLLMBaseModel):
    discovery_schema: str = Field(default=AGENT_SKILLS_DISCOVERY_SCHEMA_URL, alias="$schema")
    skills: tuple[AgentSkillsIndexEntry, ...]
