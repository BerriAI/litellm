from typing import List
from enum import Enum

from pydantic import BaseModel

"""
Pydantic object defining how to set guardrails on litellm proxy

litellm_settings:
  guardrails:
    - prompt_injection:
        callbacks: [lakera_prompt_injection, prompt_injection_api_2]
        default_on: true
        enabled_roles: [user]
    - detect_secrets:
        callbacks: [hide_secrets]
        default_on: true
"""

class Role(Enum):
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"

default_roles = [Role.SYSTEM, Role.ASSISTANT, Role.USER];

class GuardrailItem(BaseModel):
    callbacks: List[str]
    default_on: bool
    guardrail_name: str
    enabled_roles: List[Role] = default_roles
