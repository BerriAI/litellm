from enum import Enum
from typing import Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, ConfigDict
from typing_extensions import Required, TypedDict

"""
Pydantic object defining how to set guardrails on litellm proxy

litellm_settings:
  guardrails:
    - prompt_injection:
        callbacks: [lakera_prompt_injection, prompt_injection_api_2]
        default_on: true
        enabled_roles: [system, user]
    - detect_secrets:
        callbacks: [hide_secrets]
        default_on: true
"""


class Role(Enum):
    SYSTEM = "system"
    ASSISTANT = "assistant"
    USER = "user"


default_roles = [Role.SYSTEM, Role.ASSISTANT, Role.USER]


class GuardrailItemSpec(TypedDict, total=False):
    callbacks: Required[List[str]]
    default_on: bool
    logging_only: Optional[bool]
    enabled_roles: Optional[List[Role]]
    callback_args: Dict[str, Dict]


class GuardrailItem(BaseModel):
    callbacks: List[str]
    default_on: bool
    logging_only: Optional[bool]
    guardrail_name: str
    callback_args: Dict[str, Dict]
    enabled_roles: Optional[List[Role]]

    model_config = ConfigDict(use_enum_values=True)

    def __init__(
        self,
        callbacks: List[str],
        guardrail_name: str,
        default_on: bool = False,
        logging_only: Optional[bool] = None,
        enabled_roles: Optional[List[Role]] = default_roles,
        callback_args: Dict[str, Dict] = {},
    ):
        super().__init__(
            callbacks=callbacks,
            default_on=default_on,
            logging_only=logging_only,
            guardrail_name=guardrail_name,
            enabled_roles=enabled_roles,
            callback_args=callback_args,
        )


# Define the TypedDicts
class LakeraCategoryThresholds(TypedDict, total=False):
    prompt_injection: float
    jailbreak: float


class LitellmParams(TypedDict, total=False):
    guardrail: str
    mode: str
    api_key: str
    api_base: Optional[str]

    # Lakera specific params
    category_thresholds: Optional[LakeraCategoryThresholds]

    # Bedrock specific params
    guardrailIdentifier: Optional[str]
    guardrailVersion: Optional[str]


class Guardrail(TypedDict):
    guardrail_name: str
    litellm_params: LitellmParams


class guardrailConfig(TypedDict):
    guardrails: List[Guardrail]


class GuardrailEventHooks(str, Enum):
    pre_call = "pre_call"
    post_call = "post_call"
    during_call = "during_call"


class BedrockTextContent(TypedDict, total=False):
    text: str


class BedrockContentItem(TypedDict, total=False):
    text: BedrockTextContent


class BedrockRequest(TypedDict, total=False):
    source: Literal["INPUT", "OUTPUT"]
    content: List[BedrockContentItem]
