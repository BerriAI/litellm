from typing import Dict, List, Optional, Union

from pydantic import BaseModel, RootModel
from typing_extensions import Required, TypedDict, override

"""
Pydantic object defining how to set guardrails on litellm proxy

litellm_settings:
  guardrails:
    - prompt_injection:
        callbacks: [lakera_prompt_injection, prompt_injection_api_2]
        default_on: true
    - detect_secrets:
        callbacks: [hide_secrets]
        default_on: true
"""


class GuardrailItemSpec(TypedDict, total=False):
    callbacks: Required[List[str]]
    default_on: bool
    logging_only: Optional[bool]


class GuardrailItem(BaseModel):
    callbacks: List[str]
    default_on: bool
    logging_only: Optional[bool]
    guardrail_name: str

    def __init__(
        self,
        callbacks: List[str],
        guardrail_name: str,
        default_on: bool = False,
        logging_only: Optional[bool] = None,
    ):
        super().__init__(
            callbacks=callbacks,
            default_on=default_on,
            logging_only=logging_only,
            guardrail_name=guardrail_name,
        )
