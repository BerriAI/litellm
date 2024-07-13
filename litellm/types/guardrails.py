from typing import Dict, List, Optional, TypedDict, Union

from pydantic import BaseModel, RootModel

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


class GuardrailItem(BaseModel):
    callbacks: List[str]
    default_on: bool
    logging_only: Optional[bool]
    guardrail_name: str

    def __init__(
        self,
        callbacks: List[str],
        default_on: bool,
        guardrail_name: str,
        logging_only: Optional[bool] = None,
    ):
        super().__init__(
            callbacks=callbacks,
            default_on=default_on,
            logging_only=logging_only,
            guardrail_name=guardrail_name,
        )
