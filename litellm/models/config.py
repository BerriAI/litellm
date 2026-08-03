"""
Config table model.

Canonical definition for ``litellm_config``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_Config(LiteLLMPydanticObjectBase):
    param_name: str
    param_value: dict
