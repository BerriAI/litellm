"""
Proxy model table model.

Canonical definition for ``litellm_proxymodeltable``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

import json
from datetime import datetime
from typing import Optional

from pydantic import ConfigDict, model_validator

from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_ProxyModelTable(LiteLLMPydanticObjectBase):
    model_id: str
    model_name: str
    litellm_params: dict
    model_info: Optional[dict] = None
    blocked: bool = False
    created_at: Optional[datetime] = None
    created_by: Optional[str] = None
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None

    model_config = ConfigDict(protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def check_potential_json_str(cls, values):
        # FastAPI re-validates the endpoint's response model with
        # `from_attributes=True` when the handler returns a Pydantic instance
        # (e.g. the Prisma `update` return in `/model/block` / `/model/unblock`).
        # In that path `values` is the model instance, not a dict, so
        # `values.get(...)` raises `AttributeError`. Pydantic documents that a
        # before-validator may receive any input:
        # https://docs.pydantic.dev/latest/concepts/validators/#model-validators
        # Returning non-dict input unchanged lets Pydantic's normal
        # attribute-based validation pick up the fields. See #35597.
        if not isinstance(values, dict):
            return values
        if isinstance(values.get("litellm_params"), str):
            try:
                values["litellm_params"] = json.loads(values["litellm_params"])
            except json.JSONDecodeError:
                pass
        if isinstance(values.get("model_info"), str):
            try:
                values["model_info"] = json.loads(values["model_info"])
            except json.JSONDecodeError:
                pass
        return values

    @property
    def is_blocked(self) -> bool:
        return self.blocked

    @property
    def team_id(self) -> Optional[str]:
        if self.model_info:
            return self.model_info.get("team_id")
        return None

    @property
    def team_public_model_name(self) -> Optional[str]:
        if self.model_info:
            return self.model_info.get("team_public_model_name")
        return None
