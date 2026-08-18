"""
Managed file, object, and vector store table models.

Canonical definitions for the ``litellm_managed*`` tables. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from datetime import datetime
from typing import Any, Literal

from litellm.types.llms.base import LiteLLMPydanticObjectBase
from litellm.types.llms.openai import OpenAIFileObject, ResponsesAPIResponse
from litellm.types.utils import LiteLLMBatch, LiteLLMFineTuningJob


class LiteLLM_ManagedFileTable(LiteLLMPydanticObjectBase):
    unified_file_id: str
    file_object: OpenAIFileObject | None = None
    model_mappings: dict[str, str]
    flat_model_file_ids: list[str]
    created_by: str | None = None
    team_id: str | None = None
    updated_by: str | None = None
    storage_backend: str | None = None
    storage_url: str | None = None


class LiteLLM_ManagedObjectTable(LiteLLMPydanticObjectBase):
    unified_object_id: str
    model_object_id: str
    file_purpose: Literal["batch", "fine-tune", "response", "container"]
    file_object: LiteLLMBatch | LiteLLMFineTuningJob | ResponsesAPIResponse
    created_by: str | None = None
    team_id: str | None = None


class LiteLLM_ManagedVectorStoreTable(LiteLLMPydanticObjectBase):
    """Table for managing vector stores with target_model_names support."""

    unified_resource_id: str
    resource_object: Any | None = None
    model_mappings: dict[str, str]
    flat_model_resource_ids: list[str]
    created_by: str | None = None
    team_id: str | None = None
    updated_by: str | None = None
    storage_backend: str | None = None
    storage_url: str | None = None


class LiteLLM_ManagedVectorStoresTable(LiteLLMPydanticObjectBase):
    vector_store_id: str
    custom_llm_provider: str
    vector_store_name: str | None = None
    vector_store_description: str | None = None
    vector_store_metadata: dict[str, Any] | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    litellm_credential_name: str | None = None
    litellm_params: dict[str, Any] | None = None
    team_id: str | None = None
    user_id: str | None = None
