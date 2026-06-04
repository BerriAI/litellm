"""
Managed file, object, and vector store table models.

Canonical definitions for the ``litellm_managed*`` tables. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from litellm.types.llms.base import LiteLLMPydanticObjectBase
from litellm.types.llms.openai import OpenAIFileObject, ResponsesAPIResponse
from litellm.types.utils import LiteLLMBatch, LiteLLMFineTuningJob


class LiteLLM_ManagedFileTable(LiteLLMPydanticObjectBase):
    unified_file_id: str
    file_object: Optional[OpenAIFileObject] = None
    model_mappings: Dict[str, str]
    flat_model_file_ids: List[str]
    created_by: Optional[str] = None
    team_id: Optional[str] = None
    updated_by: Optional[str] = None
    storage_backend: Optional[str] = None
    storage_url: Optional[str] = None


class LiteLLM_ManagedObjectTable(LiteLLMPydanticObjectBase):
    unified_object_id: str
    model_object_id: str
    file_purpose: Literal["batch", "fine-tune", "response", "container"]
    file_object: Union[LiteLLMBatch, LiteLLMFineTuningJob, ResponsesAPIResponse]
    created_by: Optional[str] = None
    team_id: Optional[str] = None


class LiteLLM_ManagedVectorStoreTable(LiteLLMPydanticObjectBase):
    """Table for managing vector stores with target_model_names support."""

    unified_resource_id: str
    resource_object: Optional[Any] = None
    model_mappings: Dict[str, str]
    flat_model_resource_ids: List[str]
    created_by: Optional[str] = None
    team_id: Optional[str] = None
    updated_by: Optional[str] = None
    storage_backend: Optional[str] = None
    storage_url: Optional[str] = None


class LiteLLM_ManagedVectorStoresTable(LiteLLMPydanticObjectBase):
    vector_store_id: str
    custom_llm_provider: str
    vector_store_name: Optional[str]
    vector_store_description: Optional[str]
    vector_store_metadata: Optional[Dict[str, Any]]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    litellm_credential_name: Optional[str]
    litellm_params: Optional[Dict[str, Any]]
    team_id: Optional[str]
    user_id: Optional[str]
