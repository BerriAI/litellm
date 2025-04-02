# What is this?
## This hook is used to check for LiteLLM managed files in the request body, and replace them with model-specific file id

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union, cast

from fastapi import HTTPException

from litellm import verbose_logger
from litellm._logging import verbose_proxy_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_file_ids_from_messages,
)
from litellm.proxy._types import CallTypes, SpecialEnums, UserAPIKeyAuth
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIFileObject,
    OpenAIFilesPurpose,
)
from litellm.types.utils import StandardCallbackDynamicParams

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache

    Span = Union[_Span, Any]
    InternalUsageCache = _InternalUsageCache
else:
    Span = Any
    InternalUsageCache = Any


class _PROXY_LiteLLMManagedFiles(CustomLogger):
    # Class variables or attributes
    def __init__(self, internal_usage_cache: InternalUsageCache):
        self.internal_usage_cache = internal_usage_cache

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: Dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
            "pass_through_endpoint",
            "rerank",
        ],
    ) -> Union[Exception, str, Dict, None]:
        """
        - Detect litellm_proxy/ file_id
        - Update data with a specified prompt_id
        """
        pass

    def translate_managed_files(self, file_ids: List[str], model_id: str) -> List[str]:
        for file_id in file_ids:
            if file_id.startswith(SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value):
                return self.internal_usage_cache.get_cache(
                    "{}:{}".format(model_id, file_id)
                )
        return file_ids

    async def get_provider_file_id_mapping(
        self, file_ids: List[str], model: str, litellm_parent_otel_span: Span
    ) -> dict:
        """
        Get provider-specific file IDs for a list of proxy file IDs.
        Returns a dictionary mapping proxy_file_id -> provider_file_id
        """
        file_id_mapping = {}
        for file_id in file_ids:
            provider_file_id = await self.internal_usage_cache.async_get_cache(
                file_id, litellm_parent_otel_span=litellm_parent_otel_span
            )
            if provider_file_id:
                file_id_mapping[file_id] = provider_file_id
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"File ID {file_id} not found or not mapped for model {model}",
                )
        return file_id_mapping

    def get_chat_completion_prompt(
        self,
        model: str,
        messages: List[AllMessageValues],
        non_default_params: dict,
        prompt_id: str,
        prompt_variables: Optional[dict],
        dynamic_callback_params: StandardCallbackDynamicParams,
    ) -> Tuple[str, List[AllMessageValues], dict]:
        """
        Modify the message to replace the file id with the provider-specific file id
        """
        return super().get_chat_completion_prompt(
            model,
            messages,
            non_default_params,
            prompt_id,
            prompt_variables,
            dynamic_callback_params,
        )

    @staticmethod
    async def return_unified_file_id(
        file_objects: List[OpenAIFileObject],
        purpose: OpenAIFilesPurpose,
        internal_usage_cache: InternalUsageCache,
        litellm_parent_otel_span: Span,
    ) -> OpenAIFileObject:
        unified_file_id = SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value + str(
            uuid.uuid4()
        )

        ## CREATE RESPONSE OBJECT
        response = OpenAIFileObject(
            id=unified_file_id,
            object="file",
            purpose=cast(OpenAIFilesPurpose, purpose),
            created_at=file_objects[0].created_at,
            bytes=1234,
            filename=file_objects[0].filename or str(datetime.now().timestamp()),
            status="uploaded",
        )

        ## STORE RESPONSE IN DB + CACHE
        for file_object in file_objects:
            cache_key = "{}:{}".format(
                file_object._hidden_params["model_id"], file_object.id
            )
            await internal_usage_cache.async_set_cache(
                key=cache_key,
                value=file_object.id,
                litellm_parent_otel_span=litellm_parent_otel_span,
            )
        return response
