# What is this?
## This hook is used to check for LiteLLM managed files in the request body, and replace them with model-specific file id

import base64
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Union, cast

from litellm import verbose_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    extract_file_data,
    get_file_ids_from_messages,
)
from litellm.proxy._types import CallTypes, UserAPIKeyAuth
from litellm.types.llms.openai import (
    CreateFileRequest,
    OpenAIFileObject,
    OpenAIFilesPurpose,
)
from litellm.types.utils import SpecialEnums
from litellm.utils import is_base64_encoded

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
        - add dictionary of mappings of litellm_proxy/ file_id -> provider_file_id => {litellm_proxy/file_id: {"model_id": id, "file_id": provider_file_id}}
        """
        if call_type == CallTypes.completion.value:
            messages = data.get("messages")
            if messages:
                file_ids = get_file_ids_from_messages(messages)
                if file_ids:
                    file_ids = [
                        self.convert_b64_uid_to_unified_uid(file_id)
                        for file_id in file_ids
                    ]
                    model_file_id_mapping = await self.get_model_file_id_mapping(
                        file_ids, user_api_key_dict.parent_otel_span
                    )
                    data["model_file_id_mapping"] = model_file_id_mapping

        return data

    def _is_base64_encoded_unified_file_id(
        self, b64_uid: str
    ) -> Union[str, Literal[False]]:
        # Add padding back if needed
        padded = b64_uid + "=" * (-len(b64_uid) % 4)
        # Decode from base64
        try:
            decoded = base64.urlsafe_b64decode(padded).decode()
            if decoded.startswith(SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value):
                return decoded
            else:
                return False
        except Exception:
            return False

    def convert_b64_uid_to_unified_uid(self, b64_uid: str) -> str:
        is_base64_unified_file_id = self._is_base64_encoded_unified_file_id(b64_uid)
        if is_base64_unified_file_id:
            return is_base64_unified_file_id
        else:
            return b64_uid

    async def get_model_file_id_mapping(
        self, file_ids: List[str], litellm_parent_otel_span: Span
    ) -> dict:
        """
        Get model-specific file IDs for a list of proxy file IDs.
        Returns a dictionary mapping litellm_proxy/ file_id -> model_id -> model_file_id

        1. Get all the litellm_proxy/ file_ids from the messages
        2. For each file_id, search for cache keys matching the pattern file_id:*
        3. Return a dictionary of mappings of litellm_proxy/ file_id -> model_id -> model_file_id

        Example:
        {
            "litellm_proxy/file_id": {
                "model_id": "model_file_id"
            }
        }
        """
        file_id_mapping: Dict[str, Dict[str, str]] = {}
        litellm_managed_file_ids = []

        for file_id in file_ids:
            ## CHECK IF FILE ID IS MANAGED BY LITELM
            if file_id.startswith(SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value):
                litellm_managed_file_ids.append(file_id)

        if litellm_managed_file_ids:
            # Get all cache keys matching the pattern file_id:*
            for file_id in litellm_managed_file_ids:
                # Search for any cache key starting with this file_id
                cached_values = cast(
                    Dict[str, str],
                    await self.internal_usage_cache.async_get_cache(
                        key=file_id, litellm_parent_otel_span=litellm_parent_otel_span
                    ),
                )
                if cached_values:
                    file_id_mapping[file_id] = cached_values

        return file_id_mapping

    @staticmethod
    async def return_unified_file_id(
        file_objects: List[OpenAIFileObject],
        create_file_request: CreateFileRequest,
        purpose: OpenAIFilesPurpose,
        internal_usage_cache: InternalUsageCache,
        litellm_parent_otel_span: Span,
    ) -> OpenAIFileObject:
        ## GET THE FILE TYPE FROM THE CREATE FILE REQUEST
        file_data = extract_file_data(create_file_request["file"])

        file_type = file_data["content_type"]

        unified_file_id = SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
            file_type, str(uuid.uuid4())
        )

        # Convert to URL-safe base64 and strip padding
        base64_unified_file_id = (
            base64.urlsafe_b64encode(unified_file_id.encode()).decode().rstrip("=")
        )

        ## CREATE RESPONSE OBJECT

        response = OpenAIFileObject(
            id=base64_unified_file_id,
            object="file",
            purpose=cast(OpenAIFilesPurpose, purpose),
            created_at=file_objects[0].created_at,
            bytes=file_objects[0].bytes,
            filename=str(datetime.now().timestamp()),
            status="uploaded",
        )

        ## STORE RESPONSE IN DB + CACHE
        stored_values: Dict[str, str] = {}
        for file_object in file_objects:
            model_id = file_object._hidden_params.get("model_id")
            if model_id is None:
                verbose_logger.warning(
                    f"Skipping file_object: {file_object} because model_id in hidden_params={file_object._hidden_params} is None"
                )
                continue
            file_id = file_object.id
            stored_values[model_id] = file_id
        await internal_usage_cache.async_set_cache(
            key=unified_file_id,
            value=stored_values,
            litellm_parent_otel_span=litellm_parent_otel_span,
        )

        return response
