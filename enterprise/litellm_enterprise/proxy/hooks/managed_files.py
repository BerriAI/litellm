# What is this?
## This hook is used to check for LiteLLM managed files in the request body, and replace them with model-specific file id

import asyncio
import base64
import json
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union, cast

from fastapi import HTTPException

import litellm
from litellm import Router, verbose_logger
from litellm._uuid import uuid
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.prompt_templates.common_utils import extract_file_data
from litellm.llms.base_llm.files.transformation import BaseFileEndpoints
from litellm.proxy._types import (
    CallTypes,
    LiteLLM_ManagedFileTable,
    LiteLLM_ManagedObjectTable,
    UserAPIKeyAuth,
)
from litellm.proxy.openai_files_endpoints.common_utils import (
    _is_base64_encoded_unified_file_id,
    get_batch_id_from_unified_batch_id,
    get_content_type_from_file_object,
    get_model_id_from_unified_batch_id,
    normalize_mime_type_for_provider,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    AsyncCursorPage,
    ChatCompletionFileObject,
    CreateFileRequest,
    FileObject,
    OpenAIFileObject,
    OpenAIFilesPurpose,
    ResponsesAPIResponse,
)
from litellm.types.utils import (
    CallTypesLiteral,
    LiteLLMBatch,
    LiteLLMFineTuningJob,
    LLMResponseTypes,
    SpecialEnums,
)

if TYPE_CHECKING:
    from litellm.types.llms.openai import HttpxBinaryResponseContent


if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache
    from litellm.proxy.utils import PrismaClient as _PrismaClient

    Span = Union[_Span, Any]
    InternalUsageCache = _InternalUsageCache
    PrismaClient = _PrismaClient
else:
    Span = Any
    InternalUsageCache = Any
    PrismaClient = Any


class _PROXY_LiteLLMManagedFiles(CustomLogger, BaseFileEndpoints):
    # Class variables or attributes
    def __init__(
        self, internal_usage_cache: InternalUsageCache, prisma_client: PrismaClient
    ):
        self.internal_usage_cache = internal_usage_cache
        self.prisma_client = prisma_client

    async def store_unified_file_id(
        self,
        file_id: str,
        file_object: Optional[OpenAIFileObject],
        litellm_parent_otel_span: Optional[Span],
        model_mappings: Dict[str, str],
        user_api_key_dict: UserAPIKeyAuth,
    ) -> None:
        verbose_logger.info(
            f"Storing LiteLLM Managed File object with id={file_id} in cache"
        )
        if file_object is not None:
            litellm_managed_file_object = LiteLLM_ManagedFileTable(
                unified_file_id=file_id,
                file_object=file_object,
                model_mappings=model_mappings,
                flat_model_file_ids=list(model_mappings.values()),
                created_by=user_api_key_dict.user_id,
                updated_by=user_api_key_dict.user_id,
            )
            await self.internal_usage_cache.async_set_cache(
                key=file_id,
                value=litellm_managed_file_object.model_dump(),
                litellm_parent_otel_span=litellm_parent_otel_span,
            )

        ## STORE MODEL MAPPINGS IN DB

        db_data = {
            "unified_file_id": file_id,
            "model_mappings": json.dumps(model_mappings),
            "flat_model_file_ids": list(model_mappings.values()),
            "created_by": user_api_key_dict.user_id,
            "updated_by": user_api_key_dict.user_id,
        }

        if file_object is not None:
            db_data["file_object"] = file_object.model_dump_json()
            # Extract storage metadata from hidden params if present
            hidden_params = getattr(file_object, "_hidden_params", {}) or {}
            if "storage_backend" in hidden_params:
                db_data["storage_backend"] = hidden_params["storage_backend"]
            if "storage_url" in hidden_params:
                db_data["storage_url"] = hidden_params["storage_url"]
            
            verbose_logger.debug(
                f"Storage metadata: storage_backend={db_data.get('storage_backend')}, "
                f"storage_url={db_data.get('storage_url')}"
            )

        result = await self.prisma_client.db.litellm_managedfiletable.create(
            data=db_data
        )
        verbose_logger.debug(
            f"LiteLLM Managed File object with id={file_id} stored in db: {result}"
        )

    async def store_unified_object_id(
        self,
        unified_object_id: str,
        file_object: Union[LiteLLMBatch, LiteLLMFineTuningJob, "ResponsesAPIResponse"],
        litellm_parent_otel_span: Optional[Span],
        model_object_id: str,
        file_purpose: Literal["batch", "fine-tune", "response"],
        user_api_key_dict: UserAPIKeyAuth,
    ) -> None:
        verbose_logger.info(
            f"Storing LiteLLM Managed {file_purpose} object with id={unified_object_id} in cache"
        )
        litellm_managed_object = LiteLLM_ManagedObjectTable(
            unified_object_id=unified_object_id,
            model_object_id=model_object_id,
            file_purpose=file_purpose,
            file_object=file_object,
        )
        await self.internal_usage_cache.async_set_cache(
            key=unified_object_id,
            value=litellm_managed_object.model_dump(),
            litellm_parent_otel_span=litellm_parent_otel_span,
        )

        await self.prisma_client.db.litellm_managedobjecttable.upsert(
            where={"unified_object_id": unified_object_id},
            data={
                "create": {
                    "unified_object_id": unified_object_id,
                    "file_object": file_object.model_dump_json(),
                    "model_object_id": model_object_id,
                    "file_purpose": file_purpose,
                    "created_by": user_api_key_dict.user_id,
                    "updated_by": user_api_key_dict.user_id,
                    "status": file_object.status,
                },
                "update": {
                    "file_object": file_object.model_dump_json(),
                    "status": file_object.status,
                    "updated_by": user_api_key_dict.user_id,
                },  # FIX: Update status and file_object on every operation to keep state in sync
            },
        )

    async def get_unified_file_id(
        self, file_id: str, litellm_parent_otel_span: Optional[Span] = None
    ) -> Optional[LiteLLM_ManagedFileTable]:
        ## CHECK CACHE
        result = cast(
            Optional[dict],
            await self.internal_usage_cache.async_get_cache(
                key=file_id,
                litellm_parent_otel_span=litellm_parent_otel_span,
            ),
        )

        if result:
            return LiteLLM_ManagedFileTable(**result)

        ## CHECK DB
        db_object = await self.prisma_client.db.litellm_managedfiletable.find_first(
            where={"unified_file_id": file_id}
        )

        if db_object:
            return LiteLLM_ManagedFileTable(**db_object.model_dump())
        return None

    async def delete_unified_file_id(
        self, file_id: str, litellm_parent_otel_span: Optional[Span] = None
    ) -> OpenAIFileObject:
        ## get old value
        initial_value = await self.prisma_client.db.litellm_managedfiletable.find_first(
            where={"unified_file_id": file_id}
        )
        if initial_value is None:
            raise Exception(f"LiteLLM Managed File object with id={file_id} not found")
        ## delete old value
        await self.internal_usage_cache.async_set_cache(
            key=file_id,
            value=None,
            litellm_parent_otel_span=litellm_parent_otel_span,
        )
        await self.prisma_client.db.litellm_managedfiletable.delete(
            where={"unified_file_id": file_id}
        )
        return initial_value.file_object

    async def can_user_call_unified_file_id(
        self, unified_file_id: str, user_api_key_dict: UserAPIKeyAuth
    ) -> bool:
        ## check if the user has access to the unified file id

        user_id = user_api_key_dict.user_id
        managed_file = await self.prisma_client.db.litellm_managedfiletable.find_first(
            where={"unified_file_id": unified_file_id}
        )

        if managed_file:
            return managed_file.created_by == user_id
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {unified_file_id}",
        )

    async def can_user_call_unified_object_id(
        self, unified_object_id: str, user_api_key_dict: UserAPIKeyAuth
    ) -> bool:
        ## check if the user has access to the unified object id
        user_id = user_api_key_dict.user_id
        managed_object = (
            await self.prisma_client.db.litellm_managedobjecttable.find_first(
                where={"unified_object_id": unified_object_id}
            )
        )

        if managed_object:
            return managed_object.created_by == user_id
        raise HTTPException(
            status_code=404,
            detail=f"Object not found: {unified_object_id}",
        )

    async def list_user_batches(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        limit: Optional[int] = None,
        after: Optional[str] = None,
        provider: Optional[str] = None,
        target_model_names: Optional[str] = None,
        llm_router: Optional[Router] = None,
    ) -> Dict[str, Any]:
        # Provider filtering is not supported for managed batches
        # This is because the encoded object ids stored in the managed objects table do not contain the provider information
        # To support provider filtering, we would need to store the provider information in the encoded object ids
        if provider:
            raise Exception(
                "Filtering by 'provider' is not supported when using managed batches."
            )

        # Model name filtering is not supported for managed batches
        # This is because the encoded object ids stored in the managed objects table do not contain the model name
        # A hash of the model name + litellm_params for the model name is encoded as the model id. This is not sufficient to reliably map the target model names to the model ids.
        if target_model_names:
            raise Exception(
                "Filtering by 'target_model_names' is not supported when using managed batches."
            )
        
        where_clause: Dict[str, Any] = {"file_purpose": "batch"}
        
        # Filter by user who created the batch
        if user_api_key_dict.user_id:
            where_clause["created_by"] = user_api_key_dict.user_id
        
        if after:
            where_clause["id"] = {"gt": after}
        
        # Fetch more than needed to allow for post-fetch filtering
        fetch_limit = limit or 20
        if target_model_names:
            # Fetch extra to account for filtering
            fetch_limit = max(fetch_limit * 3, 100)
        
        batches = await self.prisma_client.db.litellm_managedobjecttable.find_many(
            where=where_clause,
            take=fetch_limit,
            order={"created_at": "desc"},
        )
                
        batch_objects: List[LiteLLMBatch] = []
        for batch in batches:
            try:
                # Stop once we have enough after filtering
                if len(batch_objects) >= (limit or 20):
                    break

                batch_data = json.loads(batch.file_object) if isinstance(batch.file_object, str) else batch.file_object
                batch_obj = LiteLLMBatch(**batch_data)
                batch_obj.id = batch.unified_object_id
                batch_objects.append(batch_obj)

            except Exception as e:
                verbose_logger.warning(
                    f"Failed to parse batch object {batch.unified_object_id}: {e}"
                )
                continue
        
        return {
            "object": "list",
            "data": batch_objects,
            "first_id": batch_objects[0].id if batch_objects else None,
            "last_id": batch_objects[-1].id if batch_objects else None,
            "has_more": len(batch_objects) == (limit or 20),
        }

    async def get_user_created_file_ids(
        self, user_api_key_dict: UserAPIKeyAuth, model_object_ids: List[str]
    ) -> List[OpenAIFileObject]:
        """
        Get all file ids created by the user for a list of model object ids

        Returns:
         - List of OpenAIFileObject's
        """
        file_ids = await self.prisma_client.db.litellm_managedfiletable.find_many(
            where={
                "created_by": user_api_key_dict.user_id,
                "flat_model_file_ids": {"hasSome": model_object_ids},
            }
        )
        return [OpenAIFileObject(**file_object.file_object) for file_object in file_ids]

    async def check_managed_file_id_access(
        self, data: Dict, user_api_key_dict: UserAPIKeyAuth
    ) -> bool:
        retrieve_file_id = cast(Optional[str], data.get("file_id"))
        potential_file_id = (
            _is_base64_encoded_unified_file_id(retrieve_file_id)
            if retrieve_file_id
            else False
        )
        if potential_file_id and retrieve_file_id:
            if await self.can_user_call_unified_file_id(
                retrieve_file_id, user_api_key_dict
            ):
                return True
            else:
                raise HTTPException(
                    status_code=403,
                    detail=f"User {user_api_key_dict.user_id} does not have access to the file {retrieve_file_id}",
                )
        return False

    async def check_file_ids_access(
        self, file_ids: List[str], user_api_key_dict: UserAPIKeyAuth
    ) -> None:
        """
        Check if the user has access to a list of file IDs.
        Only checks managed (unified) file IDs.
        
        Args:
            file_ids: List of file IDs to check access for
            user_api_key_dict: User API key authentication details
            
        Raises:
            HTTPException: If user doesn't have access to any of the files
        """
        for file_id in file_ids:
            is_unified_file_id = _is_base64_encoded_unified_file_id(file_id)
            if is_unified_file_id:
                if not await self.can_user_call_unified_file_id(
                    file_id, user_api_key_dict
                ):
                    raise HTTPException(
                        status_code=403,
                        detail=f"User {user_api_key_dict.user_id} does not have access to the file {file_id}",
                    )

    async def async_pre_call_hook(  # noqa: PLR0915
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: Dict,
        call_type: CallTypesLiteral,
    ) -> Union[Exception, str, Dict, None]:
        """
        - Detect litellm_proxy/ file_id
        - add dictionary of mappings of litellm_proxy/ file_id -> provider_file_id => {litellm_proxy/file_id: {"model_id": id, "file_id": provider_file_id}}
        """
        ### HANDLE FILE ACCESS ###  - ensure user has access to the file
        if (
            call_type == CallTypes.afile_content.value
            or call_type == CallTypes.afile_delete.value
            or call_type == CallTypes.afile_retrieve.value
            or call_type == CallTypes.afile_content.value
        ):
            await self.check_managed_file_id_access(data, user_api_key_dict)

        ### HANDLE TRANSFORMATIONS ###
        # Check both completion and acompletion call types
        is_completion_call = (
            call_type == CallTypes.completion.value 
            or call_type == CallTypes.acompletion.value
        )
        
        if is_completion_call:
            messages = data.get("messages")
            model = data.get("model", "")
            if messages:
                file_ids = self.get_file_ids_from_messages(messages)
                if file_ids:
                    # Check user has access to all managed files
                    await self.check_file_ids_access(file_ids, user_api_key_dict)
                    
                    # Check if any files are stored in storage backends and need base64 conversion
                    # This is needed for Vertex AI/Gemini which requires base64 content
                    is_vertex_ai = model and ("vertex_ai" in model or "gemini" in model.lower())
                    if is_vertex_ai:
                        await self._convert_storage_files_to_base64(
                            messages=messages,
                            file_ids=file_ids,
                            litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
                        )
                    
                    model_file_id_mapping = await self.get_model_file_id_mapping(
                        file_ids, user_api_key_dict.parent_otel_span
                    )
                    data["model_file_id_mapping"] = model_file_id_mapping
        elif call_type == CallTypes.aresponses.value or call_type == CallTypes.responses.value:
            # Handle managed files in responses API input and tools
            file_ids = []
            
            # Extract file IDs from input parameter
            input_data = data.get("input")
            if input_data:
                file_ids.extend(self.get_file_ids_from_responses_input(input_data))
            
            # Extract file IDs from tools parameter (e.g., code_interpreter container)
            tools = data.get("tools")
            if tools:
                file_ids.extend(self.get_file_ids_from_responses_tools(tools))
            
            if file_ids:
                # Check user has access to all managed files
                await self.check_file_ids_access(file_ids, user_api_key_dict)
                
                model_file_id_mapping = await self.get_model_file_id_mapping(
                    file_ids, user_api_key_dict.parent_otel_span
                )
                data["model_file_id_mapping"] = model_file_id_mapping
        elif call_type == CallTypes.afile_content.value:
            retrieve_file_id = cast(Optional[str], data.get("file_id"))
            potential_file_id = (
                _is_base64_encoded_unified_file_id(retrieve_file_id)
                if retrieve_file_id
                else False
            )
            if potential_file_id:
                model_id = self.get_model_id_from_unified_file_id(potential_file_id)
                if model_id:
                    data["model"] = model_id
                    data["file_id"] = self.get_output_file_id_from_unified_file_id(
                        potential_file_id
                    )
        elif call_type == CallTypes.acreate_batch.value:
            input_file_id = cast(Optional[str], data.get("input_file_id"))
            if input_file_id:
                model_file_id_mapping = await self.get_model_file_id_mapping(
                    [input_file_id], user_api_key_dict.parent_otel_span
                )

                data["model_file_id_mapping"] = model_file_id_mapping
        elif (
            call_type == CallTypes.aretrieve_batch.value
            or call_type == CallTypes.acancel_batch.value
            or call_type == CallTypes.acancel_fine_tuning_job.value
            or call_type == CallTypes.aretrieve_fine_tuning_job.value
        ):
            accessor_key: Optional[str] = None
            retrieve_object_id: Optional[str] = None
            if (
                call_type == CallTypes.aretrieve_batch.value
                or call_type == CallTypes.acancel_batch.value
            ):
                accessor_key = "batch_id"
            elif (
                call_type == CallTypes.acancel_fine_tuning_job.value
                or call_type == CallTypes.aretrieve_fine_tuning_job.value
            ):
                accessor_key = "fine_tuning_job_id"

            if accessor_key:
                retrieve_object_id = cast(Optional[str], data.get(accessor_key))

            potential_llm_object_id = (
                _is_base64_encoded_unified_file_id(retrieve_object_id)
                if retrieve_object_id
                else False
            )
            if potential_llm_object_id and retrieve_object_id:
                ## VALIDATE USER HAS ACCESS TO THE OBJECT ##
                if not await self.can_user_call_unified_object_id(
                    retrieve_object_id, user_api_key_dict
                ):
                    raise HTTPException(
                        status_code=403,
                        detail=f"User {user_api_key_dict.user_id} does not have access to the object {retrieve_object_id}",
                    )

                ## for managed batch id - get the model id
                potential_model_id = get_model_id_from_unified_batch_id(
                    potential_llm_object_id
                )
                if potential_model_id is None:
                    raise Exception(
                        f"LiteLLM Managed {accessor_key} with id={retrieve_object_id} is invalid - does not contain encoded model_id."
                    )
                data["model"] = potential_model_id
                data[accessor_key] = get_batch_id_from_unified_batch_id(
                    potential_llm_object_id
                )
        elif call_type == CallTypes.acreate_fine_tuning_job.value:
            input_file_id = cast(Optional[str], data.get("training_file"))
            if input_file_id:
                model_file_id_mapping = await self.get_model_file_id_mapping(
                    [input_file_id], user_api_key_dict.parent_otel_span
                )

        return data

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: List,
        messages: Optional[List[AllMessageValues]],
        request_kwargs: Optional[Dict] = None,
        parent_otel_span: Optional[Span] = None,
    ) -> List[Dict]:
        if request_kwargs is None:
            return healthy_deployments

        input_file_id = cast(Optional[str], request_kwargs.get("input_file_id"))
        model_file_id_mapping = cast(
            Optional[Dict[str, Dict[str, str]]],
            request_kwargs.get("model_file_id_mapping"),
        )
        allowed_model_ids = []
        if input_file_id and model_file_id_mapping:
            model_id_dict = model_file_id_mapping.get(input_file_id, {})
            allowed_model_ids = list(model_id_dict.keys())

        if len(allowed_model_ids) == 0:
            return healthy_deployments

        return [
            deployment
            for deployment in healthy_deployments
            if deployment.get("model_info", {}).get("id") in allowed_model_ids
        ]

    async def async_pre_call_deployment_hook(
        self, kwargs: Dict[str, Any], call_type: Optional[CallTypes]
    ) -> Optional[dict]:
        """
        Allow modifying the request just before it's sent to the deployment.
        """
        accessor_key: Optional[str] = None
        if call_type and call_type == CallTypes.acreate_batch:
            accessor_key = "input_file_id"
        elif call_type and call_type == CallTypes.acreate_fine_tuning_job:
            accessor_key = "training_file"
        else:
            return kwargs

        if accessor_key:
            input_file_id = cast(Optional[str], kwargs.get(accessor_key))
            model_file_id_mapping = cast(
                Optional[Dict[str, Dict[str, str]]], kwargs.get("model_file_id_mapping")
            )
            model_id = cast(Optional[str], kwargs.get("model_info", {}).get("id", None))
            mapped_file_id: Optional[str] = None
            if input_file_id and model_file_id_mapping and model_id:
                mapped_file_id = model_file_id_mapping.get(input_file_id, {}).get(
                    model_id, None
                )
            if mapped_file_id:
                kwargs[accessor_key] = mapped_file_id

        return kwargs

    def get_file_ids_from_messages(self, messages: List[AllMessageValues]) -> List[str]:
        """
        Gets file ids from messages
        """
        file_ids = []
        for message in messages:
            if message.get("role") == "user":
                content = message.get("content")
                if content:
                    if isinstance(content, str):
                        continue
                    for c in content:
                        if c["type"] == "file":
                            file_object = cast(ChatCompletionFileObject, c)
                            file_object_file_field = file_object["file"]
                            file_id = file_object_file_field.get("file_id")
                            if file_id:
                                file_ids.append(file_id)
        return file_ids

    def get_file_ids_from_responses_input(
        self, input: Union[str, List[Dict[str, Any]]]
    ) -> List[str]:
        """
        Gets file ids from responses API input.
        
        The input can be:
        - A string (no files)
        - A list of input items, where each item can have:
          - type: "input_file" with file_id
          - content: a list that can contain items with type: "input_file" and file_id
        """
        file_ids: List[str] = []
        
        if isinstance(input, str):
            return file_ids
        
        if not isinstance(input, list):
            return file_ids
        
        for item in input:
            if not isinstance(item, dict):
                continue
            
            # Check for direct input_file type
            if item.get("type") == "input_file":
                file_id = item.get("file_id")
                if file_id:
                    file_ids.append(file_id)
            
            # Check for input_file in content array
            content = item.get("content")
            if isinstance(content, list):
                for content_item in content:
                    if isinstance(content_item, dict) and content_item.get("type") == "input_file":
                        file_id = content_item.get("file_id")
                        if file_id:
                            file_ids.append(file_id)
        
        return file_ids

    def get_file_ids_from_responses_tools(
        self, tools: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Gets file ids from responses API tools parameter.
        
        The tools can contain code_interpreter with container.file_ids:
        [
            {
                "type": "code_interpreter",
                "container": {"type": "auto", "file_ids": ["file-123", "file-456"]}
            }
        ]
        """
        file_ids: List[str] = []
        
        if not isinstance(tools, list):
            return file_ids
        
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            
            # Check for code_interpreter with container file_ids
            if tool.get("type") == "code_interpreter":
                container = tool.get("container")
                if isinstance(container, dict):
                    container_file_ids = container.get("file_ids")
                    if isinstance(container_file_ids, list):
                        for file_id in container_file_ids:
                            if isinstance(file_id, str):
                                file_ids.append(file_id)
        
        return file_ids

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
            is_base64_unified_file_id = _is_base64_encoded_unified_file_id(file_id)
            if is_base64_unified_file_id:
                litellm_managed_file_ids.append(file_id)

        if litellm_managed_file_ids:
            # Get all cache keys matching the pattern file_id:*
            for file_id in litellm_managed_file_ids:
                # Search for any cache key starting with this file_id
                unified_file_object = await self.get_unified_file_id(
                    file_id, litellm_parent_otel_span
                )

                if unified_file_object:
                    file_id_mapping[file_id] = unified_file_object.model_mappings

        return file_id_mapping

    async def create_file_for_each_model(
        self,
        llm_router: Optional[Router],
        _create_file_request: CreateFileRequest,
        target_model_names_list: List[str],
        litellm_parent_otel_span: Span,
    ) -> List[OpenAIFileObject]:
        if llm_router is None:
            raise Exception("LLM Router not initialized. Ensure models added to proxy.")
        responses = []
        for model in target_model_names_list:
            individual_response = await llm_router.acreate_file(
                model=model, **_create_file_request
            )
            responses.append(individual_response)

        return responses

    async def acreate_file(
        self,
        create_file_request: CreateFileRequest,
        llm_router: Router,
        target_model_names_list: List[str],
        litellm_parent_otel_span: Span,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> OpenAIFileObject:
        responses = await self.create_file_for_each_model(
            llm_router=llm_router,
            _create_file_request=create_file_request,
            target_model_names_list=target_model_names_list,
            litellm_parent_otel_span=litellm_parent_otel_span,
        )
        response = await _PROXY_LiteLLMManagedFiles.return_unified_file_id(
            file_objects=responses,
            create_file_request=create_file_request,
            internal_usage_cache=self.internal_usage_cache,
            litellm_parent_otel_span=litellm_parent_otel_span,
            target_model_names_list=target_model_names_list,
        )

        ## STORE MODEL MAPPINGS IN DB
        model_mappings: Dict[str, str] = {}

        for file_object in responses:
            model_file_id_mapping = file_object._hidden_params.get(
                "model_file_id_mapping"
            )
            if model_file_id_mapping and isinstance(model_file_id_mapping, dict):
                model_mappings.update(model_file_id_mapping)

        await self.store_unified_file_id(
            file_id=response.id,
            file_object=response,
            litellm_parent_otel_span=litellm_parent_otel_span,
            model_mappings=model_mappings,
            user_api_key_dict=user_api_key_dict,
        )
        return response

    @staticmethod
    async def return_unified_file_id(
        file_objects: List[OpenAIFileObject],
        create_file_request: CreateFileRequest,
        internal_usage_cache: InternalUsageCache,
        litellm_parent_otel_span: Span,
        target_model_names_list: List[str],
    ) -> OpenAIFileObject:
        ## GET THE FILE TYPE FROM THE CREATE FILE REQUEST
        file_data = extract_file_data(create_file_request["file"])

        file_type = file_data["content_type"]

        output_file_id = file_objects[0].id
        model_id = file_objects[0]._hidden_params.get("model_id")

        unified_file_id = SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
            file_type,
            str(uuid.uuid4()),
            ",".join(target_model_names_list),
            output_file_id,
            model_id,
        )

        # Convert to URL-safe base64 and strip padding
        base64_unified_file_id = (
            base64.urlsafe_b64encode(unified_file_id.encode()).decode().rstrip("=")
        )

        ## CREATE RESPONSE OBJECT

        response = OpenAIFileObject(
            id=base64_unified_file_id,
            object="file",
            purpose=create_file_request["purpose"],
            created_at=file_objects[0].created_at,
            bytes=file_objects[0].bytes,
            filename=file_objects[0].filename,
            status="uploaded",
            expires_at=file_objects[0].expires_at,
        )

        return response

    def get_unified_generic_response_id(
        self, model_id: str, generic_response_id: str
    ) -> str:
        unified_generic_response_id = (
            SpecialEnums.LITELLM_MANAGED_GENERIC_RESPONSE_COMPLETE_STR.value.format(
                model_id, generic_response_id
            )
        )
        return (
            base64.urlsafe_b64encode(unified_generic_response_id.encode())
            .decode()
            .rstrip("=")
        )

    def get_unified_batch_id(self, batch_id: str, model_id: str) -> str:
        unified_batch_id = SpecialEnums.LITELLM_MANAGED_BATCH_COMPLETE_STR.value.format(
            model_id, batch_id
        )
        return base64.urlsafe_b64encode(unified_batch_id.encode()).decode().rstrip("=")

    def get_unified_output_file_id(
        self, output_file_id: str, model_id: str, model_name: Optional[str]
    ) -> str:
        unified_output_file_id = (
            SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
                "application/json",
                str(uuid.uuid4()),
                model_name or "",
                output_file_id,
                model_id,
            )
        )
        return (
            base64.urlsafe_b64encode(unified_output_file_id.encode())
            .decode()
            .rstrip("=")
        )

    def get_model_id_from_unified_file_id(self, file_id: str) -> str:
        return file_id.split("llm_output_file_model_id,")[1].split(";")[0]

    def get_output_file_id_from_unified_file_id(self, file_id: str) -> str:
        return file_id.split("llm_output_file_id,")[1].split(";")[0]

    async def async_post_call_success_hook(
        self, data: Dict, user_api_key_dict: UserAPIKeyAuth, response: LLMResponseTypes
    ) -> Any:
        if isinstance(response, LiteLLMBatch):
            ## Check if unified_file_id is in the response
            unified_file_id = response._hidden_params.get(
                "unified_file_id"
            )  # managed file id
            unified_batch_id = response._hidden_params.get(
                "unified_batch_id"
            )  # managed batch id
            model_id = cast(Optional[str], response._hidden_params.get("model_id"))
            model_name = cast(Optional[str], response._hidden_params.get("model_name"))
            original_response_id = response.id

            if (unified_batch_id or unified_file_id) and model_id:
                response.id = self.get_unified_batch_id(
                    batch_id=response.id, model_id=model_id
                )

                # Handle both output_file_id and error_file_id
                for file_attr in ["output_file_id", "error_file_id"]:
                    file_id_value = getattr(response, file_attr, None)
                    if file_id_value and model_id:
                        original_file_id = file_id_value
                        unified_file_id = self.get_unified_output_file_id(
                            output_file_id=original_file_id,
                            model_id=model_id,
                            model_name=model_name,
                        )
                        setattr(response, file_attr, unified_file_id)
                        
                        # Use llm_router credentials when available. Without credentials,
                        # Azure and other auth-required providers return 500/401.
                        file_object = None
                        try:
                            # Import module and use getattr for better testability with mocks
                            import litellm.proxy.proxy_server as proxy_server_module
                            _llm_router = getattr(proxy_server_module, 'llm_router', None)
                            if _llm_router is not None and model_id:
                                _creds = _llm_router.get_deployment_credentials_with_provider(model_id) or {}
                                file_object = await litellm.afile_retrieve(
                                    file_id=original_file_id,
                                    **_creds,
                                )
                            else:
                                file_object = await litellm.afile_retrieve(
                                    custom_llm_provider=model_name.split("/")[0] if model_name and "/" in model_name else "openai",
                                    file_id=original_file_id,
                                )
                            verbose_logger.debug(
                                f"Successfully retrieved file object for {file_attr}={original_file_id}"
                            )
                        except Exception as e:
                            verbose_logger.warning(
                                f"Failed to retrieve file object for {file_attr}={original_file_id}: {str(e)}. Storing with None and will fetch on-demand."
                            )
                        
                        await self.store_unified_file_id(
                            file_id=unified_file_id,
                            file_object=file_object,
                            litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
                            model_mappings={model_id: original_file_id},
                            user_api_key_dict=user_api_key_dict,
                        )
            await self.store_unified_object_id(
                unified_object_id=response.id,
                file_object=response,
                litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
                model_object_id=original_response_id,
                file_purpose="batch",
                user_api_key_dict=user_api_key_dict,
            )
        elif isinstance(response, LiteLLMFineTuningJob):
            ## Check if unified_file_id is in the response
            unified_file_id = response._hidden_params.get(
                "unified_file_id"
            )  # managed file id
            unified_finetuning_job_id = response._hidden_params.get(
                "unified_finetuning_job_id"
            )  # managed finetuning job id
            model_id = cast(Optional[str], response._hidden_params.get("model_id"))
            model_name = cast(Optional[str], response._hidden_params.get("model_name"))
            original_response_id = response.id
            if (unified_file_id or unified_finetuning_job_id) and model_id:
                response.id = self.get_unified_generic_response_id(
                    model_id=model_id, generic_response_id=response.id
                )
            await self.store_unified_object_id(
                unified_object_id=response.id,
                file_object=response,
                litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
                model_object_id=original_response_id,
                file_purpose="fine-tune",
                user_api_key_dict=user_api_key_dict,
            )
        elif isinstance(response, AsyncCursorPage):
            """
            For listing files, filter for the ones created by the user
            """
            ## check if file object
            if hasattr(response, "data") and isinstance(response.data, list):
                if all(
                    isinstance(file_object, FileObject) for file_object in response.data
                ):
                    ## Get all file id's
                    ## Check which file id's were created by the user
                    ## Filter the response to only include the files created by the user
                    ## Return the filtered response
                    file_ids = [
                        file_object.id
                        for file_object in cast(List[FileObject], response.data)  # type: ignore
                    ]
                    user_created_file_ids = await self.get_user_created_file_ids(
                        user_api_key_dict, file_ids
                    )
                    ## Filter the response to only include the files created by the user
                    response.data = user_created_file_ids  # type: ignore
                    return response
            return response
        return response

    async def afile_retrieve(
        self, file_id: str, litellm_parent_otel_span: Optional[Span], llm_router=None
    ) -> OpenAIFileObject:
        stored_file_object = await self.get_unified_file_id(
            file_id, litellm_parent_otel_span
        )

        # Case 1 : This is not a managed file
        if not stored_file_object:
            raise Exception(f"LiteLLM Managed File object with id={file_id} not found")
        
        # Case 2: Managed file and the file object exists in the database
        # The stored file_object has the raw provider ID. Replace with the unified ID
        # so callers see a consistent ID (matching Case 3 which does response.id = file_id).
        if stored_file_object and stored_file_object.file_object:
            # Use model_copy to ensure the ID update persists (Pydantic v2 compatibility)
            response = stored_file_object.file_object.model_copy(update={"id": file_id})
            return response

        # Case 3: Managed file exists in the database but not the file object (for. e.g the batch task might not have run)
        # So we fetch the file object from the provider. We deliberately do not store the result to avoid interfering with batch cost tracking code.
        if not llm_router:
            raise Exception(
                f"LiteLLM Managed File object with id={file_id} has no file_object "
                f"and llm_router is required to fetch from provider"
            )

        try:
            model_id, model_file_id = next(iter(stored_file_object.model_mappings.items()))
            credentials = llm_router.get_deployment_credentials_with_provider(model_id) or {}
            response = await litellm.afile_retrieve(file_id=model_file_id, **credentials)
            response.id = file_id  # Replace with unified ID
            return response
        except Exception as e:
            raise Exception(f"Failed to retrieve file {file_id} from provider: {str(e)}") from e

    async def afile_list(
        self,
        purpose: Optional[OpenAIFilesPurpose],
        litellm_parent_otel_span: Optional[Span],
        **data: Dict,
    ) -> List[OpenAIFileObject]:
        """Handled in files_endpoints.py"""
        return []

    async def afile_delete(
        self,
        file_id: str,
        litellm_parent_otel_span: Optional[Span],
        llm_router: Router,
        **data: Dict,
    ) -> OpenAIFileObject:

        # file_id = convert_b64_uid_to_unified_uid(file_id)
        model_file_id_mapping = await self.get_model_file_id_mapping(
            [file_id], litellm_parent_otel_span
        )

        delete_response = None
        specific_model_file_id_mapping = model_file_id_mapping.get(file_id)
        if specific_model_file_id_mapping:
            # Remove conflicting keys from data to avoid duplicate keyword arguments
            filtered_data = {k: v for k, v in data.items() if k not in ("model", "file_id")}
            for model_id, model_file_id in specific_model_file_id_mapping.items():
                delete_response = await llm_router.afile_delete(model=model_id, file_id=model_file_id, **filtered_data)  # type: ignore

        stored_file_object = await self.delete_unified_file_id(
            file_id, litellm_parent_otel_span
        )

        if stored_file_object:
            return stored_file_object
        elif delete_response:
            delete_response.id = file_id
            return delete_response
        else:
            raise Exception(f"LiteLLM Managed File object with id={file_id} not found")

    async def afile_content(
        self,
        file_id: str,
        litellm_parent_otel_span: Optional[Span],
        llm_router: Router,
        **data: Dict,
    ) -> "HttpxBinaryResponseContent":
        """
        Get the content of a file from first model that has it
        """
        model_file_id_mapping = data.pop("model_file_id_mapping", None)
        model_file_id_mapping = (
            model_file_id_mapping
            or await self.get_model_file_id_mapping([file_id], litellm_parent_otel_span)
        )

        specific_model_file_id_mapping = model_file_id_mapping.get(file_id)

        if specific_model_file_id_mapping:
            exception_dict = {}
            for model_id, file_id in specific_model_file_id_mapping.items():
                try:
                    return await llm_router.afile_content(model=model_id, file_id=file_id, **data)  # type: ignore
                except Exception as e:
                    exception_dict[model_id] = str(e)
            raise Exception(
                f"LiteLLM Managed File object with id={file_id} not found. Checked model id's: {specific_model_file_id_mapping.keys()}. Errors: {exception_dict}"
            )
        else:
            raise Exception(f"LiteLLM Managed File object with id={file_id} not found")

    async def _convert_storage_files_to_base64(
        self,
        messages: List[AllMessageValues],
        file_ids: List[str],
        litellm_parent_otel_span: Optional[Span],
    ) -> None:
        """
        Convert files stored in storage backends to base64 format for Vertex AI/Gemini.
        
        This method checks if any managed files are stored in storage backends,
        downloads them, and converts them to base64 format in the messages.
        """
        # Check each file_id to see if it's stored in a storage backend
        for file_id in file_ids:
            # Check if this is a base64 encoded unified file ID
            decoded_unified_file_id = _is_base64_encoded_unified_file_id(file_id)
            
            if not decoded_unified_file_id:
                continue
            
            # Check database for storage backend info
            # IMPORTANT: The database stores the base64 encoded unified_file_id (not the decoded version)
            # So we query with the original file_id (which is base64 encoded)
            db_file = await self.prisma_client.db.litellm_managedfiletable.find_first(
                where={"unified_file_id": file_id}
            )
            
            if not db_file or not db_file.storage_backend or not db_file.storage_url:
                continue
            
            # File is stored in a storage backend, download and convert to base64
            try:
                from litellm.llms.base_llm.files.storage_backend_factory import (
                    get_storage_backend,
                )
                
                storage_backend_name = db_file.storage_backend
                storage_url = db_file.storage_url
                
                # Get storage backend (uses same env vars as callback)
                try:
                    storage_backend = get_storage_backend(storage_backend_name)
                except ValueError as e:
                    verbose_logger.warning(
                        f"Storage backend '{storage_backend_name}' error for file {file_id}: {str(e)}"
                    )
                    continue
                
                file_content = await storage_backend.download_file(storage_url)
                
                # Determine content type from file object
                content_type = self._get_content_type_from_file_object(db_file.file_object)
                
                # Convert to base64
                base64_data = base64.b64encode(file_content).decode("utf-8")
                base64_data_uri = f"data:{content_type};base64,{base64_data}"
                
                # Update messages to use base64 instead of file_id
                self._update_messages_with_base64_data(messages, file_id, base64_data_uri, content_type)
            except Exception as e:
                verbose_logger.exception(
                    f"Error converting file {file_id} from storage backend to base64: {str(e)}"
                )
                # Continue with other files even if one fails
                continue

    def _get_content_type_from_file_object(self, file_object: Optional[Any]) -> str:
        """
        Determine content type from file object.
        
        Uses the MIME type utility for consistent detection and normalization.
        
        Args:
            file_object: The file object from the database (can be dict, JSON string, or None)
        
        Returns:
            str: MIME type (defaults to "application/octet-stream" if cannot be determined)
        """
        # Use utility function for detection
        content_type = get_content_type_from_file_object(file_object)
        
        # Normalize for Gemini/Vertex AI (requires image/jpeg, not image/jpg)
        content_type = normalize_mime_type_for_provider(content_type, provider="gemini")
        
        return content_type

    def _update_messages_with_base64_data(
        self,
        messages: List[AllMessageValues],
        file_id: str,
        base64_data_uri: str,
        content_type: str,
    ) -> None:
        """
        Update messages to replace file_id with base64 data URI.
        
        Args:
            messages: List of messages to update
            file_id: The file ID to replace
            base64_data_uri: The base64 data URI to use as replacement
            content_type: The MIME type of the file (e.g., "image/jpeg", "application/pdf")
        """
        for message in messages:
            if message.get("role") == "user":
                content = message.get("content")
                if content and isinstance(content, list):
                    for element in content:
                        if element.get("type") == "file":
                            file_element = cast(ChatCompletionFileObject, element)
                            file_element_file = file_element.get("file", {})
                            
                            if file_element_file.get("file_id") == file_id:
                                # Replace file_id with base64 data
                                file_element_file["file_data"] = base64_data_uri
                                # Set format to help Gemini determine mime type
                                file_element_file["format"] = content_type
                                # Remove file_id to ensure only file_data is used
                                file_element_file.pop("file_id", None)
                                
                                verbose_logger.debug(
                                    f"Converted file {file_id} from storage backend to base64 with format {content_type}"
                                )
