# What is this?
## This hook is used to check for LiteLLM managed files in the request body, and replace them with model-specific file id

import asyncio
import base64
import json
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union, cast

from fastapi import HTTPException

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
    convert_b64_uid_to_unified_uid,
    get_batch_id_from_unified_batch_id,
    get_model_id_from_unified_batch_id,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    AsyncCursorPage,
    ChatCompletionFileObject,
    CreateFileRequest,
    FileObject,
    OpenAIFileObject,
    OpenAIFilesPurpose,
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

        result = await self.prisma_client.db.litellm_managedfiletable.create(
            data=db_data
        )
        verbose_logger.debug(
            f"LiteLLM Managed File object with id={file_id} stored in db: {result}"
        )

    async def store_unified_object_id(
        self,
        unified_object_id: str,
        file_object: Union[LiteLLMBatch, LiteLLMFineTuningJob],
        litellm_parent_otel_span: Optional[Span],
        model_object_id: str,
        file_purpose: Literal["batch", "fine-tune"],
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
                "update": {},  # don't do anything if it already exists
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
        return False

    async def can_user_call_unified_object_id(
        self, unified_object_id: str, user_api_key_dict: UserAPIKeyAuth
    ) -> bool:
        ## check if the user has access to the unified object id
        ## check if the user has access to the unified object id
        user_id = user_api_key_dict.user_id
        managed_object = (
            await self.prisma_client.db.litellm_managedobjecttable.find_first(
                where={"unified_object_id": unified_object_id}
            )
        )

        if managed_object:
            return managed_object.created_by == user_id
        return True  # don't raise error if managed object is not found

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

    async def async_pre_call_hook(
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
        ):
            await self.check_managed_file_id_access(data, user_api_key_dict)

        ### HANDLE TRANSFORMATIONS ###
        if call_type == CallTypes.completion.value:
            messages = data.get("messages")
            if messages:
                file_ids = self.get_file_ids_from_messages(messages)
                if file_ids:
                    model_file_id_mapping = await self.get_model_file_id_mapping(
                        file_ids, user_api_key_dict.parent_otel_span
                    )

                    data["model_file_id_mapping"] = model_file_id_mapping
        elif call_type == CallTypes.aresponses.value or call_type == CallTypes.responses.value:
            # Handle managed files in responses API input
            input_data = data.get("input")
            if input_data:
                file_ids = self.get_file_ids_from_responses_input(input_data)
                if file_ids:
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
            or call_type == CallTypes.acancel_fine_tuning_job.value
            or call_type == CallTypes.aretrieve_fine_tuning_job.value
        ):
            accessor_key: Optional[str] = None
            retrieve_object_id: Optional[str] = None
            if call_type == CallTypes.aretrieve_batch.value:
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

                if (
                    response.output_file_id and model_id
                ):  # return a file id with the model_id and output_file_id
                    original_output_file_id = response.output_file_id
                    response.output_file_id = self.get_unified_output_file_id(
                        output_file_id=response.output_file_id,
                        model_id=model_id,
                        model_name=model_name,
                    )
                    await self.store_unified_file_id(  # need to store otherwise any retrieve call will fail
                        file_id=response.output_file_id,
                        file_object=None,
                        litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
                        model_mappings={model_id: original_output_file_id},
                        user_api_key_dict=user_api_key_dict,
                    )
            asyncio.create_task(
                self.store_unified_object_id(
                    unified_object_id=response.id,
                    file_object=response,
                    litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
                    model_object_id=original_response_id,
                    file_purpose="batch",
                    user_api_key_dict=user_api_key_dict,
                )
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
            asyncio.create_task(
                self.store_unified_object_id(
                    unified_object_id=response.id,
                    file_object=response,
                    litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
                    model_object_id=original_response_id,
                    file_purpose="fine-tune",
                    user_api_key_dict=user_api_key_dict,
                )
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
        self, file_id: str, litellm_parent_otel_span: Optional[Span]
    ) -> OpenAIFileObject:
        stored_file_object = await self.get_unified_file_id(
            file_id, litellm_parent_otel_span
        )
        if stored_file_object:
            return stored_file_object.file_object
        else:
            raise Exception(f"LiteLLM Managed File object with id={file_id} not found")

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

        specific_model_file_id_mapping = model_file_id_mapping.get(file_id)
        if specific_model_file_id_mapping:
            for model_id, model_file_id in specific_model_file_id_mapping.items():
                await llm_router.afile_delete(model=model_id, file_id=model_file_id, **data)  # type: ignore

        stored_file_object = await self.delete_unified_file_id(
            file_id, litellm_parent_otel_span
        )

        if stored_file_object:
            return stored_file_object
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
