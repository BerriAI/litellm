# What is this?
## This hook is used to check for LiteLLM managed files in the request body, and replace them with model-specific file id

import base64
import json
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union, cast

from litellm import Router, verbose_logger
from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.prompt_templates.common_utils import extract_file_data
from litellm.llms.base_llm.files.transformation import BaseFileEndpoints
from litellm.proxy._types import CallTypes, LiteLLM_ManagedFileTable, UserAPIKeyAuth
from litellm.proxy.openai_files_endpoints.common_utils import (
    _is_base64_encoded_unified_file_id,
    convert_b64_uid_to_unified_uid,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionFileObject,
    CreateFileRequest,
    OpenAIFileObject,
    OpenAIFilesPurpose,
)
from litellm.types.utils import LiteLLMBatch, LLMResponseTypes, SpecialEnums

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
        file_object: OpenAIFileObject,
        litellm_parent_otel_span: Optional[Span],
        model_mappings: Dict[str, str],
    ) -> None:
        verbose_logger.info(
            f"Storing LiteLLM Managed File object with id={file_id} in cache"
        )
        litellm_managed_file_object = LiteLLM_ManagedFileTable(
            unified_file_id=file_id,
            file_object=file_object,
            model_mappings=model_mappings,
        )
        await self.internal_usage_cache.async_set_cache(
            key=file_id,
            value=litellm_managed_file_object.model_dump(),
            litellm_parent_otel_span=litellm_parent_otel_span,
        )

        await self.prisma_client.db.litellm_managedfiletable.create(
            data={
                "unified_file_id": file_id,
                "file_object": file_object.model_dump_json(),
                "model_mappings": json.dumps(model_mappings),
            }
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
            "acreate_batch",
            "aretrieve_batch",
            "afile_content",
        ],
    ) -> Union[Exception, str, Dict, None]:
        """
        - Detect litellm_proxy/ file_id
        - add dictionary of mappings of litellm_proxy/ file_id -> provider_file_id => {litellm_proxy/file_id: {"model_id": id, "file_id": provider_file_id}}
        """
        if call_type == CallTypes.completion.value:
            messages = data.get("messages")
            if messages:
                file_ids = self.get_file_ids_from_messages(messages)
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
        elif call_type == CallTypes.aretrieve_batch.value:
            retrieve_batch_id = cast(Optional[str], data.get("batch_id"))
            potential_batch_id = (
                _is_base64_encoded_unified_file_id(retrieve_batch_id)
                if retrieve_batch_id
                else False
            )
            if potential_batch_id:
                ## for managed batch id - get the model id
                potential_model_id = self.get_model_id_from_unified_batch_id(
                    potential_batch_id
                )
                if potential_model_id is None:
                    raise Exception(
                        f"LiteLLM Managed Batch ID with id={retrieve_batch_id} is invalid - does not contain encoded model_id."
                    )
                data["model"] = potential_model_id
                data["batch_id"] = self.get_batch_id_from_unified_batch_id(
                    potential_batch_id
                )

        return data

    async def async_pre_call_deployment_hook(
        self, kwargs: Dict[str, Any], call_type: Optional[CallTypes]
    ) -> Optional[dict]:
        """
        Allow modifying the request just before it's sent to the deployment.
        """
        if call_type and call_type == CallTypes.acreate_batch:
            input_file_id = cast(Optional[str], kwargs.get("input_file_id"))
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
                kwargs["input_file_id"] = mapped_file_id
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
            model_id = file_object._hidden_params.get("model_id")
            if model_id is None:
                verbose_logger.warning(
                    f"Skipping file_object: {file_object} because model_id in hidden_params={file_object._hidden_params} is None"
                )
                continue
            file_id = file_object.id
            model_mappings[model_id] = file_id

        await self.store_unified_file_id(
            file_id=response.id,
            file_object=response,
            litellm_parent_otel_span=litellm_parent_otel_span,
            model_mappings=model_mappings,
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

    def get_unified_batch_id(self, batch_id: str, model_id: str) -> str:
        unified_batch_id = SpecialEnums.LITELLM_MANAGED_BATCH_COMPLETE_STR.value.format(
            model_id, batch_id
        )
        return base64.urlsafe_b64encode(unified_batch_id.encode()).decode().rstrip("=")

    def get_unified_output_file_id(
        self, output_file_id: str, model_id: str, model_name: str
    ) -> str:
        unified_output_file_id = (
            SpecialEnums.LITELLM_MANAGED_FILE_COMPLETE_STR.value.format(
                "application/json",
                str(uuid.uuid4()),
                model_name,
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

    def get_model_id_from_unified_batch_id(self, file_id: str) -> Optional[str]:
        """
        Get the model_id from the file_id

        Expected format: litellm_proxy;model_id:{};llm_batch_id:{};llm_output_file_id:{}
        """
        ## use regex to get the model_id from the file_id
        try:
            return file_id.split("model_id:")[1].split(";")[0]
        except Exception:
            return None

    def get_batch_id_from_unified_batch_id(self, file_id: str) -> str:
        ## use regex to get the batch_id from the file_id
        return file_id.split("llm_batch_id:")[1].split(",")[0]

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
            if (unified_batch_id or unified_file_id) and model_id:
                response.id = self.get_unified_batch_id(
                    batch_id=response.id, model_id=model_id
                )

                if (
                    response.output_file_id and model_name and model_id
                ):  # return a file id with the model_id and output_file_id
                    response.output_file_id = self.get_unified_output_file_id(
                        output_file_id=response.output_file_id,
                        model_id=model_id,
                        model_name=model_name,
                    )

        return await super().async_post_call_success_hook(
            data, user_api_key_dict, response
        )

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
        return []

    async def afile_delete(
        self,
        file_id: str,
        litellm_parent_otel_span: Optional[Span],
        llm_router: Router,
        **data: Dict,
    ) -> OpenAIFileObject:
        file_id = convert_b64_uid_to_unified_uid(file_id)
        model_file_id_mapping = await self.get_model_file_id_mapping(
            [file_id], litellm_parent_otel_span
        )
        specific_model_file_id_mapping = model_file_id_mapping.get(file_id)
        if specific_model_file_id_mapping:
            for model_id, file_id in specific_model_file_id_mapping.items():
                await llm_router.afile_delete(model=model_id, file_id=file_id, **data)  # type: ignore

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
    ) -> str:
        """
        Get the content of a file from first model that has it
        """
        model_file_id_mapping = await self.get_model_file_id_mapping(
            [file_id], litellm_parent_otel_span
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
