"""
Mistral Files API. Reference: https://docs.mistral.ai/api/#tag/files

Mistral's file objects already carry the OpenAI field names (id, bytes, created_at,
filename, purpose), so this config is URL routing, auth, and a purpose mapping:
Mistral only accepts ``fine-tune``, ``batch`` and ``ocr`` as upload purposes.
"""

import time
from typing import Final, Literal

import httpx
from openai.types.file_deleted import FileDeleted
from pydantic import BaseModel, ConfigDict

from litellm.litellm_core_utils.prompt_templates.common_utils import extract_file_data
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.files.transformation import BaseFilesConfig, LiteLLMLoggingObj
from litellm.types.llms.openai import (
    CreateFileRequest,
    FileContentRequest,
    HttpxBinaryResponseContent,
    OpenAICreateFileRequestOptionalParams,
    OpenAIFileObject,
    OpenAIFilesPurpose,
)
from litellm.types.utils import LlmProviders

from ..common_utils import get_mistral_api_base, get_mistral_auth_headers, mistral_error

MistralFilePurpose = Literal["fine-tune", "batch", "ocr"]


class MistralFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    bytes: int = 0
    created_at: int | None = None
    filename: str = ""
    purpose: MistralFilePurpose = "batch"
    expires_at: int | None = None


class MistralFileList(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    data: tuple[MistralFile, ...] = ()


class MistralFileDeleted(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    deleted: bool = True


def _to_openai_file_object(file: MistralFile) -> OpenAIFileObject:
    return OpenAIFileObject(
        id=file.id,
        bytes=file.bytes,
        created_at=file.created_at if file.created_at is not None else int(time.time()),
        filename=file.filename,
        object="file",
        purpose=_to_openai_purpose(file.purpose),
        status="uploaded",
        expires_at=file.expires_at,
    )


def _to_openai_purpose(purpose: MistralFilePurpose) -> OpenAIFilesPurpose:
    match purpose:
        case "fine-tune" | "batch":
            return purpose
        case "ocr":
            return "user_data"


def _to_mistral_purpose(purpose: str) -> MistralFilePurpose:
    match purpose:
        case "fine-tune" | "ocr":
            return purpose
        case _:
            return "batch"


class MistralFilesConfig(BaseFilesConfig):
    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.MISTRAL

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        return f"{get_mistral_api_base(api_base)}/v1/files"

    def _file_url(self, file_id: str, litellm_params: dict, suffix: str = "") -> str:
        encoded_file_id: Final = encode_url_path_segment(file_id, field_name="file_id")
        return f"{get_mistral_api_base(litellm_params.get('api_base'))}/v1/files/{encoded_file_id}{suffix}"

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return mistral_error(error_message, status_code, headers)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        return get_mistral_auth_headers(headers, api_key)

    def get_supported_openai_params(self, model: str) -> list[OpenAICreateFileRequestOptionalParams]:
        return ["purpose"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        return optional_params

    def transform_create_file_request(
        self,
        model: str,
        create_file_data: CreateFileRequest,
        optional_params: dict,
        litellm_params: dict,
    ) -> dict:
        file_data: Final = create_file_data.get("file")
        if file_data is None:
            raise ValueError("File data is required")
        extracted: Final = extract_file_data(file_data)
        filename: Final = extracted["filename"] or f"file_{int(time.time())}.jsonl"
        content_type: Final = extracted.get("content_type") or "application/octet-stream"
        return {
            "file": (filename, extracted["content"], content_type),
            "purpose": (None, _to_mistral_purpose(create_file_data.get("purpose", "batch"))),
        }

    def transform_create_file_response(
        self,
        model: str | None,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> OpenAIFileObject:
        return _to_openai_file_object(MistralFile.model_validate(raw_response.json()))

    def transform_retrieve_file_request(
        self,
        file_id: str,
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        return self._file_url(file_id, litellm_params), {}

    def transform_retrieve_file_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> OpenAIFileObject:
        return _to_openai_file_object(MistralFile.model_validate(raw_response.json()))

    def transform_delete_file_request(
        self,
        file_id: str,
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        return self._file_url(file_id, litellm_params), {}

    def transform_delete_file_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> FileDeleted:
        deleted: Final = MistralFileDeleted.model_validate(raw_response.json())
        return FileDeleted(id=deleted.id, deleted=deleted.deleted, object="file")

    def transform_list_files_request(
        self,
        purpose: str | None,
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        params: Final = {"purpose": _to_mistral_purpose(purpose)} if purpose else {}
        return f"{get_mistral_api_base(litellm_params.get('api_base'))}/v1/files", params

    def transform_list_files_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> list[OpenAIFileObject]:
        return [_to_openai_file_object(f) for f in MistralFileList.model_validate(raw_response.json()).data]

    def transform_file_content_request(
        self,
        file_content_request: FileContentRequest,
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        return self._file_url(file_content_request["file_id"], litellm_params, suffix="/content"), {}

    def transform_file_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> HttpxBinaryResponseContent:
        return HttpxBinaryResponseContent(response=raw_response)
