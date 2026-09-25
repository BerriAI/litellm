"""
xAI Files API. Reference: https://docs.x.ai/developers/rest-api-reference/inference/files

xAI's file objects carry the OpenAI field names, so this config is URL routing and auth. The one
difference is ``purpose``: xAI stores it as an empty string, and LiteLLM reports uploads as ``batch``,
the only purpose xAI files serve today.
"""

import time
from collections.abc import Mapping, Sequence
from typing import Final

import httpx
from openai.types.file_deleted import FileDeleted
from pydantic import BaseModel, ConfigDict
from typing_extensions import ReadOnly, TypedDict

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

from ..batches.transformation import (
    get_xai_api_base,
    get_xai_auth_headers,
    raise_for_xai_status,
    xai_batches_error,
)

_NO_QUERY_PARAMS: Final[dict[str, str]] = {}  # mutable-ok: BaseFilesConfig request transforms return tuple[str, dict]
_DEFAULT_PURPOSE: Final[OpenAIFilesPurpose] = "batch"


class XAIMultipartUpload(TypedDict):
    file: ReadOnly[tuple[str, object, str]]
    purpose: ReadOnly[tuple[None, str]]


class XAIFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    bytes: int = 0
    created_at: int | None = None
    filename: str = ""
    purpose: str = ""
    expires_at: int | None = None


class XAIFileList(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    data: tuple[XAIFile, ...] = ()


class XAIFileDeleted(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    deleted: bool = True


def _to_openai_file_object(file: XAIFile) -> OpenAIFileObject:
    return OpenAIFileObject(
        id=file.id,
        bytes=file.bytes,
        created_at=file.created_at if file.created_at is not None else int(time.time()),
        filename=file.filename,
        object="file",
        purpose=_DEFAULT_PURPOSE,
        status="uploaded",
        expires_at=file.expires_at,
    )


def _api_base_from(litellm_params: Mapping[str, object]) -> str:
    api_base: Final = litellm_params.get("api_base")
    return get_xai_api_base(api_base if isinstance(api_base, str) else None)


class XAIFilesConfig(BaseFilesConfig):
    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.XAI

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        return f"{get_xai_api_base(api_base)}/v1/files"

    def _file_url(self, file_id: str, litellm_params: Mapping[str, object], suffix: str = "") -> str:
        encoded_file_id: Final = encode_url_path_segment(file_id, field_name="file_id")
        return f"{_api_base_from(litellm_params)}/v1/files/{encoded_file_id}{suffix}"

    def get_error_class(
        self, error_message: str, status_code: int, headers: Mapping[str, str] | httpx.Headers
    ) -> BaseLLMException:
        return xai_batches_error(error_message, status_code, headers)

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[object],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseFilesConfig signature
        return get_xai_auth_headers(headers, api_key)

    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAICreateFileRequestOptionalParams]:  # mutable-ok: BaseFilesConfig signature
        return ["purpose"]  # mutable-ok: BaseFilesConfig signature

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: dict[str, object],  # mutable-ok: BaseConfig signature, returned as-is
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: BaseConfig signature
        return optional_params

    def transform_create_file_request(
        self,
        model: str,
        create_file_data: CreateFileRequest,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
    ) -> dict[str, object]:  # mutable-ok: BaseFilesConfig signature
        if "file" not in create_file_data:
            raise ValueError("File data is required")
        extracted: Final = extract_file_data(create_file_data["file"])
        filename: Final = extracted["filename"] or f"file_{int(time.time())}.jsonl"
        content_type: Final = extracted.get("content_type") or "application/octet-stream"
        upload: Final = XAIMultipartUpload(
            file=(filename, extracted["content"], content_type),
            purpose=(None, create_file_data.get("purpose") or _DEFAULT_PURPOSE),
        )
        return dict(upload)  # mutable-ok: BaseFilesConfig signature

    def transform_create_file_response(
        self,
        model: str | None,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: Mapping[str, object],
    ) -> OpenAIFileObject:
        return _to_openai_file_object(XAIFile.model_validate(raise_for_xai_status(raw_response).json()))

    def transform_retrieve_file_request(
        self,
        file_id: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
    ) -> tuple[str, dict[str, str]]:  # mutable-ok: BaseFilesConfig signature
        return self._file_url(file_id, litellm_params), _NO_QUERY_PARAMS

    def transform_retrieve_file_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: Mapping[str, object],
    ) -> OpenAIFileObject:
        return _to_openai_file_object(XAIFile.model_validate(raise_for_xai_status(raw_response).json()))

    def transform_delete_file_request(
        self,
        file_id: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
    ) -> tuple[str, dict[str, str]]:  # mutable-ok: BaseFilesConfig signature
        return self._file_url(file_id, litellm_params), _NO_QUERY_PARAMS

    def transform_delete_file_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: Mapping[str, object],
    ) -> FileDeleted:
        deleted: Final = XAIFileDeleted.model_validate(raise_for_xai_status(raw_response).json())
        return FileDeleted(id=deleted.id, deleted=deleted.deleted, object="file")

    def transform_list_files_request(
        self,
        purpose: str | None,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
    ) -> tuple[str, dict[str, str]]:  # mutable-ok: BaseFilesConfig signature
        return f"{_api_base_from(litellm_params)}/v1/files", _NO_QUERY_PARAMS

    def transform_list_files_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: Mapping[str, object],
    ) -> list[OpenAIFileObject]:  # mutable-ok: BaseFilesConfig signature
        return [  # mutable-ok: BaseFilesConfig signature
            _to_openai_file_object(f)
            for f in XAIFileList.model_validate(raise_for_xai_status(raw_response).json()).data
        ]

    def transform_file_content_request(
        self,
        file_content_request: FileContentRequest,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
    ) -> tuple[str, dict[str, str]]:  # mutable-ok: BaseFilesConfig signature
        file_id: Final = file_content_request.get("file_id")
        if file_id is None:
            raise ValueError("file_id is required to download file content")
        return self._file_url(file_id, litellm_params, suffix="/content"), _NO_QUERY_PARAMS

    def transform_file_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: Mapping[str, object],
    ) -> HttpxBinaryResponseContent:
        return HttpxBinaryResponseContent(response=raw_response)
