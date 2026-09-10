"""
Mistral Files API. Reference: https://docs.mistral.ai/api/#tag/files

Mistral's file objects already carry the OpenAI field names (id, bytes, created_at,
filename, purpose), so this config is URL routing, auth, and a purpose mapping:
Mistral only accepts ``fine-tune``, ``batch`` and ``ocr`` as upload purposes.
"""

import time
from collections.abc import Mapping, Sequence
from typing import Final, Literal, TypeAlias

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

from ..common_utils import get_mistral_api_base, get_mistral_auth_headers, mistral_error

MistralFilePurpose: TypeAlias = Literal["fine-tune", "batch", "ocr"]

_NO_QUERY_PARAMS: Final[dict[str, str]] = {}  # mutable-ok: BaseFilesConfig request transforms return tuple[str, dict]


class MistralMultipartUpload(TypedDict):
    """``files=`` payload for ``POST /v1/files``: each value is an httpx multipart tuple."""

    file: ReadOnly[tuple[str, object, str]]
    purpose: ReadOnly[tuple[None, MistralFilePurpose]]


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
    """Only Mistral's own purposes pass through. Silently mapping anything else to ``batch``
    would let an upload skip the proxy's batch-file validation and guardrails, which only
    run when the caller says ``purpose=batch``."""
    match purpose:
        case "batch" | "fine-tune" | "ocr":
            return purpose
        case _:
            raise ValueError(f"Mistral does not support purpose={purpose!r}. Use one of: batch, fine-tune, ocr")


def _api_base_from(litellm_params: Mapping[str, object]) -> str:
    api_base: Final = litellm_params.get("api_base")
    return get_mistral_api_base(api_base if isinstance(api_base, str) else None)


class MistralFilesConfig(BaseFilesConfig):
    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.MISTRAL

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        return f"{get_mistral_api_base(api_base)}/v1/files"

    def _file_url(self, file_id: str, litellm_params: Mapping[str, object], suffix: str = "") -> str:
        encoded_file_id: Final = encode_url_path_segment(file_id, field_name="file_id")
        return f"{_api_base_from(litellm_params)}/v1/files/{encoded_file_id}{suffix}"

    def get_error_class(
        self, error_message: str, status_code: int, headers: Mapping[str, str] | httpx.Headers
    ) -> BaseLLMException:
        return mistral_error(error_message, status_code, headers)

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
        return get_mistral_auth_headers(headers, api_key)

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
        upload: Final = MistralMultipartUpload(
            file=(filename, extracted["content"], content_type),
            purpose=(None, _to_mistral_purpose(create_file_data.get("purpose") or "batch")),
        )
        return dict(upload)  # mutable-ok: BaseFilesConfig signature

    def transform_create_file_response(
        self,
        model: str | None,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: Mapping[str, object],
    ) -> OpenAIFileObject:
        return _to_openai_file_object(MistralFile.model_validate(raw_response.json()))

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
        return _to_openai_file_object(MistralFile.model_validate(raw_response.json()))

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
        deleted: Final = MistralFileDeleted.model_validate(raw_response.json())
        return FileDeleted(id=deleted.id, deleted=deleted.deleted, object="file")

    def transform_list_files_request(
        self,
        purpose: str | None,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
    ) -> tuple[str, dict[str, str]]:  # mutable-ok: BaseFilesConfig signature
        url: Final = f"{_api_base_from(litellm_params)}/v1/files"
        if not purpose:
            return url, _NO_QUERY_PARAMS
        return url, {"purpose": _to_mistral_purpose(purpose)}  # mutable-ok: BaseFilesConfig signature returns dict

    def transform_list_files_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: Mapping[str, object],
    ) -> list[OpenAIFileObject]:  # mutable-ok: BaseFilesConfig signature
        return [  # mutable-ok: BaseFilesConfig signature
            _to_openai_file_object(f) for f in MistralFileList.model_validate(raw_response.json()).data
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
