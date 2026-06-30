import base64
import json
import os
import time
from collections.abc import Mapping, MutableMapping
from types import MappingProxyType
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from urllib.parse import unquote

import httpx
from httpx import Headers, Response
from openai.types.file_deleted import FileDeleted
from pydantic import BaseModel, ConfigDict

from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.files.utils import FilesAPIUtils
from litellm.litellm_core_utils.cloud_storage_security import (
    BEDROCK_MANAGED_S3_BATCH_PREFIX,
    BEDROCK_MANAGED_S3_PREFIXES,
    BEDROCK_MANAGED_S3_UPLOAD_PREFIX,
    build_managed_cloud_object_name,
    encode_s3_object_key_for_url,
    sanitize_cloud_object_component,
    should_allow_legacy_cloud_file_ids,
    split_configured_cloud_bucket_name,
    validate_managed_cloud_file_id,
)
from litellm.litellm_core_utils.prompt_templates.common_utils import extract_file_data
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.files.transformation import (
    BaseFilesConfig,
    LiteLLMLoggingObj,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    CreateFileRequest,
    FileContentRequest,
    FileTypes,
    HttpxBinaryResponseContent,
    OpenAICreateFileRequestOptionalParams,
    OpenAIFileObject,
    PathLike,
)
from litellm.types.utils import ExtractedFileData, LlmProviders, SpecialEnums
from litellm.utils import get_llm_provider

from ..base_aws_llm import BaseAWSLLM
from ..common_utils import BedrockError

# litellm_params key used to hand the SigV4-signed GET headers from
# `transform_file_content_request` to `validate_environment` (the only hook
# the shared file-content HTTP handler exposes for setting request headers).
# Same pattern as the `upload_url` handoff in `transform_create_file_request`.
S3_SIGNED_GET_HEADERS_PARAM = "_s3_signed_get_headers"


class _BedrockS3RequestParams(BaseModel):
    """Typed view of the credential/region params the S3 GetObject path reads."""

    model_config = ConfigDict(extra="ignore")

    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None
    aws_region_name: str | None = None
    aws_session_name: str | None = None
    aws_profile_name: str | None = None
    aws_role_name: str | None = None
    aws_web_identity_token: str | None = None
    aws_sts_endpoint: str | None = None
    s3_region_name: str | None = None
    s3_endpoint_url: str | None = None


class _TrustedS3ModelCredentials(BaseModel):
    """The S3 bucket the server trusts file ids against, from the deployment snapshot."""

    model_config = ConfigDict(extra="ignore")

    s3_bucket_name: str | None = None


def extract_s3_uri_from_file_id(file_id: str) -> str:
    """
    Resolve a Bedrock file id to its S3 URI.

    Accepts either a base64-encoded LiteLLM unified file id (whose decoded
    form carries `llm_output_file_id,s3://...`) or a direct `s3://` URI.
    """
    try:
        padded = file_id + "=" * (-len(file_id) % 4)
        decoded = base64.urlsafe_b64decode(padded).decode()

        if decoded.startswith(SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value):
            if "llm_output_file_id," in decoded:
                return decoded.split("llm_output_file_id,")[1].split(";")[0]
    except Exception:
        pass

    if file_id.startswith("s3://"):
        return file_id

    raise ValueError("file_id must be a managed LiteLLM S3 file id")


def get_configured_s3_bucket_name(litellm_params: Mapping[str, object]) -> str:
    """
    Resolve the server-configured S3 bucket for Bedrock file operations.

    Only trusts the immutable server-side credential snapshot or the
    environment; never a request-supplied param, since the bucket is what
    `validate_managed_cloud_file_id` checks file ids against.
    """
    trusted_model_credentials = litellm_params.get("_litellm_internal_model_credentials")
    bucket_name: str | None = None
    if isinstance(trusted_model_credentials, MappingProxyType):
        snapshot: dict[str, object] = {}
        snapshot.update(trusted_model_credentials)  # any-ok: untyped snapshot
        bucket_name = _TrustedS3ModelCredentials.model_validate(snapshot).s3_bucket_name
    bucket_name = bucket_name or os.getenv("AWS_S3_BUCKET_NAME")
    if not bucket_name:
        raise ValueError(
            "S3 bucket_name is required. Set 's3_bucket_name' in proxy config or AWS_S3_BUCKET_NAME for Bedrock file content retrieval."
        )
    return bucket_name


class BedrockFilesConfig(BaseAWSLLM, BaseFilesConfig):
    """
    Config for Bedrock Files - handles S3 uploads for Bedrock batch processing
    """

    def __init__(self):
        self.jsonl_transformation = BedrockJsonlFilesTransformation()
        super().__init__()

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.BEDROCK

    @property
    def file_upload_http_method(self) -> str:
        """
        Bedrock files are uploaded to S3, which requires PUT requests
        """
        return "PUT"

    def validate_environment(
        self,
        headers: MutableMapping[str, object],
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: MutableMapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        result: dict[str, object] = {}
        result.update(headers)
        signed_headers = litellm_params.pop(S3_SIGNED_GET_HEADERS_PARAM, None)
        if isinstance(signed_headers, Mapping):
            result.update(signed_headers)  # any-ok: untyped handoff headers
        # otherwise no extra headers - AWS credentials are handled by BaseAWSLLM
        return result

    def _get_content_from_openai_file(self, openai_file_content: FileTypes) -> str:
        """
        Helper to extract content from various OpenAI file types and return as string.

        Handles:
        - Direct content (str, bytes, IO[bytes])
        - Tuple formats: (filename, content, [content_type], [headers])
        - PathLike objects
        """
        content: Union[str, bytes] = b""
        # Extract file content from tuple if necessary
        if isinstance(openai_file_content, tuple):
            # Take the second element which is always the file content
            file_content = openai_file_content[1]
        else:
            file_content = openai_file_content

        # Handle different file content types
        if isinstance(file_content, str):
            # String content can be used directly
            content = file_content
        elif isinstance(file_content, bytes):
            # Bytes content can be decoded
            content = file_content
        elif isinstance(file_content, PathLike):  # PathLike
            with open(str(file_content), "rb") as f:
                content = f.read()
        elif hasattr(file_content, "read"):  # IO[bytes]
            # File-like objects need to be read
            content = file_content.read()

        # Ensure content is string
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        return content

    def _get_s3_object_name_from_batch_jsonl(
        self,
        openai_jsonl_content: List[Dict[str, Any]],
    ) -> str:
        """
        Gets a unique S3 object name for the Bedrock batch processing job

        named as: litellm-bedrock-files/{model}/{uuid}
        """
        _model = openai_jsonl_content[0].get("body", {}).get("model", "")
        # Remove bedrock/ prefix if present
        if _model.startswith("bedrock/"):
            _model = _model[8:]

        safe_model = sanitize_cloud_object_component(_model.replace(":", "-"), fallback="model")

        object_name = f"{BEDROCK_MANAGED_S3_BATCH_PREFIX}{safe_model}-{uuid.uuid4()}.jsonl"
        return object_name

    def get_object_name(self, extracted_file_data: ExtractedFileData, purpose: str) -> str:
        """
        Get the object name for the request
        """
        extracted_file_data_content = extracted_file_data.get("content")

        if extracted_file_data_content is None:
            raise ValueError("file content is required")

        if purpose == "batch":
            ## 1. If jsonl, check if there's a model name
            file_content = self._get_content_from_openai_file(extracted_file_data_content)

            # Split into lines and parse each line as JSON
            openai_jsonl_content = [json.loads(line) for line in file_content.splitlines() if line.strip()]
            if len(openai_jsonl_content) > 0:
                return self._get_s3_object_name_from_batch_jsonl(openai_jsonl_content)

        ## 2. If not jsonl, store under a server-generated managed object name
        filename = extracted_file_data.get("filename")
        return build_managed_cloud_object_name(
            prefix=BEDROCK_MANAGED_S3_UPLOAD_PREFIX,
            filename=filename,
            fallback_filename="file",
        )

    def get_complete_file_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: Dict,
        litellm_params: Dict,
        data: CreateFileRequest,
    ) -> str:
        """
        Get the complete S3 URL for the file upload request
        """
        bucket_name = litellm_params.get("s3_bucket_name") or os.getenv("AWS_S3_BUCKET_NAME")
        if not bucket_name:
            raise ValueError(
                "S3 bucket_name is required. Set 's3_bucket_name' in litellm_params or AWS_S3_BUCKET_NAME env var"
            )
        bucket_name, object_prefix = split_configured_cloud_bucket_name(bucket_name)

        s3_region_name = litellm_params.get("s3_region_name") or optional_params.get("s3_region_name")
        aws_region_name = s3_region_name or self._get_aws_region_name(optional_params, model)

        file_data = data.get("file")
        purpose = data.get("purpose")
        if file_data is None:
            raise ValueError("file is required")
        if purpose is None:
            raise ValueError("purpose is required")
        extracted_file_data = extract_file_data(file_data)
        object_name = self.get_object_name(extracted_file_data, purpose)
        if object_prefix:
            object_name = f"{object_prefix}/{object_name}"
        encoded_object_name = encode_s3_object_key_for_url(object_name)

        # S3 endpoint URL format
        s3_endpoint_url = (
            optional_params.get("s3_endpoint_url") or f"https://s3.{aws_region_name}.amazonaws.com"
        ).rstrip("/")

        return f"{s3_endpoint_url}/{bucket_name}/{encoded_object_name}"

    def get_supported_openai_params(self, model: str) -> List[OpenAICreateFileRequestOptionalParams]:
        return []

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        return optional_params

    # Providers whose InvokeModel body uses the Converse API format
    # (messages + inferenceConfig + image blocks). Nova is the primary
    # example; add others here as they adopt the same schema.
    CONVERSE_INVOKE_PROVIDERS = ("nova",)

    # OpenAI batch URL that signals an embedding request. Per OpenAI Batch API
    # spec, every JSONL record carries a `url` field; we use it as the
    # authoritative signal to route the line to the embedding code path
    # instead of inferring from the presence of `input` vs `messages`.
    OPENAI_EMBEDDINGS_URL = "/v1/embeddings"

    @staticmethod
    def _is_embedding_record(openai_jsonl_record: Dict[str, Any]) -> bool:
        """
        Decide whether an OpenAI batch JSONL line is an embedding request.

        Precedence (strict - any explicit `url` short-circuits):
          1. `url == "/v1/embeddings"` -> embedding. Authoritative per the
             OpenAI Batch API spec.
          2. Any other non-empty `url` (e.g. `/v1/chat/completions`) -> NOT
             embedding. We trust the caller's explicit signal even if the
             body would otherwise suggest embedding; misrouting a chat
             record into the embedding transformer would corrupt the
             modelInput, while a chat-shaped body sent to the chat path
             either succeeds or fails cleanly inside that transformer.
          3. `url` missing/empty -> fall back to body shape. Requires
             `input` present AND `messages` absent so a malformed record
             carrying both keys routes to the chat path (safer default:
             Anthropic transforms ignore unknown top-level keys, whereas
             the embedding transformer would silently drop the messages).
        """
        url = openai_jsonl_record.get("url")
        if url == BedrockFilesConfig.OPENAI_EMBEDDINGS_URL:
            return True
        if url:
            return False
        body = openai_jsonl_record.get("body", {})
        if not isinstance(body, dict):
            return False
        return "input" in body and "messages" not in body

    # Identifier for the Bedrock Titan v2 InvokeModel body schema as stored
    # in `model_prices_and_context_window.json`. Centralized so future
    # embedding-schema variants can add their own value
    # (e.g. `cohere_v3`, `titan_g1`, `titan_multimodal`) without touching
    # the detection logic.
    _TITAN_V2_INVOCATION_SCHEMA = "titan_v2"

    # Substring marker used as a fallback when the registry can't resolve
    # the model id - notably cross-region inference profile prefixes
    # (`us.amazon.titan-embed-text-v2:0`) and Bedrock ARN forms, which
    # `get_model_info` doesn't normalize today.
    _TITAN_V2_EMBED_MODEL_MARKER = "titan-embed-text-v2"

    # Nested field name under `provider_specific_entry` that identifies the
    # Bedrock InvokeModel body schema for batch inference.
    # `provider_specific_entry` is the registry's escape hatch for fields
    # `get_model_info` doesn't promote to top-level - exactly what we need
    # here. Documented in the `sample_spec` entry of
    # `model_prices_and_context_window.json` and surfaced by
    # `get_model_info` (see `ModelInfo.provider_specific_entry`).
    _BEDROCK_INVOCATION_SCHEMA_FIELD = "bedrock_invocation_schema"

    @staticmethod
    def _is_titan_v2_embed_model(model: str) -> bool:
        """
        True iff `model` refers to Amazon Titan Text Embeddings V2.

        Resolution order:
          1. `model_prices_and_context_window.json` via `get_model_info`.
             The Titan v2 registry entry carries an explicit
             `provider_specific_entry.bedrock_invocation_schema` discriminator
             (`"titan_v2"`). When the registry resolves the id we trust that
             field as the source of truth - no hardcoded model-id comparison
             needed.
          2. Substring fallback (`titan-embed-text-v2` followed by `:`, `/`,
             or end-of-string) for ids the registry can't normalize. This
             catches cross-region inference profile prefixes
             (`us.amazon.titan-embed-text-v2:0`) and Bedrock ARN forms; the
             marker boundary check rejects lookalikes like
             `titan-embed-text-v20` or `titan-embed-text-v2-experimental`.

        Tolerant of common id shapes:
          - "amazon.titan-embed-text-v2:0"
          - "bedrock/amazon.titan-embed-text-v2:0"
          - "us.amazon.titan-embed-text-v2:0" (cross-region inference profile)
          - ARN forms ending in ".../amazon.titan-embed-text-v2:0"
        """
        # Registry-driven path: when get_model_info resolves the id we trust
        # the registry's discriminator. A resolved id with a different (or
        # absent) schema value here is intentionally not given a substring
        # second-chance - the registry is authoritative for ids it knows.
        registry_schema = BedrockFilesConfig._lookup_provider_specific_field(
            model, BedrockFilesConfig._BEDROCK_INVOCATION_SCHEMA_FIELD
        )
        if registry_schema is not None:
            return registry_schema == BedrockFilesConfig._TITAN_V2_INVOCATION_SCHEMA

        # Registry silence -> substring fallback for unmapped ids only.
        normalized = model.lower()
        if normalized.startswith("bedrock/"):
            normalized = normalized[len("bedrock/") :]
        marker = BedrockFilesConfig._TITAN_V2_EMBED_MODEL_MARKER
        idx = normalized.find(marker)
        if idx < 0:
            return False
        end = idx + len(marker)
        return end == len(normalized) or normalized[end] in (":", "/")

    @staticmethod
    def _lookup_provider_specific_field(model_id: str, field: str) -> Optional[str]:
        """
        Read a nested string field from the registry entry's
        `provider_specific_entry` dict via `litellm.get_model_info`.

        Returns the field's string value when:
          - the registry resolves `model_id`,
          - the entry exposes `provider_specific_entry` as a dict, and
          - that dict has `field` mapped to a non-empty string.
        Otherwise returns `None`.

        Isolating this means feature detectors (Titan v2 today, future
        Cohere Embed / Nova Multimodal branches) share one defensive
        try/except shape instead of duplicating it. The `None` return
        covers every realistic failure mode: `get_model_info` raises
        (cross-region profile prefixes, Bedrock ARN forms, unreleased
        models), returns a non-dict, has no `provider_specific_entry`, or
        the requested field is missing / non-string / empty.
        """
        try:
            from litellm import get_model_info

            info = get_model_info(model_id)
        except Exception:
            return None
        if not isinstance(info, dict):
            return None
        provider_specific = info.get("provider_specific_entry")
        if not isinstance(provider_specific, dict):
            return None
        value = provider_specific.get(field)
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _coerce_embedding_input_to_string(raw_input: Any, model: str = "") -> str:
        """
        Normalize an OpenAI /v1/embeddings `input` field into the single
        string that Bedrock Titan v2 InvokeModel expects in `inputText`.

        Accepts: a string, or a single-element list containing one string.
        Rejects (with actionable messages):
          - None / missing -> ValueError
          - Multi-element string lists -> ValueError, prompts caller to
            emit one JSONL line per input
          - Pre-tokenized inputs (List[int], List[List[int]]) -> NotImplementedError
          - Any other type -> ValueError

        Extracted so the validation can be exercised in isolation and so
        future embedding-provider branches (Titan G1, Cohere) can reuse it
        without duplicating the type-shaping logic.
        """
        if raw_input is None:
            raise ValueError(f"Embedding batch record is missing required `input` field: model={model}")

        # Bedrock InvokeModel for Titan v2 takes exactly one string `inputText`
        # per call. Pre-tokenized inputs and multi-element string lists are
        # explicitly unsupported so callers emit one JSONL line per embedding
        # instead of relying on us to silently fan out or concatenate.
        if isinstance(raw_input, list):
            if len(raw_input) == 1:
                candidate = raw_input[0]
            else:
                raise ValueError(
                    "Bedrock batch embedding requires one input per JSONL "
                    "record. Got a list with "
                    f"{len(raw_input)} items for model={model}; emit one "
                    "JSONL line per input string instead."
                )
        else:
            candidate = raw_input

        # Catches pre-tokenized inputs (List[int] from OpenAI spec, or a
        # single int slipping past the list-unwrap above).
        # NOTE: bool is a subclass of int but treating True/False as a token
        # is meaningless either way, so the broad check is fine.
        if isinstance(candidate, (list, int)):
            raise NotImplementedError(
                "Bedrock Titan v2 batch embedding does not support "
                "pre-tokenized integer inputs. Pass `input` as a string "
                f"(model={model})."
            )
        if not isinstance(candidate, str):
            raise ValueError(
                "Bedrock batch embedding `input` must be a string (or a "
                "single-element list of strings). Got type "
                f"{type(candidate).__name__} for model={model}."
            )
        return candidate

    def _map_openai_embedding_to_bedrock_params(
        self,
        openai_request_body: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Transform an OpenAI /v1/embeddings request body into the
        Bedrock InvokeModel `modelInput` for embedding models that AWS
        supports via batch inference (CreateModelInvocationJob).

        Currently routes Amazon Titan Text Embeddings V2 only; other
        embedding providers (Titan G1, Titan Multimodal, Cohere Embed,
        Nova Multimodal Embeddings) raise NotImplementedError until they
        get a dedicated branch. Splitting them keeps PR scope tight and
        lets each model's request schema be exercised by its own tests.

        AWS docs (Titan v2 InvokeModel body):
        https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
        """
        from litellm.llms.bedrock.embed.amazon_titan_v2_transformation import (
            AmazonTitanV2Config,
        )

        _model = openai_request_body.get("model", "")
        if not self._is_titan_v2_embed_model(_model):
            # Refuse early instead of silently shaping the body for the wrong
            # provider. The synchronous /v1/embeddings path supports more
            # models, but each has a different InvokeModel schema; mapping
            # them here without dedicated tests would risk corrupt batches.
            raise NotImplementedError(
                "Bedrock batch embedding currently supports only Amazon "
                "Titan Text Embeddings V2 (model id contains "
                f"'titan-embed-text-v2'). Got model={_model!r}. Track other "
                "embedding models in https://github.com/BerriAI/litellm/issues."
            )

        input_text = self._coerce_embedding_input_to_string(openai_request_body.get("input"), model=_model)

        # Map OpenAI-style params (dimensions, encoding_format) onto the
        # Titan v2 schema (dimensions, embeddingTypes) via the embed config
        # so this stays in sync with the synchronous /v1/embeddings path.
        non_default_params = {k: v for k, v in openai_request_body.items() if k not in ("model", "input")}
        titan_config = AmazonTitanV2Config()
        inference_params = titan_config.map_openai_params(
            non_default_params=non_default_params,
            optional_params={},
        )
        return dict(titan_config._transform_request(input=input_text, inference_params=inference_params))

    def _map_openai_to_bedrock_params(
        self,
        openai_request_body: Dict[str, Any],
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Transform OpenAI request body to Bedrock-compatible modelInput
        parameters using existing transformation logic.

        Routes to the correct per-provider transformation so that the
        resulting dict matches the InvokeModel body that Bedrock expects
        for batch inference.
        """
        from litellm.types.utils import LlmProviders

        _model = openai_request_body.get("model", "")
        messages = openai_request_body.get("messages", [])
        optional_params = {k: v for k, v in openai_request_body.items() if k not in ["model", "messages"]}

        # --- Anthropic: use existing AmazonAnthropicClaudeConfig ---
        if provider == LlmProviders.ANTHROPIC:
            from litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation import (
                AmazonAnthropicClaudeConfig,
            )

            config = AmazonAnthropicClaudeConfig()
            mapped_params = config.map_openai_params(
                non_default_params={},
                optional_params=optional_params,
                model=_model,
                drop_params=False,
            )
            return config.transform_request(
                model=_model,
                messages=messages,
                optional_params=mapped_params,
                litellm_params={},
                headers={},
            )

        # --- Converse API providers (e.g. Nova): use AmazonConverseConfig
        #     to correctly convert image_url blocks to Bedrock image format
        #     and wrap inference params inside inferenceConfig. ---
        if provider in self.CONVERSE_INVOKE_PROVIDERS:
            from litellm.llms.bedrock.chat.converse_transformation import (
                AmazonConverseConfig,
            )

            converse_config = AmazonConverseConfig()
            mapped_params = converse_config.map_openai_params(
                non_default_params=optional_params,
                optional_params={},
                model=_model,
                drop_params=False,
            )
            return converse_config.transform_request(
                model=_model,
                messages=messages,
                optional_params=mapped_params,
                litellm_params={},
                headers={},
            )

        # --- All other providers: passthrough (OpenAI-compatible models
        #     like openai.gpt-oss-*, qwen, deepseek, etc.) ---
        return {
            "messages": messages,
            **optional_params,
        }

    def _transform_openai_jsonl_content_to_bedrock_jsonl_content(
        self, openai_jsonl_content: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Transforms OpenAI JSONL content to Bedrock batch format

        Bedrock batch format: { "recordId": "alphanumeric string", "modelInput": {JSON body} }
        Example:
        {
            "recordId": "CALL0000001",
            "modelInput": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "Hello"}]
                    }
                ]
            }
        }
        """

        bedrock_jsonl_content = []
        for idx, _openai_jsonl_content in enumerate(openai_jsonl_content):
            # Extract the request body from OpenAI format
            openai_body = _openai_jsonl_content.get("body", {})
            model = openai_body.get("model", "")

            try:
                model, _, _, _ = get_llm_provider(
                    model=model,
                    custom_llm_provider=None,
                )
            except Exception as e:
                verbose_logger.exception(
                    f"litellm.llms.bedrock.files.transformation.py::_transform_openai_jsonl_content_to_bedrock_jsonl_content() - Error inferring custom_llm_provider - {str(e)}"
                )

            # Determine provider from model name
            provider = self.get_bedrock_invoke_provider(model)

            # Route to the embedding transformer when the OpenAI batch line
            # targets /v1/embeddings; otherwise fall back to the existing
            # chat-completion path. We branch here (rather than inside
            # `_map_openai_to_bedrock_params`) so the chat helper keeps its
            # narrow contract and the embedding helper can evolve independently.
            if self._is_embedding_record(_openai_jsonl_content):
                model_input = self._map_openai_embedding_to_bedrock_params(openai_request_body=openai_body)
            else:
                model_input = self._map_openai_to_bedrock_params(openai_request_body=openai_body, provider=provider)

            # Create Bedrock batch record
            record_id = _openai_jsonl_content.get("custom_id", f"CALL{str(idx).zfill(7)}")
            bedrock_record = {"recordId": record_id, "modelInput": model_input}

            bedrock_jsonl_content.append(bedrock_record)
        return bedrock_jsonl_content

    def transform_create_file_request(
        self,
        model: str,
        create_file_data: CreateFileRequest,
        optional_params: dict,
        litellm_params: dict,
    ) -> Union[bytes, str, dict]:
        """
        Transform file request and return a pre-signed request for S3.
        This keeps the HTTP handler clean by doing all the signing here.
        """
        file_data = create_file_data.get("file")
        if file_data is None:
            raise ValueError("file is required")
        extracted_file_data = extract_file_data(file_data)
        extracted_file_data_content = extracted_file_data.get("content")

        if extracted_file_data_content is None:
            raise ValueError("file content is required")

        # Get and transform the file content
        if FilesAPIUtils.is_batch_jsonl_file(
            create_file_data=create_file_data,
            extracted_file_data=extracted_file_data,
        ):
            ## Transform JSONL content to Bedrock format
            original_file_content = self._get_content_from_openai_file(extracted_file_data_content)
            openai_jsonl_content = [json.loads(line) for line in original_file_content.splitlines() if line.strip()]
            bedrock_jsonl_content = self._transform_openai_jsonl_content_to_bedrock_jsonl_content(openai_jsonl_content)
            file_content = "\n".join(json.dumps(item) for item in bedrock_jsonl_content)
        elif isinstance(extracted_file_data_content, bytes):
            file_content = extracted_file_data_content.decode("utf-8")
        elif isinstance(extracted_file_data_content, str):
            file_content = extracted_file_data_content
        else:
            raise ValueError("Unsupported file content type")

        # Get the S3 URL for upload
        api_base = self.get_complete_file_url(
            api_base=None,
            api_key=None,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
            data=create_file_data,
        )

        # s3_region_name always wins for S3 operations (same priority as in
        # get_complete_file_url above). Overwrite aws_region_name unconditionally
        # so the SigV4 region matches the URL region, avoiding SignatureDoesNotMatch.
        s3_region_name = litellm_params.get("s3_region_name") or optional_params.get("s3_region_name")
        if s3_region_name:
            optional_params = {**optional_params, "aws_region_name": s3_region_name}

        # Sign the request and return a pre-signed request object
        signed_headers, signed_body = self._sign_s3_request(
            content=file_content,
            api_base=api_base,
            optional_params=optional_params,
        )

        litellm_params["upload_url"] = api_base

        # Return a dict that tells the HTTP handler exactly what to do
        return {
            "method": "PUT",
            "url": api_base,
            "headers": signed_headers,
            "data": signed_body or file_content,
        }

    def _sign_s3_request(
        self,
        content: str,
        api_base: str,
        optional_params: dict,
    ) -> Tuple[dict, str]:
        """
        Sign S3 PUT request using the same proven logic as S3Logger.
        Reuses the exact pattern from litellm/integrations/s3_v2.py
        """
        try:
            import hashlib

            import requests
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

        # Get AWS credentials using existing methods
        aws_region_name = self._get_aws_region_name(optional_params=optional_params, model="")
        credentials = self.get_credentials(
            aws_access_key_id=optional_params.get("aws_access_key_id"),
            aws_secret_access_key=optional_params.get("aws_secret_access_key"),
            aws_session_token=optional_params.get("aws_session_token"),
            aws_region_name=aws_region_name,
            aws_session_name=optional_params.get("aws_session_name"),
            aws_profile_name=optional_params.get("aws_profile_name"),
            aws_role_name=optional_params.get("aws_role_name"),
            aws_web_identity_token=optional_params.get("aws_web_identity_token"),
            aws_sts_endpoint=optional_params.get("aws_sts_endpoint"),
        )

        # Calculate SHA256 hash of the content (REQUIRED for S3)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # Prepare headers with required S3 headers (same as s3_v2.py)
        request_headers = {
            "Content-Type": "application/json",  # JSONL files are JSON content
            "x-amz-content-sha256": content_hash,  # REQUIRED by S3
            "Content-Language": "en",
            "Cache-Control": "private, immutable, max-age=31536000, s-maxage=0",
        }

        # Use requests.Request to prepare the request (same pattern as s3_v2.py)
        req = requests.Request("PUT", api_base, data=content, headers=request_headers)
        prepped = req.prepare()

        # Sign the request with S3 service
        aws_request = AWSRequest(
            method=prepped.method,
            url=prepped.url,
            data=prepped.body,
            headers=prepped.headers,
        )

        # Get region name for non-LLM API calls (same as s3_v2.py)
        signing_region = self.get_aws_region_name_for_non_llm_api_calls(aws_region_name=aws_region_name)

        SigV4Auth(credentials, "s3", signing_region).add_auth(aws_request)

        # Return signed headers and body
        signed_body = aws_request.body
        if isinstance(signed_body, bytes):
            signed_body = signed_body.decode("utf-8")
        elif signed_body is None:
            signed_body = content  # Fallback to original content

        return dict(aws_request.headers), signed_body

    def _convert_https_url_to_s3_uri(self, https_url: str) -> tuple[str, str]:
        """
        Convert HTTPS S3 URL to s3:// URI format.

        Args:
            https_url: HTTPS S3 URL (e.g., "https://s3.us-west-2.amazonaws.com/bucket/key")

        Returns:
            Tuple of (s3_uri, filename)

        Example:
            Input: "https://s3.us-west-2.amazonaws.com/litellm-proxy/file.jsonl"
            Output: ("s3://litellm-proxy/file.jsonl", "file.jsonl")
        """
        import re

        # Match HTTPS S3 URL patterns
        # Pattern 1: https://s3.region.amazonaws.com/bucket/key
        # Pattern 2: https://bucket.s3.region.amazonaws.com/key

        pattern1 = r"https://s3\.([^.]+)\.amazonaws\.com/([^/]+)/(.+)"
        pattern2 = r"https://([^.]+)\.s3\.([^.]+)\.amazonaws\.com/(.+)"

        match1 = re.match(pattern1, https_url)
        match2 = re.match(pattern2, https_url)

        if match1:
            # Pattern: https://s3.region.amazonaws.com/bucket/key
            region, bucket, key = match1.groups()
            key = unquote(key)
            s3_uri = f"s3://{bucket}/{key}"
        elif match2:
            # Pattern: https://bucket.s3.region.amazonaws.com/key
            bucket, region, key = match2.groups()
            key = unquote(key)
            s3_uri = f"s3://{bucket}/{key}"
        else:
            # Fallback: try to extract bucket and key from URL path
            from urllib.parse import urlparse

            parsed = urlparse(https_url)
            path_parts = parsed.path.lstrip("/").split("/", 1)
            if len(path_parts) >= 2:
                bucket, key = path_parts[0], path_parts[1]
                key = unquote(key)
                s3_uri = f"s3://{bucket}/{key}"
            else:
                raise ValueError(f"Unable to parse S3 URL: {https_url}")

        # Extract filename from key
        filename = key.split("/")[-1] if "/" in key else key

        return s3_uri, filename

    def transform_create_file_response(
        self,
        model: Optional[str],
        raw_response: Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> OpenAIFileObject:
        """
        Transform S3 File upload response into OpenAI-style FileObject
        """
        # For S3 uploads, we typically get an ETag and other metadata
        response_headers = raw_response.headers
        # Extract S3 object information from the response
        # S3 PUT object returns ETag and other metadata in headers
        content_length = response_headers.get("Content-Length", "0")

        # Use the actual upload URL that was used for the S3 upload
        upload_url = litellm_params.get("upload_url")
        file_id: str = ""
        filename: str = ""
        if upload_url:
            # Convert HTTPS S3 URL to s3:// URI format
            file_id, filename = self._convert_https_url_to_s3_uri(upload_url)

        return OpenAIFileObject(
            purpose="batch",  # Default purpose for Bedrock files
            id=file_id,
            filename=filename,
            created_at=int(time.time()),  # Current timestamp
            status="uploaded",
            bytes=int(content_length) if content_length.isdigit() else 0,
            object="file",
        )

    def get_error_class(self, error_message: str, status_code: int, headers: Union[Dict, Headers]) -> BaseLLMException:
        return BedrockError(status_code=status_code, message=error_message, headers=headers)

    def transform_retrieve_file_request(
        self,
        file_id: str,
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        raise NotImplementedError("BedrockFilesConfig does not support file retrieval")

    def transform_retrieve_file_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> OpenAIFileObject:
        raise NotImplementedError("BedrockFilesConfig does not support file retrieval")

    def transform_delete_file_request(
        self,
        file_id: str,
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        raise NotImplementedError("BedrockFilesConfig does not support file deletion")

    def transform_delete_file_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> FileDeleted:
        raise NotImplementedError("BedrockFilesConfig does not support file deletion")

    def transform_list_files_request(
        self,
        purpose: Optional[str],
        optional_params: dict,
        litellm_params: dict,
    ) -> tuple[str, dict]:
        raise NotImplementedError("BedrockFilesConfig does not support file listing")

    def transform_list_files_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> List[OpenAIFileObject]:
        raise NotImplementedError("BedrockFilesConfig does not support file listing")

    def transform_file_content_request(
        self,
        file_content_request: FileContentRequest,
        optional_params: Mapping[str, object],
        litellm_params: MutableMapping[str, object],
    ) -> tuple[str, dict[str, str]]:
        """
        Build a SigV4-signed S3 GetObject request for a Bedrock batch file.

        Bedrock batch file ids are `s3://bucket/key` URIs (or unified ids
        that decode to one); the bucket and key are validated against the
        server-configured bucket before any request is signed.
        """
        file_id = file_content_request.get("file_id")
        if not file_id:
            raise ValueError("file_id is required for Bedrock file content retrieval")

        s3_uri = extract_s3_uri_from_file_id(file_id)
        bucket_name, object_key = validate_managed_cloud_file_id(
            file_id=s3_uri,
            scheme="s3://",
            configured_bucket_name=get_configured_s3_bucket_name(litellm_params),
            allowed_object_prefixes=BEDROCK_MANAGED_S3_PREFIXES,
            allow_legacy_cloud_file_ids=should_allow_legacy_cloud_file_ids(litellm_params),
        )

        # The shared file-content handler passes optional_params={}, so AWS
        # credentials/region arrive via litellm_params here (unlike the upload
        # path). s3_region_name wins over aws_region_name, same priority as
        # get_complete_file_url above.
        merged_params: dict[str, object] = {}
        merged_params.update(litellm_params)
        merged_params.update(optional_params)
        request_params = _BedrockS3RequestParams.model_validate(merged_params)

        region_preference = request_params.s3_region_name or request_params.aws_region_name
        region_params: dict[str, str | None] = {"aws_region_name": region_preference}
        aws_region_name = self._get_aws_region_name(optional_params=region_params, model="")

        s3_endpoint_url = (request_params.s3_endpoint_url or f"https://s3.{aws_region_name}.amazonaws.com").rstrip("/")
        url = f"{s3_endpoint_url}/{bucket_name}/{encode_s3_object_key_for_url(object_key)}"

        litellm_params[S3_SIGNED_GET_HEADERS_PARAM] = self._sign_s3_get_request(
            api_base=url,
            aws_region_name=aws_region_name,
            request_params=request_params,
        )
        return url, {}

    def _sign_s3_get_request(
        self,
        api_base: str,
        aws_region_name: str,
        request_params: _BedrockS3RequestParams,
    ) -> dict[str, str]:
        """
        SigV4-sign an S3 GetObject request, mirroring `_sign_s3_request` (PUT).
        """
        try:
            import hashlib

            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest
        except ImportError:
            raise ImportError("Missing boto3 to call bedrock. Run 'pip install boto3'.")

        credentials = self.get_credentials(  # any-ok: boto3 Credentials is untyped
            aws_access_key_id=request_params.aws_access_key_id,
            aws_secret_access_key=request_params.aws_secret_access_key,
            aws_session_token=request_params.aws_session_token,
            aws_region_name=aws_region_name,
            aws_session_name=request_params.aws_session_name,
            aws_profile_name=request_params.aws_profile_name,
            aws_role_name=request_params.aws_role_name,
            aws_web_identity_token=request_params.aws_web_identity_token,
            aws_sts_endpoint=request_params.aws_sts_endpoint,
        )

        empty_body_hash = hashlib.sha256(b"").hexdigest()
        aws_request = AWSRequest(  # any-ok: botocore AWSRequest is untyped
            method="GET",
            url=api_base,
            headers={"x-amz-content-sha256": empty_body_hash},
        )
        auth = SigV4Auth(credentials, "s3", aws_region_name)  # any-ok: botocore untyped
        auth.add_auth(aws_request)  # any-ok: botocore request mutation is untyped
        return dict(aws_request.headers)  # any-ok: botocore headers are untyped

    def transform_file_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
    ) -> HttpxBinaryResponseContent:
        if raw_response.status_code >= 400:
            raise BedrockError(
                status_code=raw_response.status_code,
                message=raw_response.text,
                headers=raw_response.headers,
            )
        return HttpxBinaryResponseContent(response=raw_response)


class BedrockJsonlFilesTransformation:
    """
    Transforms OpenAI /v1/files/* requests to Bedrock S3 file uploads for batch processing
    """

    def transform_openai_file_content_to_bedrock_file_content(
        self, openai_file_content: Optional[FileTypes] = None
    ) -> Tuple[str, str]:
        """
        Transforms OpenAI FileContentRequest to Bedrock S3 file format
        """

        if openai_file_content is None:
            raise ValueError("contents of file are None")
        # Read the content of the file
        file_content = self._get_content_from_openai_file(openai_file_content)

        # Split into lines and parse each line as JSON
        openai_jsonl_content = [json.loads(line) for line in file_content.splitlines() if line.strip()]
        bedrock_jsonl_content = self._transform_openai_jsonl_content_to_bedrock_jsonl_content(openai_jsonl_content)
        bedrock_jsonl_string = "\n".join(json.dumps(item) for item in bedrock_jsonl_content)
        object_name = self._get_s3_object_name(openai_jsonl_content=openai_jsonl_content)
        return bedrock_jsonl_string, object_name

    def _transform_openai_jsonl_content_to_bedrock_jsonl_content(self, openai_jsonl_content: List[Dict[str, Any]]):
        """
        Delegate to the main BedrockFilesConfig transformation method
        """
        config = BedrockFilesConfig()
        return config._transform_openai_jsonl_content_to_bedrock_jsonl_content(openai_jsonl_content)

    def _get_s3_object_name(
        self,
        openai_jsonl_content: List[Dict[str, Any]],
    ) -> str:
        """
        Gets a unique S3 object name for the Bedrock batch processing job

        named as: litellm-bedrock-files-{model}-{uuid}
        """
        _model = openai_jsonl_content[0].get("body", {}).get("model", "")
        # Remove bedrock/ prefix if present
        if _model.startswith("bedrock/"):
            _model = _model[8:]
        safe_model = sanitize_cloud_object_component(_model.replace(":", "-"), fallback="model")
        object_name = f"{BEDROCK_MANAGED_S3_BATCH_PREFIX}{safe_model}-{uuid.uuid4()}.jsonl"
        return object_name

    def _get_content_from_openai_file(self, openai_file_content: FileTypes) -> str:
        """
        Helper to extract content from various OpenAI file types and return as string.

        Handles:
        - Direct content (str, bytes, IO[bytes])
        - Tuple formats: (filename, content, [content_type], [headers])
        - PathLike objects
        """
        content: Union[str, bytes] = b""
        # Extract file content from tuple if necessary
        if isinstance(openai_file_content, tuple):
            # Take the second element which is always the file content
            file_content = openai_file_content[1]
        else:
            file_content = openai_file_content

        # Handle different file content types
        if isinstance(file_content, str):
            # String content can be used directly
            content = file_content
        elif isinstance(file_content, bytes):
            # Bytes content can be decoded
            content = file_content
        elif isinstance(file_content, PathLike):  # PathLike
            with open(str(file_content), "rb") as f:
                content = f.read()
        elif hasattr(file_content, "read"):  # IO[bytes]
            # File-like objects need to be read
            content = file_content.read()

        # Ensure content is string
        if isinstance(content, bytes):
            content = content.decode("utf-8")

        return content

    def transform_s3_bucket_response_to_openai_file_object(
        self, create_file_data: CreateFileRequest, s3_upload_response: Dict[str, Any]
    ) -> OpenAIFileObject:
        """
        Transforms S3 Bucket upload file response to OpenAI FileObject
        """
        # S3 response typically contains ETag, key, etc.
        object_key = s3_upload_response.get("Key", "")
        bucket_name = s3_upload_response.get("Bucket", "")

        # Extract filename from object key
        filename = object_key.split("/")[-1] if "/" in object_key else object_key

        return OpenAIFileObject(
            purpose=create_file_data.get("purpose", "batch"),
            id=f"s3://{bucket_name}/{object_key}",
            filename=filename,
            created_at=int(time.time()),  # Current timestamp
            status="uploaded",
            bytes=s3_upload_response.get("ContentLength", 0),
            object="file",
        )
