import json
import time
import uuid
from typing import Dict, Optional, Union

from httpx import Headers, Response

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import extract_file_data
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.files.transformation import (
    LiteLLMLoggingObj,
)
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.types.llms.openai import (
    CreateFileRequest,
    OpenAIFileObject,
    PathLike,
)

from ..common_utils import BedrockError


class BedrockJsonlFilesTransformation(BaseAWSLLM):
    """
    Transforms OpenAI /v1/files/* requests to Bedrock /v1/files/* requests
    """

    def __init__(self):
        super().__init__()

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[Dict, Headers],
    ) -> BaseLLMException:
        return BedrockError(status_code=status_code, message=error_message)

    def get_complete_file_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        litellm_params: Dict,
        data: CreateFileRequest,
    ) -> str:
        """
        Compose the S3 object URL to PUT the object (virtual-hosted-style URL)
        https://{bucket}.s3.{region}.amazonaws.com/{key}
        """
        # Pull S3 config from global s3_callback_params
        import litellm

        s3_params = getattr(litellm, "s3_callback_params", {}) or {}
        bucket = s3_params.get("s3_bucket_name")
        region = s3_params.get("s3_region_name") or "us-west-2"
        endpoint_url = s3_params.get("s3_endpoint_url")
        base_path = s3_params.get("s3_path") or ""
        if not bucket:
            raise ValueError("s3_bucket_name is required for Bedrock files")

        # Build object key (deterministic path using model if present; fallback to uuid)
        file_data = data.get("file")
        if file_data is None:
            raise ValueError("file data is required")
        extracted = extract_file_data(file_data)
        filename = extracted.get("filename") or "upload.jsonl"

        # For batch files, infer model_id and include it in the object key
        model_id = ""
        if data.get("purpose") == "batch":
            content = extracted.get("content")
            if content:
                if isinstance(content, bytes):
                    content_str = content.decode("utf-8")
                elif isinstance(content, str):
                    content_str = content
                elif hasattr(content, "read"):
                    content_str = (
                        content.read().decode("utf-8")
                        if isinstance(content.read(), bytes)
                        else content.read()
                    )
                else:
                    content_str = ""

                model_id = self.infer_model_id_from_openai_jsonl(content_str) or ""

        # Include model_id in object key for batch files to enable proper batch processing
        if model_id:
            object_key = f"{base_path.rstrip('/') + '/' if base_path else ''}{model_id}/{uuid.uuid4()}-{filename}"
        else:
            object_key = f"{base_path.rstrip('/') + '/' if base_path else ''}{uuid.uuid4()}-{filename}"

        # Store values for response transformation
        self._s3_bucket = bucket
        self._s3_region = region
        self._s3_endpoint_url = endpoint_url
        self._s3_object_key = object_key
        self._model_id = model_id

        if endpoint_url:
            return f"{endpoint_url}/{object_key}"
        return f"https://{bucket}.s3.{region}.amazonaws.com/{object_key}"

    def transform_openai_file_content_to_bedrock_jsonl_str(
        self, openai_jsonl_content: str
    ) -> str:
        """
        Transforms OpenAI JSONL content to Bedrock JSONL content

        Args:
            openai_jsonl_content: String containing JSONL formatted OpenAI batch requests

        Returns:
            String containing JSONL formatted Bedrock batch requests

        Example OpenAI JSONL input:
        {"custom_id": "request-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "anthropic.claude-v2", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 10}}

        Example Bedrock JSONL output:
        {"recordId": "request-1", "modelInput": {"anthropic_version": "bedrock-2023-05-31", "max_tokens": 10, "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}]}}
        """
        import uuid

        from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
            AmazonInvokeConfig,
        )

        bedrock_jsonl_lines = []
        amazon_invoke_config = AmazonInvokeConfig()
        verbose_logger.debug(f"openai_jsonl_content: {openai_jsonl_content}")
        # Parse JSONL string into individual JSON objects
        for line in openai_jsonl_content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            try:
                openai_request = json.loads(line)
            except json.JSONDecodeError as e:
                verbose_logger.warning(
                    f"Skipping malformed JSONL line: {line}. Error: {e}"
                )
                continue

            # Extract request details from OpenAI format
            body = openai_request.get("body", {})
            custom_id = openai_request.get("custom_id", f"request-{uuid.uuid4()}")
            model = body.get("model", "")
            messages = body.get("messages", [])

            # Extract optional parameters (everything except model and messages)
            optional_params = {
                k: v for k, v in body.items() if k not in ("model", "messages")
            }

            try:
                # Transform using Amazon invoke config
                output_body = amazon_invoke_config.transform_request(
                    model=model,
                    messages=messages,
                    optional_params=optional_params,
                    litellm_params={},
                    headers={},
                )
                verbose_logger.debug(f"output_body: {output_body}")

                # Create Bedrock JSONL format
                bedrock_request = {"recordId": custom_id, "modelInput": output_body}

                bedrock_jsonl_lines.append(json.dumps(bedrock_request))

            except Exception as e:
                verbose_logger.warning(f"Failed to transform request {custom_id}: {e}")
                continue

        return "\n".join(bedrock_jsonl_lines)

    def transform_openai_file_content_to_bedrock_file_content(
        self,
        create_file_data: CreateFileRequest,
        litellm_params: dict,
        optional_params: Optional[dict] = {},
    ) -> Union[bytes, str, dict]:
        """
        Return a single SigV4-signed request definition for S3 PUT using httpx.
        The BaseLLMHTTPHandler is extended to handle this format.
        """
        file_data = create_file_data.get("file")
        if file_data is None:
            raise ValueError("file is required")
        extracted = extract_file_data(file_data)
        content = extracted.get("content")
        verbose_logger.debug(f"content: {content!r}")
        content_type = extracted.get("content_type") or "application/octet-stream"

        # For batch files, assume JSONL content type if purpose is batch
        filename = extracted.get("filename") or ""
        if create_file_data.get("purpose") == "batch":
            # Try to detect if it's JSONL content
            if filename.endswith(".jsonl") or content_type in [
                "application/jsonl",
                "text/jsonl",
            ]:
                content_type = "application/jsonl"
            else:
                # For batch purpose, assume it's JSONL even if we can't detect from filename
                content_type = "application/jsonl"

        if hasattr(self, "_s3_object_key") is False:
            raise ValueError("S3 upload URL/object key not prepared")

        # Get the S3 URL (stored from previous call)
        s3_url = getattr(self, "_s3_object_key", None)
        if not s3_url:
            raise ValueError("S3 upload URL/object key not prepared")
        # For S3 PUT
        if isinstance(content, bytes):
            raw_bytes = content
        elif isinstance(content, str):
            raw_bytes = content.encode("utf-8")
        elif hasattr(content, "read") and content is not None:
            raw_bytes = content.read()
        elif isinstance(content, PathLike):
            with open(str(content), "rb") as f:
                raw_bytes = f.read()
        else:
            raise ValueError("Unsupported file content type for Bedrock S3 upload")

        # If batch JSONL, transform OpenAI JSONL → Bedrock JSONL before upload
        payload_bytes = raw_bytes
        bedrock_model_id = ""
        verbose_logger.debug(
            f"Checking transformation conditions: purpose={create_file_data.get('purpose')}, content_type={content_type}, is_bytes={isinstance(raw_bytes, (bytes, bytearray))}"
        )
        if (
            create_file_data.get("purpose") == "batch"
            and content_type == "application/jsonl"
            and isinstance(raw_bytes, (bytes, bytearray))
        ):
            try:
                verbose_logger.debug("Starting JSONL transformation...")
                # try infer model id from first line
                bedrock_model_id = (
                    self.infer_model_id_from_openai_jsonl(raw_bytes.decode("utf-8"))
                    or ""
                )
                verbose_logger.debug(f"Inferred model ID: {bedrock_model_id}")
                transformed_jsonl_str = (
                    self.transform_openai_file_content_to_bedrock_jsonl_str(
                        raw_bytes.decode("utf-8")
                    )
                )
                verbose_logger.debug(
                    f"Transformed JSONL: {transformed_jsonl_str[:200]}..."
                )
                payload_bytes = transformed_jsonl_str.encode("utf-8")
                verbose_logger.debug(
                    f"Transformation completed. Original size: {len(raw_bytes)}, Transformed size: {len(payload_bytes)}"
                )
            except Exception as e:
                verbose_logger.exception(
                    f"Error transforming batch JSONL for Bedrock: {e}"
                )
                verbose_logger.warning("Transformation failed, using original content")
                payload_bytes = raw_bytes
        else:
            verbose_logger.debug("Skipping transformation - conditions not met")

        # Stash content length for response object
        self._uploaded_payload_len = len(payload_bytes)

        return {
            "request": "boto3_s3",  # Flag to use boto3 S3 client
            "bucket": self._s3_bucket,
            "key": self._s3_object_key,
            "content": payload_bytes,
            "content_type": content_type,
            "region": self._s3_region,
            "model_id": getattr(self, "_model_id", "") or bedrock_model_id,
        }

    def transform_s3_bucket_response_to_openai_file_object(
        self,
        model: Optional[str],
        raw_response: Response,
        logging_obj: LiteLLMLoggingObj,
        litellm_params: dict,
        model_id: Optional[str],
    ) -> OpenAIFileObject:
        # Build an OpenAI-like FileObject pointing to the S3 URI
        if raw_response.status_code not in (200, 201):
            raise BaseLLMException(
                status_code=raw_response.status_code,
                message=f"S3 upload failed with status {raw_response.status_code}: {raw_response.text!r}",
            )

        # Get the model_id from parameter or stored value
        effective_model_id = model_id or getattr(self, "_model_id", "")

        # Build S3 URI - for batch files, we want the path up to the model_id folder
        # so the batch handler can extract the model_id properly
        if getattr(self, "_s3_bucket", None) and getattr(self, "_s3_object_key", None):
            if effective_model_id and f"/{effective_model_id}/" in self._s3_object_key:
                # For batch files with model_id in path, return S3 URI pointing to the model folder
                # This allows batch handler to extract model_id: s3://bucket/path/model_id/
                bucket_and_path = self._s3_object_key.split(f"/{effective_model_id}/")[
                    0
                ]
                s3_uri = (
                    f"s3://{self._s3_bucket}/{bucket_and_path}/{effective_model_id}/"
                    if bucket_and_path
                    else f"s3://{self._s3_bucket}/{effective_model_id}/"
                )
            else:
                # Standard S3 URI for non-batch files
                s3_uri = f"s3://{self._s3_bucket}/{self._s3_object_key}"
        else:
            s3_uri = ""
        return OpenAIFileObject(
            purpose="batch",
            id=s3_uri,
            filename=self._s3_object_key.split("/")[-1]
            if getattr(self, "_s3_object_key", None)
            else "",
            created_at=int(time.time()),
            status="uploaded",
            bytes=getattr(self, "_uploaded_payload_len", 0),
            object="file",
        )

    def infer_model_id_from_openai_jsonl(self, file_content: str) -> Optional[str]:
        for raw in file_content.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            body = obj.get("body") or obj
            model = body.get("model")
            if isinstance(model, str) and model:
                return model
        return None
