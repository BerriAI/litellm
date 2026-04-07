"""
OCI Generative AI — Embedding transformation.

Endpoint: POST /20231130/actions/embedText
Supported models: cohere.embed-english-v3.0, cohere.embed-multilingual-v3.0,
cohere.embed-v4.0, and all other Cohere embed variants available on OCI
(including dedicated endpoints).

Authentication follows the same RSA-SHA256 / OCI SDK signer pattern as chat.
The base handler (base_llm_http_handler.embedding) calls sign_request after
building the body, so signing happens automatically.

Supported models:
- cohere.embed-english-v3.0
- cohere.embed-english-light-v3.0
- cohere.embed-multilingual-v3.0
- cohere.embed-multilingual-light-v3.0
- cohere.embed-english-image-v3.0
- cohere.embed-multilingual-image-v3.0
- cohere.embed-v4.0

Reference: https://docs.oracle.com/en-us/iaas/api/#/en/generative-ai-inference/latest/EmbedTextResult/EmbedText
"""

from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union

import httpx

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.embedding.transformation import BaseEmbeddingConfig
from litellm.llms.oci.common_utils import (
    OCI_API_VERSION,
    OCIError,
    get_oci_base_url,
    resolve_oci_credentials,
    sign_oci_request,
    validate_oci_environment,
)
from litellm.types.llms.oci import (
    OCIEmbedRequest,
    OCIEmbedResponse,
    OCIServingMode,
)
from litellm.types.llms.openai import AllEmbeddingInputValues, AllMessageValues
from litellm.types.utils import EmbeddingResponse, Usage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

# OCI sends up to 96 texts per embedText request (Cohere limit).
OCI_EMBED_BATCH_LIMIT = 96

# Input type mapping from OpenAI conventions to OCI/Cohere conventions
_INPUT_TYPE_MAP = {
    "search_document": "SEARCH_DOCUMENT",
    "search_query": "SEARCH_QUERY",
    "classification": "CLASSIFICATION",
    "clustering": "CLUSTERING",
}


class OCIEmbedConfig(BaseEmbeddingConfig):
    """
    Transformation config for OCI Generative AI embeddings.

    Supports both text and (on cohere.embed-v4.0) multimodal inputs.

    Authentication — same two modes as chat:
    - **OCI SDK signer**: pass ``oci_signer`` in optional_params.
    - **Manual RSA-SHA256**: pass ``oci_user``, ``oci_fingerprint``, ``oci_tenancy``,
      and ``oci_key`` or ``oci_key_file``, or set the corresponding ``OCI_*`` env vars.

    Required call-time params (via optional_params or env vars):
    - ``oci_compartment_id`` / ``OCI_COMPARTMENT_ID``
    - ``oci_region`` / ``OCI_REGION`` (default: ``us-ashburn-1``)

    Optional call-time params:
    - ``oci_serving_mode``: ``"ON_DEMAND"`` (default) or ``"DEDICATED"``
    - ``oci_endpoint_id``: endpoint OCID for dedicated serving mode
    - ``input_type``: ``SEARCH_DOCUMENT``, ``SEARCH_QUERY``, ``CLASSIFICATION``, ``CLUSTERING``
    - ``truncate``: ``NONE``, ``START``, or ``END`` (default ``END``)
    - ``dimensions``: output embedding dimensions (cohere.embed-v4.0+)
    """

    def get_supported_openai_params(self, model: str) -> List[str]:
        return ["dimensions"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool = False,
    ) -> dict:
        for key, value in non_default_params.items():
            if key == "dimensions":
                # OCI API uses outputDimensions (cohere.embed-v4.0+)
                optional_params["outputDimensions"] = value
        return optional_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        return validate_oci_environment(headers, optional_params, api_key)

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        # If the caller provides a full endpoint URL, use it as-is.
        # Otherwise construct the standard OCI GenAI embedText endpoint from the region.
        resolved_base = api_base or litellm.api_base
        if resolved_base:
            return resolved_base.rstrip("/")
        base = get_oci_base_url(optional_params, None)
        return f"{base}/{OCI_API_VERSION}/actions/embedText"

    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        fake_stream: Optional[bool] = None,
    ) -> Tuple[dict, Optional[bytes]]:
        return sign_oci_request(
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base=api_base,
            api_key=api_key,
            model=model,
        )

    def transform_embedding_request(
        self,
        model: str,
        input: AllEmbeddingInputValues,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        creds = resolve_oci_credentials(optional_params)
        compartment_id = creds["oci_compartment_id"]
        if not compartment_id:
            raise OCIError(
                status_code=400,
                message=(
                    "oci_compartment_id is required for OCI embedding requests. "
                    "Pass it as optional_params or set the OCI_COMPARTMENT_ID env var."
                ),
            )

        # Normalise input to a flat list of strings
        if isinstance(input, str):
            texts = [input]
        elif isinstance(input, list):
            texts = []
            for item in input:
                if isinstance(item, list):
                    raise OCIError(
                        status_code=400,
                        message=(
                            "OCI embedText does not support token-array inputs. "
                            "Convert token lists to strings before calling embedding()."
                        ),
                    )
                texts.append(item if isinstance(item, str) else str(item))
        else:
            texts = [str(input)]

        if len(texts) > OCI_EMBED_BATCH_LIMIT:
            raise OCIError(
                status_code=400,
                message=(
                    f"OCI embedText accepts at most {OCI_EMBED_BATCH_LIMIT} inputs per request "
                    f"(got {len(texts)}). Batch your requests."
                ),
            )

        serving_mode_type = optional_params.get("oci_serving_mode", "ON_DEMAND").upper()
        if serving_mode_type not in {"ON_DEMAND", "DEDICATED"}:
            raise OCIError(
                status_code=400,
                message="oci_serving_mode must be 'ON_DEMAND' or 'DEDICATED'.",
            )

        if serving_mode_type == "DEDICATED":
            endpoint_id = optional_params.get("oci_endpoint_id", model)
            serving_mode = OCIServingMode(
                servingType="DEDICATED", endpointId=endpoint_id
            )
        else:
            serving_mode = OCIServingMode(servingType="ON_DEMAND", modelId=model)

        # Map input_type from OpenAI convention to OCI/Cohere convention
        input_type = optional_params.get("input_type")
        if input_type:
            input_type = _INPUT_TYPE_MAP.get(input_type.lower(), input_type.upper())

        request = OCIEmbedRequest(
            compartmentId=compartment_id,
            servingMode=serving_mode,
            inputs=texts,
            inputType=input_type,
            truncate=optional_params.get("truncate", "END"),
            outputDimensions=optional_params.get("outputDimensions"),
        )
        return request.model_dump(exclude_none=True)

    def transform_embedding_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: EmbeddingResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str],
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
    ) -> EmbeddingResponse:
        if raw_response.status_code != 200:
            raise OCIError(
                status_code=raw_response.status_code,
                message=raw_response.text,
            )

        try:
            json_response = raw_response.json()
        except Exception as e:
            raise OCIError(
                status_code=raw_response.status_code,
                message=f"Failed to parse OCI embed response as JSON: {e}",
            )

        try:
            parsed = OCIEmbedResponse(**json_response)
        except Exception as e:
            raise OCIError(
                status_code=500,
                message=f"OCI embed response does not match expected schema: {e}",
            )

        model_response.model = parsed.modelId
        model_response.data = [
            {
                "object": "embedding",
                "index": i,
                "embedding": embedding,
            }
            for i, embedding in enumerate(parsed.embeddings)
        ]

        if parsed.inputTextTokenCounts is not None:
            # Actual OCI API returns per-input token counts — sum for total usage
            total = sum(parsed.inputTextTokenCounts)
            model_response.usage = Usage(prompt_tokens=total, total_tokens=total)
        elif parsed.usage is not None:
            # Some deployments may return a usage object directly
            model_response.usage = Usage(
                prompt_tokens=parsed.usage.promptTokens,
                total_tokens=parsed.usage.totalTokens,
            )

        return model_response

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> BaseLLMException:
        return OCIError(status_code=status_code, message=error_message)


# Alias for backwards compatibility with any code that imports OCIEmbeddingConfig
OCIEmbeddingConfig = OCIEmbedConfig
