"""
Translate between Cohere's `/rerank` format and Deepinfra's `/rerank` format. 
"""

from typing import Any, Dict, List, Optional, Union

import httpx

from litellm._uuid import uuid
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.rerank.transformation import (
    BaseLLMException,
    BaseRerankConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.rerank import (
    OptionalRerankParams,
    RerankBilledUnits,
    RerankResponse,
    RerankResponseMeta,
    RerankResponseResult,
    RerankTokens,
)


class DeepinfraRerankConfig(BaseRerankConfig):
    """
    Deepinfra Rerank - Follows the same Spec as Cohere Rerank
    """

    def get_complete_url(self, api_base: Optional[str], model: str) -> str:
        """
        Constructs the complete DeepInfra inference endpoint URL for rerank.

        Args:
            api_base (Optional[str]): The base URL for the DeepInfra API.
            model (str): The model identifier.

        Returns:
            str: The complete URL for the DeepInfra rerank inference endpoint.

        Raises:
            ValueError: If api_base is None.
        """
        if not api_base:
            raise ValueError(
                "Deepinfra API Base is required. api_base=None. Set in call or via `DEEPINFRA_API_BASE` env var."
            )

        # Remove 'openai' from the base if present
        api_base_clean = (
            api_base.replace("openai", "") if "openai" in api_base else api_base
        )

        # Remove any trailing slashes for consistency, then add one
        api_base_clean = api_base_clean.rstrip("/") + "/"

        # Compose the full endpoint
        return f"{api_base_clean}inference/{model}"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        if api_key is None:
            api_key = get_secret_str("DEEPINFRA_API_KEY")

        if api_key is None:
            raise ValueError(
                "Deepinfra API key is required. Please set 'DEEPINFRA_API_KEY' environment variable"
            )

        default_headers = {
            "Authorization": f"Bearer {api_key}",
            "accept": "application/json",
            "content-type": "application/json",
        }

        # If 'Authorization' is provided in headers, it overrides the default.
        if "Authorization" in headers:
            default_headers["Authorization"] = headers["Authorization"]

        # Merge other headers, overriding any default ones except Authorization
        return {**default_headers, **headers}

    def map_cohere_rerank_params(
        self,
        non_default_params: dict,
        model: str,
        drop_params: bool,
        query: str,
        documents: List[Union[str, Dict[str, Any]]],
        custom_llm_provider: Optional[str] = None,
        top_n: Optional[int] = None,
        rank_fields: Optional[List[str]] = None,
        return_documents: Optional[bool] = True,
        max_chunks_per_doc: Optional[int] = None,
        max_tokens_per_doc: Optional[int] = None,
    ) -> Dict:
        # Start with the basic parameters
        optional_rerank_params = {}
        if query:
            optional_rerank_params["queries"] = [query] * len(
                documents
            )  # Deepinfra rerank requires queries to be of same length as documents

        if non_default_params is not None:
            for k, v in non_default_params.items():
                if k == "queries" and v is not None:
                    # This should override the query parameter if it is provided
                    optional_rerank_params["queries"] = v
                elif k == "documents" and v is not None:
                    optional_rerank_params["documents"] = v
                elif k == "service_tier" and v is not None:
                    optional_rerank_params["service_tier"] = v
                elif k == "instruction" and v is not None:
                    optional_rerank_params["instruction"] = v
                elif k == "webhook" and v is not None:
                    optional_rerank_params["webhook"] = v
        return OptionalRerankParams(**optional_rerank_params)  # type: ignore

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: Dict,
        headers: dict,
    ) -> dict:
        # Convert OptionalRerankParams to dict as expected by parent class
        if optional_rerank_params is None:
            return {}
        return dict(optional_rerank_params)

    def transform_rerank_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: RerankResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: Optional[str] = None,
        request_data: dict = {},
        optional_params: dict = {},
        litellm_params: dict = {},
    ) -> RerankResponse:
        try:
            response_json = raw_response.json()
            logging_obj.post_call(original_response=raw_response.text)

            # Extract the scores from the response
            scores = response_json.get("scores", [])
            input_tokens = response_json.get("input_tokens", 0)
            request_id = response_json.get("request_id")

            # Create inference status information
            inference_status = response_json.get("inference_status", {})
            status = inference_status.get("status", "unknown")
            runtime_ms = inference_status.get("runtime_ms", 0)
            cost = inference_status.get("cost", 0.0)
            tokens_generated = inference_status.get("tokens_generated", 0)
            tokens_input = inference_status.get("tokens_input", 0)

            # Create RerankResponse
            results = []
            for i, score in enumerate(scores):
                results.append(
                    RerankResponseResult(index=i, relevance_score=float(score))
                )

            # Create metadata for the response
            tokens = RerankTokens(
                input_tokens=input_tokens,
                output_tokens=0,  # DeepInfra doesn't provide output tokens for rerank
            )
            billed_units = RerankBilledUnits(total_tokens=input_tokens)
            meta = RerankResponseMeta(tokens=tokens, billed_units=billed_units)

            rerank_response = RerankResponse(
                id=request_id or str(uuid.uuid4()), results=results, meta=meta
            )

            # Store additional information in hidden params
            rerank_response._hidden_params = {
                "status": status,
                "runtime_ms": runtime_ms,
                "cost": cost,
                "tokens_generated": tokens_generated,
                "tokens_input": tokens_input,
                "model": model,
            }

            return rerank_response

        except Exception:
            # If there's an error parsing the response, fall back to the parent implementation
            rerank_response = super().transform_rerank_response(
                model=model,
                raw_response=raw_response,
                model_response=model_response,
                logging_obj=logging_obj,
                api_key=api_key,
                request_data=request_data,
                optional_params=optional_params,
                litellm_params=litellm_params,
            )

            rerank_response._hidden_params["model"] = model
            return rerank_response

    def get_supported_cohere_rerank_params(self, model: str) -> list:
        return ["query", "documents"]

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        # Deepinfra errors may come as JSON: {"detail": {"error": "..."}}
        import json

        # Try to extract a more specific error message if possible
        try:
            error_data = error_message
            if isinstance(error_message, str):
                error_data = json.loads(error_message)
            if isinstance(error_data, dict):
                # Check for {"detail": {"error": "..."}}
                detail = error_data.get("detail")
                if isinstance(detail, dict) and "error" in detail:
                    error_message = detail["error"]
                elif isinstance(detail, str):
                    error_message = detail
        except Exception:
            # If parsing fails, just use the original error_message
            pass

        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
