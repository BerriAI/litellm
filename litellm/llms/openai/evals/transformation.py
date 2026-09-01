"""
OpenAI Evals API configuration and transformations
"""

from collections.abc import Mapping
from typing import Final

import httpx

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.evals.transformation import (
    BaseEvalsAPIConfig,
    LiteLLMLoggingObj,
)
from litellm.types.llms.openai_evals import (
    CancelEvalResponse,
    CancelRunResponse,
    CreateEvalRequest,
    CreateRunRequest,
    DeleteEvalResponse,
    Eval,
    ListEvalsParams,
    ListEvalsResponse,
    ListRunsParams,
    ListRunsResponse,
    Run,
    RunDeleteResponse,
    UpdateEvalRequest,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


def _parsed_response_json(raw_response: httpx.Response) -> Mapping[str, object]:
    return raw_response.json()


class OpenAIEvalsConfig(BaseEvalsAPIConfig):
    """OpenAI-specific Evals API configuration"""

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.OPENAI

    def validate_environment(self, headers: dict, litellm_params: GenericLiteLLMParams | None) -> dict:
        """Add OpenAI-specific headers"""
        import litellm
        from litellm.secret_managers.main import get_secret_str

        # Get API key following OpenAI pattern
        api_key = None
        if litellm_params:
            api_key = litellm_params.api_key

        api_key = api_key or litellm.api_key or litellm.openai_key or get_secret_str("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for Evals API")

        # Add required headers
        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"

        return headers

    def get_complete_url(
        self,
        api_base: str | None,
        endpoint: str,
        eval_id: str | None = None,
    ) -> str:
        """Get complete URL for OpenAI Evals API"""
        if api_base is None:
            api_base = "https://api.openai.com"

        if eval_id:
            encoded_eval_id: Final = encode_url_path_segment(eval_id, field_name="eval_id")
            return f"{api_base}/v1/evals/{encoded_eval_id}"
        return f"{api_base}/v1/{endpoint}"

    def transform_create_eval_request(
        self,
        create_request: CreateEvalRequest,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        """Transform create eval request for OpenAI"""
        verbose_logger.debug("Transforming create eval request: %s", create_request)

        # OpenAI expects the request body directly
        request_body: Final = {k: v for k, v in create_request.items() if v is not None}

        return request_body

    def transform_create_eval_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Eval:
        """Transform OpenAI response to Eval object"""
        response_json: Final = _parsed_response_json(raw_response)
        verbose_logger.debug("Transforming create eval response: %s", response_json)

        return Eval.model_validate(response_json)

    def transform_list_evals_request(
        self,
        list_params: ListEvalsParams,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        """Transform list evals request for OpenAI"""
        api_base = "https://api.openai.com"
        if litellm_params and litellm_params.api_base:
            api_base = litellm_params.api_base

        url: Final = self.get_complete_url(api_base=api_base, endpoint="evals")

        # Build query parameters
        query_params: Final[dict[str, object]] = {}
        if "limit" in list_params and list_params["limit"]:
            query_params["limit"] = list_params["limit"]
        if "after" in list_params and list_params["after"]:
            query_params["after"] = list_params["after"]
        if "before" in list_params and list_params["before"]:
            query_params["before"] = list_params["before"]
        if "order" in list_params and list_params["order"]:
            query_params["order"] = list_params["order"]
        if "order_by" in list_params and list_params["order_by"]:
            query_params["order_by"] = list_params["order_by"]

        verbose_logger.debug(
            "List evals request made to OpenAI Evals endpoint with params: %s",
            query_params,
        )

        return url, query_params

    def transform_list_evals_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ListEvalsResponse:
        """Transform OpenAI response to ListEvalsResponse"""
        response_json: Final = _parsed_response_json(raw_response)
        verbose_logger.debug("Transforming list evals response: %s", response_json)

        return ListEvalsResponse.model_validate(response_json)

    def transform_get_eval_request(
        self,
        eval_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        """Transform get eval request for OpenAI"""
        url: Final = self.get_complete_url(api_base=api_base, endpoint="evals", eval_id=eval_id)

        verbose_logger.debug("Get eval request - URL: %s", url)

        return url, headers

    def transform_get_eval_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Eval:
        """Transform OpenAI response to Eval object"""
        response_json: Final = _parsed_response_json(raw_response)
        verbose_logger.debug("Transforming get eval response: %s", response_json)

        return Eval.model_validate(response_json)

    def transform_update_eval_request(
        self,
        eval_id: str,
        update_request: UpdateEvalRequest,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict, dict]:
        """Transform update eval request for OpenAI"""
        url: Final = self.get_complete_url(api_base=api_base, endpoint="evals", eval_id=eval_id)

        # Build request body
        request_body: Final = {k: v for k, v in update_request.items() if v is not None}

        verbose_logger.debug("Update eval request - URL: %s, body: %s", url, request_body)

        return url, headers, request_body

    def transform_update_eval_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Eval:
        """Transform OpenAI response to Eval object"""
        response_json: Final = _parsed_response_json(raw_response)
        verbose_logger.debug("Transforming update eval response: %s", response_json)

        return Eval.model_validate(response_json)

    def transform_delete_eval_request(
        self,
        eval_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        """Transform delete eval request for OpenAI"""
        url: Final = self.get_complete_url(api_base=api_base, endpoint="evals", eval_id=eval_id)

        verbose_logger.debug("Delete eval request - URL: %s", url)

        return url, headers

    def transform_delete_eval_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> DeleteEvalResponse:
        """Transform OpenAI response to DeleteEvalResponse"""
        response_json: Final = _parsed_response_json(raw_response)
        verbose_logger.debug("Transforming delete eval response: %s", response_json)

        return DeleteEvalResponse.model_validate(response_json)

    def transform_cancel_eval_request(
        self,
        eval_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict, dict]:
        """Transform cancel eval request for OpenAI"""
        url: Final = f"{self.get_complete_url(api_base=api_base, endpoint='evals', eval_id=eval_id)}/cancel"

        # Empty body for cancel request
        request_body: Final[dict[str, object]] = {}

        verbose_logger.debug("Cancel eval request - URL: %s", url)

        return url, headers, request_body

    def transform_cancel_eval_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> CancelEvalResponse:
        """Transform OpenAI response to CancelEvalResponse"""
        response_json: Final = _parsed_response_json(raw_response)
        verbose_logger.debug("Transforming cancel eval response: %s", response_json)

        return CancelEvalResponse.model_validate(response_json)

    # Run API Transformations
    def transform_create_run_request(
        self,
        eval_id: str,
        create_request: CreateRunRequest,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        """Transform create run request for OpenAI"""
        api_base = "https://api.openai.com"
        if litellm_params and litellm_params.api_base:
            api_base = litellm_params.api_base

        encoded_eval_id: Final = encode_url_path_segment(eval_id, field_name="eval_id")
        url: Final = f"{api_base}/v1/evals/{encoded_eval_id}/runs"

        # Build request body
        request_body: Final = {k: v for k, v in create_request.items() if v is not None}

        verbose_logger.debug("Create run request - URL: %s, body: %s", url, request_body)

        return url, request_body

    def transform_create_run_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Run:
        """Transform OpenAI response to Run object"""
        response_json: Final = _parsed_response_json(raw_response)
        verbose_logger.debug("Transforming create run response: %s", response_json)

        return Run.model_validate(response_json)

    def transform_list_runs_request(
        self,
        eval_id: str,
        list_params: ListRunsParams,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        """Transform list runs request for OpenAI"""
        api_base = "https://api.openai.com"
        if litellm_params and litellm_params.api_base:
            api_base = litellm_params.api_base

        encoded_eval_id: Final = encode_url_path_segment(eval_id, field_name="eval_id")
        url: Final = f"{api_base}/v1/evals/{encoded_eval_id}/runs"

        # Build query parameters
        query_params: Final[dict[str, object]] = {}
        if "limit" in list_params and list_params["limit"]:
            query_params["limit"] = list_params["limit"]
        if "after" in list_params and list_params["after"]:
            query_params["after"] = list_params["after"]
        if "before" in list_params and list_params["before"]:
            query_params["before"] = list_params["before"]
        if "order" in list_params and list_params["order"]:
            query_params["order"] = list_params["order"]

        verbose_logger.debug(
            "List runs request made to OpenAI Evals endpoint with params: %s",
            query_params,
        )

        return url, query_params

    def transform_list_runs_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ListRunsResponse:
        """Transform OpenAI response to ListRunsResponse"""
        response_json: Final = _parsed_response_json(raw_response)
        verbose_logger.debug("Transforming list runs response: %s", response_json)

        return ListRunsResponse.model_validate(response_json)

    def transform_get_run_request(
        self,
        eval_id: str,
        run_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        """Transform get run request for OpenAI"""
        encoded_eval_id: Final = encode_url_path_segment(eval_id, field_name="eval_id")
        encoded_run_id: Final = encode_url_path_segment(run_id, field_name="run_id")
        url: Final = f"{api_base}/v1/evals/{encoded_eval_id}/runs/{encoded_run_id}"

        verbose_logger.debug("Get run request - URL: %s", url)

        return url, headers

    def transform_get_run_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Run:
        """Transform OpenAI response to Run object"""
        response_json: Final = _parsed_response_json(raw_response)
        verbose_logger.debug("Transforming get run response: %s", response_json)

        return Run.model_validate(response_json)

    def transform_cancel_run_request(
        self,
        eval_id: str,
        run_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict, dict]:
        """Transform cancel run request for OpenAI"""
        encoded_eval_id: Final = encode_url_path_segment(eval_id, field_name="eval_id")
        encoded_run_id: Final = encode_url_path_segment(run_id, field_name="run_id")
        url: Final = f"{api_base}/v1/evals/{encoded_eval_id}/runs/{encoded_run_id}/cancel"

        # Empty body for cancel request
        request_body: Final[dict[str, object]] = {}

        verbose_logger.debug("Cancel run request - URL: %s", url)

        return url, headers, request_body

    def transform_cancel_run_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> CancelRunResponse:
        """Transform OpenAI response to CancelRunResponse"""
        response_json: Final = _parsed_response_json(raw_response)
        verbose_logger.debug("Transforming cancel run response: %s", response_json)

        return CancelRunResponse.model_validate(response_json)

    def transform_delete_run_request(
        self,
        eval_id: str,
        run_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict, dict]:
        """Transform delete run request for OpenAI"""
        encoded_eval_id: Final = encode_url_path_segment(eval_id, field_name="eval_id")
        encoded_run_id: Final = encode_url_path_segment(run_id, field_name="run_id")
        url: Final = f"{api_base}/v1/evals/{encoded_eval_id}/runs/{encoded_run_id}"

        # Empty body for delete request
        request_body: Final[dict[str, object]] = {}

        verbose_logger.debug("Delete run request - URL: %s", url)

        return url, headers, request_body

    def transform_delete_run_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> RunDeleteResponse:
        """Transform OpenAI response to RunDeleteResponse"""
        response_json: Final = _parsed_response_json(raw_response)
        verbose_logger.debug("Transforming delete run response: %s", response_json)

        return RunDeleteResponse.model_validate(response_json)
