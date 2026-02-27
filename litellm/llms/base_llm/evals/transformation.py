"""
Base configuration class for Evals API
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
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

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class BaseEvalsAPIConfig(ABC):
    """Base configuration for Evals API providers"""

    def __init__(self):
        pass

    @property
    @abstractmethod
    def custom_llm_provider(self) -> LlmProviders:
        pass

    @abstractmethod
    def validate_environment(
        self, headers: dict, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        """
        Validate and update headers with provider-specific requirements

        Args:
            headers: Base headers dictionary
            litellm_params: LiteLLM parameters

        Returns:
            Updated headers dictionary
        """
        return headers

    @abstractmethod
    def get_complete_url(
        self,
        api_base: Optional[str],
        endpoint: str,
        eval_id: Optional[str] = None,
    ) -> str:
        """
        Get the complete URL for the API request

        Args:
            api_base: Base API URL
            endpoint: API endpoint (e.g., 'evals', 'evals/{id}')
            eval_id: Optional eval ID for specific eval operations

        Returns:
            Complete URL
        """
        if api_base is None:
            raise ValueError("api_base is required")
        return f"{api_base}/v1/{endpoint}"

    @abstractmethod
    def transform_create_eval_request(
        self,
        create_request: CreateEvalRequest,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        """
        Transform create eval request to provider-specific format

        Args:
            create_request: Eval creation parameters
            litellm_params: LiteLLM parameters
            headers: Request headers

        Returns:
            Provider-specific request body
        """
        pass

    @abstractmethod
    def transform_create_eval_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Eval:
        """
        Transform provider response to Eval object

        Args:
            raw_response: Raw HTTP response
            logging_obj: Logging object

        Returns:
            Eval object
        """
        pass

    @abstractmethod
    def transform_list_evals_request(
        self,
        list_params: ListEvalsParams,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform list evals request parameters

        Args:
            list_params: List parameters (pagination, filters)
            litellm_params: LiteLLM parameters
            headers: Request headers

        Returns:
            Tuple of (url, query_params)
        """
        pass

    @abstractmethod
    def transform_list_evals_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ListEvalsResponse:
        """
        Transform provider response to ListEvalsResponse

        Args:
            raw_response: Raw HTTP response
            logging_obj: Logging object

        Returns:
            ListEvalsResponse object
        """
        pass

    @abstractmethod
    def transform_get_eval_request(
        self,
        eval_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform get eval request

        Args:
            eval_id: Eval ID
            api_base: Base API URL
            litellm_params: LiteLLM parameters
            headers: Request headers

        Returns:
            Tuple of (url, headers)
        """
        pass

    @abstractmethod
    def transform_get_eval_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Eval:
        """
        Transform provider response to Eval object

        Args:
            raw_response: Raw HTTP response
            logging_obj: Logging object

        Returns:
            Eval object
        """
        pass

    @abstractmethod
    def transform_update_eval_request(
        self,
        eval_id: str,
        update_request: UpdateEvalRequest,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict, Dict]:
        """
        Transform update eval request

        Args:
            eval_id: Eval ID
            update_request: Update parameters
            api_base: Base API URL
            litellm_params: LiteLLM parameters
            headers: Request headers

        Returns:
            Tuple of (url, headers, body)
        """
        pass

    @abstractmethod
    def transform_update_eval_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Eval:
        """
        Transform provider response to Eval object

        Args:
            raw_response: Raw HTTP response
            logging_obj: Logging object

        Returns:
            Eval object
        """
        pass

    @abstractmethod
    def transform_delete_eval_request(
        self,
        eval_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform delete eval request

        Args:
            eval_id: Eval ID
            api_base: Base API URL
            litellm_params: LiteLLM parameters
            headers: Request headers

        Returns:
            Tuple of (url, headers)
        """
        pass

    @abstractmethod
    def transform_delete_eval_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> DeleteEvalResponse:
        """
        Transform provider response to DeleteEvalResponse

        Args:
            raw_response: Raw HTTP response
            logging_obj: Logging object

        Returns:
            DeleteEvalResponse object
        """
        pass

    @abstractmethod
    def transform_cancel_eval_request(
        self,
        eval_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict, Dict]:
        """
        Transform cancel eval request

        Args:
            eval_id: Eval ID
            api_base: Base API URL
            litellm_params: LiteLLM parameters
            headers: Request headers

        Returns:
            Tuple of (url, headers, body)
        """
        pass

    @abstractmethod
    def transform_cancel_eval_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> CancelEvalResponse:
        """
        Transform provider response to CancelEvalResponse

        Args:
            raw_response: Raw HTTP response
            logging_obj: Logging object

        Returns:
            CancelEvalResponse object
        """
        pass

    # Run API Transformations
    @abstractmethod
    def transform_create_run_request(
        self,
        eval_id: str,
        create_request: CreateRunRequest,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform create run request to provider-specific format

        Args:
            eval_id: Eval ID
            create_request: Run creation parameters
            litellm_params: LiteLLM parameters
            headers: Request headers

        Returns:
            Tuple of (url, request_body)
        """
        pass

    @abstractmethod
    def transform_create_run_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Run:
        """
        Transform provider response to Run object

        Args:
            raw_response: Raw HTTP response
            logging_obj: Logging object

        Returns:
            Run object
        """
        pass

    @abstractmethod
    def transform_list_runs_request(
        self,
        eval_id: str,
        list_params: ListRunsParams,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform list runs request parameters

        Args:
            eval_id: Eval ID
            list_params: List parameters (pagination, filters)
            litellm_params: LiteLLM parameters
            headers: Request headers

        Returns:
            Tuple of (url, query_params)
        """
        pass

    @abstractmethod
    def transform_list_runs_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ListRunsResponse:
        """
        Transform provider response to ListRunsResponse

        Args:
            raw_response: Raw HTTP response
            logging_obj: Logging object

        Returns:
            ListRunsResponse object
        """
        pass

    @abstractmethod
    def transform_get_run_request(
        self,
        eval_id: str,
        run_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform get run request

        Args:
            eval_id: Eval ID
            run_id: Run ID
            api_base: Base API URL
            litellm_params: LiteLLM parameters
            headers: Request headers

        Returns:
            Tuple of (url, headers)
        """
        pass

    @abstractmethod
    def transform_get_run_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Run:
        """
        Transform provider response to Run object

        Args:
            raw_response: Raw HTTP response
            logging_obj: Logging object

        Returns:
            Run object
        """
        pass

    @abstractmethod
    def transform_cancel_run_request(
        self,
        eval_id: str,
        run_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict, Dict]:
        """
        Transform cancel run request

        Args:
            eval_id: Eval ID
            run_id: Run ID
            api_base: Base API URL
            litellm_params: LiteLLM parameters
            headers: Request headers

        Returns:
            Tuple of (url, headers, body)
        """
        pass

    @abstractmethod
    def transform_cancel_run_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> CancelRunResponse:
        """
        Transform provider response to CancelRunResponse

        Args:
            raw_response: Raw HTTP response
            logging_obj: Logging object

        Returns:
            CancelRunResponse object
        """
        pass

    @abstractmethod
    def transform_delete_run_request(
        self,
        eval_id: str,
        run_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict, Dict]:
        """
        Transform delete run request

        Args:
            eval_id: Eval ID
            run_id: Run ID
            api_base: Base API URL
            litellm_params: LiteLLM parameters
            headers: Request headers

        Returns:
            Tuple of (url, headers, body)
        """
        pass

    @abstractmethod
    def transform_delete_run_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> "RunDeleteResponse":
        """
        Transform provider response to RunDeleteResponse

        Args:
            raw_response: Raw HTTP response
            logging_obj: Logging object

        Returns:
            RunDeleteResponse object
        """
        pass

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict,
    ) -> Exception:
        """Get appropriate error class for the provider."""
        return BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
