"""
Handler for Vertex AI Online Prediction
"""

import traceback
from typing import Any, Coroutine, Dict, Optional, Union

import litellm
from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import HTTPHandler, get_async_httpx_client
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import VertexLLM
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES
from litellm.types.utils import ModelResponse

from .transformation import VertexAIOnlinePredictionTransformation
from .types import OnlinePredictionRequest, OnlinePredictionResponse


class VertexAIOnlinePredictionHandler:
    """
    Handler for Vertex AI Online Prediction endpoints
    """

    def __init__(self) -> None:
        self.vertex_llm = VertexLLM()
        self.async_handler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.VERTEX_AI,
            params={"timeout": 60.0},  # Shorter timeout for online prediction
        )

    def completion(
        self,
        model: str,
        messages: list,
        model_response: ModelResponse,
        print_verbose,
        encoding,
        logging_obj,
        optional_params: dict,
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES] = None,
        litellm_params: Optional[dict] = None,
        logger_fn=None,
        acompletion: bool = False,
        **kwargs,
    ) -> Union[ModelResponse, Coroutine[Any, Any, ModelResponse]]:
        """
        Handle completion requests for Vertex AI online prediction
        """
        try:
            # Parse endpoint configuration from model
            endpoint_config = VertexAIOnlinePredictionTransformation.parse_endpoint_from_model(model)
            endpoint_config = VertexAIOnlinePredictionTransformation.validate_endpoint_config(
                endpoint_config, vertex_project, vertex_location
            )

            # Get authentication
            access_token, project_id = self.vertex_llm._ensure_access_token(
                credentials=vertex_credentials,
                project_id=vertex_project or endpoint_config.project_id,
                custom_llm_provider="vertex_ai",
            )

            # Transform messages to instances
            instances = VertexAIOnlinePredictionTransformation.transform_messages_to_instances(
                messages, model, optional_params
            )

            # Transform optional parameters to prediction parameters
            prediction_params = VertexAIOnlinePredictionTransformation.transform_optional_params_to_prediction_params(
                optional_params
            )

            # Create prediction request
            prediction_request = VertexAIOnlinePredictionTransformation.create_prediction_request(
                instances=instances,
                parameters=prediction_params.dict(exclude_none=True) if prediction_params else None
            )

            # Create prediction URL
            prediction_url = VertexAIOnlinePredictionTransformation.create_prediction_url(
                project_id=endpoint_config.project_id,
                location=endpoint_config.location,
                endpoint_id=endpoint_config.endpoint_id,
                use_raw_predict=optional_params.get("use_raw_predict", False)
            )

            # Set up headers
            headers = {
                "Content-Type": "application/json; charset=utf-8",
                "Authorization": f"Bearer {access_token}",
            }

            # Log the request
            logging_obj.pre_call(
                input=str(messages),
                api_key=None,
                additional_args={
                    "endpoint_config": endpoint_config.dict(),
                    "prediction_url": prediction_url,
                    "optional_params": optional_params,
                },
            )

            if acompletion:
                return self._async_completion(
                    prediction_url=prediction_url,
                    headers=headers,
                    prediction_request=prediction_request,
                    model=model,
                    messages=messages,
                    optional_params=optional_params,
                    model_response=model_response,
                )

            # Make synchronous request
            sync_handler = HTTPHandler()
            response = sync_handler.post(
                url=prediction_url,
                headers=headers,
                json=prediction_request.dict(),
            )

            if response.status_code != 200:
                error_message = VertexAIOnlinePredictionTransformation.extract_error_from_response(
                    response.text
                )
                raise Exception(
                    f"Vertex AI Online Prediction failed. Status: {response.status_code}. Error: {error_message}"
                )

            # Parse response
            response_data = response.json()
            prediction_response = OnlinePredictionResponse(**response_data)

            # Transform to ModelResponse
            result = VertexAIOnlinePredictionTransformation.transform_prediction_response_to_model_response(
                prediction_response, model, messages, optional_params
            )

            # Log the response
            logging_obj.post_call(
                input=str(messages),
                api_key=None,
                original_response=result,
                additional_args={
                    "endpoint_config": endpoint_config.dict(),
                    "prediction_response": response_data,
                },
            )

            return result

        except Exception as e:
            verbose_logger.error(
                "Vertex AI Online Prediction error: %s\n%s",
                str(e),
                traceback.format_exc(),
            )
            raise e

    async def _async_completion(
        self,
        prediction_url: str,
        headers: Dict[str, str],
        prediction_request: OnlinePredictionRequest,
        model: str,
        messages: list,
        optional_params: dict,
        model_response: ModelResponse,
    ) -> ModelResponse:
        """
        Handle async completion requests for Vertex AI online prediction
        """
        try:
            if self.async_handler is None:
                raise ValueError("Async handler is not initialized")

            response = await self.async_handler.post(
                url=prediction_url,
                headers=headers,
                json=prediction_request.dict(),
            )

            if response.status_code != 200:
                error_message = VertexAIOnlinePredictionTransformation.extract_error_from_response(
                    response.text
                )
                raise Exception(
                    f"Vertex AI Online Prediction failed. Status: {response.status_code}. Error: {error_message}"
                )

            # Parse response
            response_data = response.json()
            prediction_response = OnlinePredictionResponse(**response_data)

            # Transform to ModelResponse
            result = VertexAIOnlinePredictionTransformation.transform_prediction_response_to_model_response(
                prediction_response, model, messages, optional_params
            )

            return result

        except Exception as e:
            verbose_logger.error(
                "Vertex AI Online Prediction async error: %s\n%s",
                str(e),
                traceback.format_exc(),
            )
            raise e

    def get_endpoint_info(
        self,
        endpoint_id: str,
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES] = None,
    ) -> Dict[str, Any]:
        """
        Get information about a Vertex AI endpoint
        """
        try:
            # Get authentication
            access_token, project_id = self._ensure_access_token(
                credentials=vertex_credentials,
                project_id=vertex_project,
                custom_llm_provider="vertex_ai",
            )

            # Create endpoint URL
            location = vertex_location or "us-central1"
            endpoint_url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"

            # Set up headers
            headers = {
                "Authorization": f"Bearer {access_token}",
            }

            # Make request
            sync_handler = HTTPHandler()
            response = sync_handler.get(
                url=endpoint_url,
                headers=headers,
            )

            if response.status_code != 200:
                error_message = VertexAIOnlinePredictionTransformation.extract_error_from_response(
                    response.text
                )
                raise Exception(
                    f"Failed to get endpoint info. Status: {response.status_code}. Error: {error_message}"
                )

            return response.json()

        except Exception as e:
            verbose_logger.error(
                "Error getting endpoint info: %s\n%s",
                str(e),
                traceback.format_exc(),
            )
            raise e

    def list_endpoints(
        self,
        vertex_project: Optional[str] = None,
        vertex_location: Optional[str] = None,
        vertex_credentials: Optional[VERTEX_CREDENTIALS_TYPES] = None,
    ) -> Dict[str, Any]:
        """
        List Vertex AI endpoints
        """
        try:
            # Get authentication
            access_token, project_id = self._ensure_access_token(
                credentials=vertex_credentials,
                project_id=vertex_project,
                custom_llm_provider="vertex_ai",
            )

            # Create endpoints URL
            location = vertex_location or "us-central1"
            endpoints_url = f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints"

            # Set up headers
            headers = {
                "Authorization": f"Bearer {access_token}",
            }

            # Make request
            sync_handler = HTTPHandler()
            response = sync_handler.get(
                url=endpoints_url,
                headers=headers,
            )

            if response.status_code != 200:
                error_message = VertexAIOnlinePredictionTransformation.extract_error_from_response(
                    response.text
                )
                raise Exception(
                    f"Failed to list endpoints. Status: {response.status_code}. Error: {error_message}"
                )

            return response.json()

        except Exception as e:
            verbose_logger.error(
                "Error listing endpoints: %s\n%s",
                str(e),
                traceback.format_exc(),
            )
            raise e 