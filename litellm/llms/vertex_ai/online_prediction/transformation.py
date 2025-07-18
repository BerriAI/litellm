"""
Transformation logic for Vertex AI Online Prediction
"""

import json
import re
from typing import Any, Dict, List, Optional, Union

import litellm
from litellm._logging import verbose_logger
from litellm.types.llms.vertex_ai import VERTEX_CREDENTIALS_TYPES
from litellm.types.utils import ModelResponse

from .types import (
    EndpointConfig,
    OnlinePredictionRequest,
    OnlinePredictionResponse,
    PredictionParams,
)


class VertexAIOnlinePredictionTransformation:
    """Transformation logic for Vertex AI online prediction"""

    @staticmethod
    def parse_endpoint_from_model(model: str) -> EndpointConfig:
        """
        Parse endpoint configuration from model string
        
        Expected format: vertex_ai/endpoints/{endpoint_id}
        or: vertex_ai/{project_id}/{location}/endpoints/{endpoint_id}
        """
        if not model.startswith("vertex_ai/"):
            raise ValueError(f"Invalid model format: {model}")
        
        parts = model.split("/")
        
        if len(parts) == 3 and parts[1] == "endpoints":
            # Format: vertex_ai/endpoints/{endpoint_id}
            return EndpointConfig(
                project_id="",  # Will be set from environment or params
                location="",    # Will be set from environment or params
                endpoint_id=parts[2]
            )
        elif len(parts) == 5 and parts[3] == "endpoints":
            # Format: vertex_ai/{project_id}/{location}/endpoints/{endpoint_id}
            return EndpointConfig(
                project_id=parts[1],
                location=parts[2],
                endpoint_id=parts[4]
            )
        else:
            raise ValueError(f"Invalid model format: {model}")

    @staticmethod
    def transform_messages_to_instances(
        messages: List[Dict[str, Any]],
        model: str,
        optional_params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Transform LiteLLM messages to Vertex AI instances format
        """
        # Convert messages to prompt format
        prompt = litellm.prompt_factory(model=model, messages=messages)
        
        # Create instance format
        instance = {
            "prompt": prompt
        }
        
        # Add any additional instance parameters
        if "instance_params" in optional_params:
            instance.update(optional_params["instance_params"])
        
        return [instance]

    @staticmethod
    def transform_optional_params_to_prediction_params(
        optional_params: Dict[str, Any]
    ) -> PredictionParams:
        """
        Transform LiteLLM optional parameters to Vertex AI prediction parameters
        """
        prediction_params = {}
        
        # Map common parameters
        param_mapping = {
            "temperature": "temperature",
            "max_tokens": "max_tokens",
            "top_p": "top_p",
            "top_k": "top_k",
            "stop": "stop_sequences",
            "candidate_count": "candidate_count"
        }
        
        for litellm_param, vertex_param in param_mapping.items():
            if litellm_param in optional_params:
                prediction_params[vertex_param] = optional_params[litellm_param]
        
        # Handle stop sequences
        if "stop" in optional_params:
            stop = optional_params["stop"]
            if isinstance(stop, str):
                prediction_params["stop_sequences"] = [stop]
            elif isinstance(stop, list):
                prediction_params["stop_sequences"] = stop
        
        return PredictionParams(**prediction_params)

    @staticmethod
    def create_prediction_request(
        instances: List[Dict[str, Any]],
        parameters: Optional[Dict[str, Any]] = None
    ) -> OnlinePredictionRequest:
        """
        Create Vertex AI prediction request
        """
        return OnlinePredictionRequest(
            instances=instances,
            parameters=parameters
        )

    @staticmethod
    def transform_prediction_response_to_model_response(
        response: OnlinePredictionResponse,
        model: str,
        messages: List[Dict[str, Any]],
        optional_params: Dict[str, Any]
    ) -> ModelResponse:
        """
        Transform Vertex AI prediction response to LiteLLM ModelResponse
        """
        if not response.predictions:
            raise ValueError("No predictions in response")
        
        # Extract the first prediction
        prediction = response.predictions[0]
        
        # Extract text content from prediction
        # The exact format depends on the model, but typically it's in a 'predictions' field
        if isinstance(prediction, dict):
            if "predictions" in prediction:
                content = prediction["predictions"]
            elif "generated_text" in prediction:
                content = prediction["generated_text"]
            elif "text" in prediction:
                content = prediction["text"]
            else:
                # Try to find any string value
                content = str(prediction)
        else:
            content = str(prediction)
        
        # Create ModelResponse
        model_response = ModelResponse()
        model_response.model = model
        model_response.choices = [
            {
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop"
            }
        ]
        
        # Add usage information if available
        if hasattr(response, 'deployed_model_id') and response.deployed_model_id:
            model_response.model = response.deployed_model_id
        
        return model_response

    @staticmethod
    def create_prediction_url(
        project_id: str,
        location: str,
        endpoint_id: str,
        use_raw_predict: bool = False
    ) -> str:
        """
        Create the prediction URL for Vertex AI endpoint
        """
        base_url = f"https://{location}-aiplatform.googleapis.com"
        endpoint_path = f"v1/projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"
        
        if use_raw_predict:
            return f"{base_url}/{endpoint_path}:rawPredict"
        else:
            return f"{base_url}/{endpoint_path}:predict"

    @staticmethod
    def validate_endpoint_config(
        endpoint_config: EndpointConfig,
        vertex_project: Optional[str],
        vertex_location: Optional[str]
    ) -> EndpointConfig:
        """
        Validate and fill in missing endpoint configuration
        """
        if not endpoint_config.project_id:
            if not vertex_project:
                raise ValueError("Project ID is required for Vertex AI online prediction")
            endpoint_config.project_id = vertex_project
        
        if not endpoint_config.location:
            if not vertex_location:
                endpoint_config.location = "us-central1"  # Default location
            else:
                endpoint_config.location = vertex_location
        
        return endpoint_config

    @staticmethod
    def extract_error_from_response(response_text: str) -> str:
        """
        Extract error message from Vertex AI response
        """
        try:
            error_data = json.loads(response_text)
            if "error" in error_data:
                return error_data["error"].get("message", "Unknown error")
            return response_text
        except json.JSONDecodeError:
            return response_text 