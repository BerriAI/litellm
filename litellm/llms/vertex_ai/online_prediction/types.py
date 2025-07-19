"""
Type definitions for Vertex AI Online Prediction
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class VertexAIOnlinePredictionRequest(BaseModel):
    """Request format for Vertex AI online prediction"""
    
    instances: List[Dict[str, Any]] = Field(
        description="The instances to be predicted"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional parameters for the prediction"
    )


class VertexAIOnlinePredictionResponse(BaseModel):
    """Response format for Vertex AI online prediction"""
    
    predictions: List[Dict[str, Any]] = Field(
        description="The predictions returned by the model"
    )
    deployed_model_id: Optional[str] = Field(
        default=None,
        description="The ID of the deployed model"
    )
    model: Optional[str] = Field(
        default=None,
        description="The name of the model"
    )
    model_version_id: Optional[str] = Field(
        default=None,
        description="The version ID of the model"
    )
    model_display_name: Optional[str] = Field(
        default=None,
        description="The display name of the model"
    )


class VertexAIEndpointConfig(BaseModel):
    """Configuration for Vertex AI endpoint"""
    
    project_id: str = Field(description="Google Cloud project ID")
    location: str = Field(description="Vertex AI location")
    endpoint_id: str = Field(description="Endpoint ID")
    model_name: Optional[str] = Field(
        default=None,
        description="Model name (for display purposes)"
    )


class VertexAIOnlinePredictionParams(BaseModel):
    """Parameters for online prediction"""
    
    temperature: Optional[float] = Field(
        default=None,
        description="Temperature for text generation"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate"
    )
    top_p: Optional[float] = Field(
        default=None,
        description="Top-p sampling parameter"
    )
    top_k: Optional[int] = Field(
        default=None,
        description="Top-k sampling parameter"
    )
    stop_sequences: Optional[List[str]] = Field(
        default=None,
        description="Stop sequences for text generation"
    )
    candidate_count: Optional[int] = Field(
        default=None,
        description="Number of candidates to generate"
    )


class VertexAIOnlinePredictionError(BaseModel):
    """Error response from Vertex AI online prediction"""
    
    error: Dict[str, Any] = Field(description="Error details")
    code: int = Field(description="HTTP status code")


# Type aliases for convenience
OnlinePredictionRequest = VertexAIOnlinePredictionRequest
OnlinePredictionResponse = VertexAIOnlinePredictionResponse
EndpointConfig = VertexAIEndpointConfig
PredictionParams = VertexAIOnlinePredictionParams
PredictionError = VertexAIOnlinePredictionError 