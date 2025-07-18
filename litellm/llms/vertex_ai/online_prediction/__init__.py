"""
Vertex AI Online Prediction Module

This module provides support for Vertex AI Online Prediction endpoints,
enabling real-time inference on custom models deployed on Google Cloud.
"""

from .handler import VertexAIOnlinePredictionHandler
from .transformation import VertexAIOnlinePredictionTransformation
from .types import (
    EndpointConfig,
    OnlinePredictionRequest,
    OnlinePredictionResponse,
    PredictionParams,
    PredictionError,
)

__all__ = [
    "VertexAIOnlinePredictionHandler",
    "VertexAIOnlinePredictionTransformation",
    "EndpointConfig",
    "OnlinePredictionRequest",
    "OnlinePredictionResponse",
    "PredictionParams",
    "PredictionError",
] 