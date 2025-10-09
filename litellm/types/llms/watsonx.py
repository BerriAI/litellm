import json
from enum import Enum
from typing import Any, List, Optional, Union

from pydantic import BaseModel
from typing_extensions import TypedDict


class WatsonXAPIParams(TypedDict):
    project_id: str
    space_id: Optional[str]
    region_name: Optional[str]


class WatsonXCredentials(TypedDict):
    api_key: str
    api_base: str
    token: Optional[str]


class WatsonXAIEndpoint(str, Enum):
    TEXT_GENERATION = "/ml/v1/text/generation"
    TEXT_GENERATION_STREAM = "/ml/v1/text/generation_stream"
    CHAT = "/ml/v1/text/chat"
    CHAT_STREAM = "/ml/v1/text/chat_stream"
    DEPLOYMENT_TEXT_GENERATION = "/ml/v1/deployments/{deployment_id}/text/generation"
    DEPLOYMENT_TEXT_GENERATION_STREAM = (
        "/ml/v1/deployments/{deployment_id}/text/generation_stream"
    )
    DEPLOYMENT_CHAT = "/ml/v1/deployments/{deployment_id}/text/chat"
    DEPLOYMENT_CHAT_STREAM = "/ml/v1/deployments/{deployment_id}/text/chat_stream"
    EMBEDDINGS = "/ml/v1/text/embeddings"
    PROMPTS = "/ml/v1/prompts"
    AVAILABLE_MODELS = "/ml/v1/foundation_model_specs"


class WatsonXModelPattern(str, Enum):
    """Model identifier patterns for WatsonX models"""
    GRANITE_CHAT = "granite-chat"
    IBM_MISTRAL = "ibm-mistral"
    IBM_MISTRALAI = "ibm-mistralai"
    GPT_OSS = "openai/gpt-oss"
    LLAMA3_INSTRUCT = "meta-llama/llama-3-instruct"
