from enum import Enum
from typing import List, Optional

from typing_extensions import NotRequired, TypedDict


class WatsonXAPIParams(TypedDict):
    project_id: str
    space_id: Optional[str]
    region_name: Optional[str]


class WatsonXCredentials(TypedDict):
    api_key: str
    api_base: str
    token: Optional[str]


class WatsonXAudioTranscriptionRequestBody(TypedDict):
    """
    WatsonX Audio Transcription API request body.
    
    Follows multipart/form-data format for WatsonX Whisper models.
    See: https://cloud.ibm.com/apidocs/watsonx-ai
    """

    model: str
    """Model name (e.g., 'whisper-large-v3-turbo')"""

    project_id: str
    """WatsonX project ID (required)"""

    language: NotRequired[str]
    """Language code (e.g., 'en', 'es')"""

    prompt: NotRequired[str]
    """Optional prompt to guide transcription"""

    response_format: NotRequired[str]
    """Response format: 'json', 'text', 'srt', 'verbose_json', 'vtt'"""

    temperature: NotRequired[float]
    """Sampling temperature (0-1)"""

    timestamp_granularities: NotRequired[List[str]]
    """Timestamp granularities: ['word', 'segment']"""


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
